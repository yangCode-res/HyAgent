import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

from Core.Agent import Agent
from Logger.index import get_global_logger
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from Memory.index import Memory


class MechanismExtractionAgent(Agent):
    """
    只做一件事：为已有 KGTriple 补机制。

    约定（跟你给的 JSON 完全对齐）：
    - 每个 Subgraph 里：
        meta: {
          "text": {
             "id": "PMID_xxx",
             "text": "这一整段文本"
          }
        }
        relations: [KGTriple, KGTriple, ...]
    - 对同一子图内的所有 triples：
        使用同一个 meta.text.text 作为 context。
    - 子图顺序串行，子图内部 triple 并发。
    """

    def __init__(self, client: OpenAI, model_name: str) -> None:
        system_prompt = """
You are a dedicated Mechanism Extraction Agent for biomedical knowledge graphs.

INPUT:
- One biomedical text passage (context).
- One existing triple: (head, relation, tail).

TASK:
For the given triple, infer and summarize the underlying biological / pharmacological / molecular mechanism
that explains HOW or WHY the head influences the tail under the given relation.

OUTPUT:
Return EXACTLY ONE JSON object with keys:

{
  "head": "exact_head",
  "relation": "RELATIONSHIP_TYPE",
  "tail": "exact_tail",
  "mechanism": "50-120 words mechanistic explanation in English, if reliably supported or well-established; otherwise empty string",
  "evidence": "short quote or faithful paraphrase from the given text that supports this mechanism, or empty string",
  "confidence": 0.0-1.0
}

RULES:
1. (head, relation, tail) in your output MUST EXACTLY match the input triple.
2. Use ONLY the provided text as primary evidence; you MAY rely on well-known biomedical knowledge
   if it is clearly consistent and non-contradictory.
3. If mechanism is uncertain or not clearly supported, set mechanism="" and confidence<=0.4.
4. Do NOT invent new entities or relations. Do NOT change head/relation/tail.
5. Do NOT wrap the JSON in any extra text.
"""
        super().__init__(client, model_name, system_prompt)
        self.logger = get_global_logger()

    # ===================== 对外入口 =====================

    def process(
        self,
        memory: Memory,
        max_workers: int = 8,
    ) -> None:
        """
        遍历 memory.subgraphs：
        - 每个子图读取 meta.text.text 作为上下文
        - 对该子图所有 triples 并发补机制（原地修改 KGTriple）
        """
        if not memory.subgraphs:
            self.logger.info("[MechanismExtraction] no subgraphs found, nothing to do.")
            return

        total_triples = 0
        total_updated = 0

        self.logger.info(
            f"[MechanismExtraction] start. subgraphs={len(memory.subgraphs)}, "
            f"max_workers={max_workers}"
        )

        # 一个子图一个子图处理；子图内部多线程
        for sg_id, sg in memory.subgraphs.items():
            # 1. 拿文本：meta.text.text
            meta = getattr(sg, "meta", {}) or {}
            ctx_text = self._extract_text_from_meta(meta)

            if not ctx_text.strip():
                self.logger.info(
                    f"[MechanismExtraction] subgraph={sg_id} has no meta.text.text, skip."
                )
                continue

            # 2. 拿关系：子图里的 triples
            if hasattr(sg, "get_relations"):
                triples: List[KGTriple] = sg.get_relations() or []
            else:
                triples = getattr(sg, "relations", []).all() if hasattr(
                    getattr(sg, "relations", None), "all"
                ) else []

            if not triples:
                self.logger.info(
                    f"[MechanismExtraction] subgraph={sg_id} has no triples, skip."
                )
                continue

            # 3. 并发补机制
            updated = self._process_subgraph_triples(
                sg_id=sg_id,
                text=ctx_text,
                triples=triples,
                max_workers=max_workers,
            )

            total_triples += len(triples)
            total_updated += updated

            self.logger.info(
                f"[MechanismExtraction] subgraph={sg_id} triples={len(triples)}, "
                f"updated={updated}"
            )

        self.logger.info(
            f"[MechanismExtraction] done. total_triples={total_triples}, "
            f"total_mechanism_updated={total_updated}"
        )

    # ===================== 从 meta 里取 text =====================

    def _extract_text_from_meta(self, meta: Dict[str, Any]) -> str:
        """
        专门适配你给的结构：
        meta: {
          "text": {
            "id": "PMID_xxx",
            "text": "......"
          }
        }
        """
        if not isinstance(meta, dict):
            return ""

        t = meta.get("text")

        # 直接就是字符串
        if isinstance(t, str):
            return t

        # 是个对象，里面有 text 字段
        if isinstance(t, dict):
            txt = t.get("text") or t.get("content") or ""
            if isinstance(txt, str):
                return txt

        return ""

    # ===================== 子图内并发处理 =====================

    def _process_subgraph_triples(
        self,
        sg_id: str,
        text: str,
        triples: List[KGTriple],
        max_workers: int,
    ) -> int:
        """
        对单个子图的 triples 并发补机制。
        """
        updated_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._enrich_single_triple, text, t): t
                for t in triples
            }

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Mechanism sg={sg_id}",
            ):
                t = futures[fut]
                try:
                    changed = fut.result()
                    if changed:
                        updated_count += 1
                except Exception as e:
                    self.logger.info(
                        f"[MechanismExtraction] error in sg={sg_id} for triple "
                        f"({getattr(t, 'head', '?')}, {getattr(t, 'relation', '?')}, {getattr(t, 'tail', '?')}): {e}"
                    )

        return updated_count

    # ===================== 单条 triple 调用 LLM =====================

    def _enrich_single_triple(
        self,
        text: str,
        triple: KGTriple,
    ) -> bool:
        """
        对单条 KGTriple 调 LLM，原地填 mechanism/evidence/confidence。

        返回是否有更新。
        """
        head = getattr(triple, "head", "").strip()
        rel = getattr(triple, "relation", "").strip()
        tail = getattr(triple, "tail", "").strip()

        if not (head and rel and tail):
            return False

        # prompt：同一段 text，不同 triple
        prompt = f'''
Biomedical text (context):
"""{text}"""

Given this triple:
{{
  "head": "{head}",
  "relation": "{rel}",
  "tail": "{tail}"
}}

According to the system instructions, return ONE JSON object:
{{
  "head": "{head}",
  "relation": "{rel}",
  "tail": "{tail}",
  "mechanism": "...",
  "evidence": "...",
  "confidence": 0.0-1.0
}}
Remember:
- Do not change head/relation/tail.
- If mechanism is not clearly supported, set mechanism="" and confidence<=0.4.
Only output the JSON object.
'''

        # 1. 调 LLM
        try:
            raw = self.call_llm(prompt)
        except Exception as e:
            self.logger.info(f"[MechanismExtraction] LLM call failed: {e}")
            return False

        # 2. 解析返回
        try:
            try:
                obj = json.loads(raw)
            except Exception:
                obj = self.parse_json(raw)
        except Exception as e:
            self.logger.info(f"[MechanismExtraction] parse_json failed: {e}")
            return False

        if not isinstance(obj, dict):
            # 有些模型会包在数组里
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                obj = obj[0]
            else:
                return False

        # 3. 校验 triple 对齐
        oh = str(obj.get("head", "")).strip()
        orl = str(obj.get("relation", "")).strip()
        ot = str(obj.get("tail", "")).strip()
        if oh != head or orl != rel or ot != tail:
            # 防止串 triple，强制 exact match
            return False

        mechanism = str(obj.get("mechanism", "")).strip()
        evidence = str(obj.get("evidence", "")).strip()
        try:
            conf = float(obj.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        # 4. 写回 triple
        changed = False

        if mechanism:
            if getattr(triple, "mechanism", "") != mechanism:
                triple.mechanism = mechanism
                changed = True

        if evidence:
            if getattr(triple, "evidence", "") != evidence:
                triple.evidence = evidence
                changed = True

        if conf > 0:
            old_conf = getattr(triple, "confidence", 0.0)
            if conf > old_conf:
                triple.confidence = conf
                changed = True

        return changed