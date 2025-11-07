import json
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

from Core.Agent import Agent
from Logger.index import get_global_logger
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from Memory.index import Memory  # 按你的实际路径改


class MechanismExtractionAgent(Agent):
    """
    只做一件事：为已有 KGTriple 补充机制说明（mechanism）、证据（evidence）、机制置信度。

    使用方式：
        mech_agent = MechanismExtractionAgent(client, model_name)
        mech_agent.process(memory)

    行为：
        - 从 memory.relations.all() 中取出所有 KGTriple
        - 按 triple.source 分组（假设 source 是 text_id 或直接是文本）
        - 对每个文本 + 该文本下所有 triples，调用一次大模型
        - 原地更新这些 triple 的 mechanism / evidence / confidence
    """

    def __init__(self, client: OpenAI, model_name: str) -> None:
        system_prompt = """
You are a dedicated Mechanism Extraction Agent for biomedical knowledge graphs.

INPUT:
- A biomedical text passage.
- One or more pre-extracted triples: (head, relation, tail).

TASK:
For EACH triple, infer and summarize the underlying biological / pharmacological / molecular mechanism
that explains HOW or WHY the head influences the tail under the given relation.

OUTPUT FORMAT (per triple):
For each input triple, you MUST output an object:

{
  "head": "exact_head",
  "relation": "RELATIONSHIP_TYPE",
  "tail": "exact_tail",
  "mechanism": "50-120 words mechanistic explanation in English, if reliably supported or well-established; otherwise empty string",
  "evidence": "short quote or faithful paraphrase from the given text that supports this mechanism, or empty string",
  "confidence": 0.0-1.0
}

RULES:
1. Alignment:
   - (head, relation, tail) in your output MUST exactly match one of the input triples.
   - Do not invent or remove triples.

2. Grounding:
   - Use the provided text as primary evidence.
   - You MAY use widely accepted biomedical knowledge if consistent with the text.
   - If support is weak or missing, set mechanism="" and confidence<=0.4.

3. Consistency with relation:
   - TREATS: therapeutic / pharmacodynamic explanation.
   - INHIBITS: inhibitory target, pathway, or functional block.
   - ACTIVATES: activation of receptors, signaling, transcription, etc.
   - CAUSES: pathogenic / toxic / mutational mechanism.
   - ASSOCIATED_WITH: associative / correlative reasoning, clearly marked.
   - REGULATES: regulatory / feedback / expression control.
   - INCREASES/DECREASES: upstream cause of change in levels or activity.
   - INTERACTS_WITH: binding, complex formation, direct interaction.

4. Quality:
   - No contradictions with the passage.
   - No hallucinated hyper-specific mechanisms without support.
   - Be conservative when uncertain.

Return ONLY a JSON array of such objects. No extra commentary.
"""
        super().__init__(client, model_name, system_prompt)
        self.logger = get_global_logger()

    # =============== 对外入口：直接吃 Memory ==================

    def process(
        self,
        memory: Memory,
        max_workers: int = 8,
    ) -> None:
        """
        从 memory 中读取三元组，按 source 分组，并行补充机制。
        直接原地修改 memory.relations.triples 中的 KGTriple。
        不返回值（如需导出，可再调用 memory.dump_json）。
        """
        triples: List[KGTriple] = memory.relations.all()
        if not triples:
            self.logger.info("[MechanismExtraction] no triples in memory, skip.")
            return

        # 1. 基于 Memory 构建 text_id -> text 的映射
        text_map = self._build_text_map_from_memory(memory)

        # 2. 按 text_id 分组 triples（依赖 triple.source）
        grouped: Dict[str, List[KGTriple]] = {}
        for t in triples:
            src = getattr(t, "source", "") or ""
            if not src:
                # 如果完全没 source，就暂时跳过（没文本无法提机制）
                continue

            # 情况 A：source 是一个 text_id，能在 text_map 里找到
            if src in text_map:
                text_id = src
            # 情况 B：source 本身就是一段文本（包含空格等） -> 直接用文本本身做 key
            elif " " in src:
                text_id = f"inplace::{hash(src)}"
                if text_id not in text_map:
                    text_map[text_id] = src
            else:
                # 找不到对应文本，跳过
                continue

            grouped.setdefault(text_id, []).append(t)

        if not grouped:
            self.logger.info("[MechanismExtraction] no triples with resolvable source, skip.")
            return

        # 3. 并行对每个 text_id 批量补充机制
        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for text_id, ts in grouped.items():
                text = text_map.get(text_id, "")
                if not text:
                    continue
                tasks.append(
                    executor.submit(
                        self._enrich_triples_for_text,
                        text_id,
                        text,
                        ts,
                    )
                )

            for f in tqdm(as_completed(tasks), total=len(tasks), desc="Mechanism enrichment"):
                try:
                    f.result()
                except Exception as e:
                    self.logger.info(f"[MechanismExtraction] worker failed: {e}")

        # 所有 triple 都是原对象引用，已经在内存中更新完毕
        self.logger.info("[MechanismExtraction] done updating mechanisms into memory.")

    # =============== 内部工具：从 Memory 获取文本 ===============

    def _build_text_map_from_memory(self, memory: Memory) -> Dict[str, str]:
        """
        尝试从 memory 中构造 text_id -> text 的查找表。

        约定/启发式（你可以按自己存的方式调整）：
        1. 如果某个 Subgraph.meta 里有:
            - "text_map": {id: text}
            - 或 "texts": {id: text}
            - 或 直接 {id: text}
           则纳入。
        2. 如果 meta 里有 "id" 和 "text"，也视作一个条目。
        """
        text_map: Dict[str, str] = {}

        for sg_id, sg in memory.subgraphs.items():
            meta = sg.get_meta() if hasattr(sg, "get_meta") else getattr(sg, "meta", {}) or {}
            if not isinstance(meta, dict):
                continue

            # 明确 text_map
            if "text_map" in meta and isinstance(meta["text_map"], dict):
                for k, v in meta["text_map"].items():
                    if isinstance(v, str) and v.strip():
                        text_map[str(k)] = v

            # texts: {id: text}
            if "texts" in meta and isinstance(meta["texts"], dict):
                for k, v in meta["texts"].items():
                    if isinstance(v, str) and v.strip():
                        text_map[str(k)] = v

            # 扁平: {id: text}
            for k, v in meta.items():
                if isinstance(v, str) and v.strip() and not k.startswith("_"):
                    # 非常保守：要求 key 看起来像 id，value 像长文本
                    if len(v) > 40:
                        text_map[str(k)] = v

            # 单条：{"id": "...", "text": "..."}
            if "id" in meta and "text" in meta and isinstance(meta["text"], str):
                tid = str(meta["id"])
                if meta["text"].strip():
                    text_map[tid] = meta["text"]

        return text_map

    # =============== 内部核心：对单个文本批量补机制 ===============

    def _enrich_triples_for_text(
        self,
        text_id: str,
        text: str,
        triples: List[KGTriple],
    ) -> None:
        """
        针对单个文本：
        - 把该文本及其 triples 打包给 LLM
        - 根据返回结果原地更新这些 triples
        """
        if not triples:
            return

        triples_payload = [
            {
                "head": t.head,
                "relation": t.relation,
                "tail": t.tail,
            }
            for t in triples
        ]

        prompt = f"""
Biomedical text:
\"\"\"{text}\"\"\"

Existing triples extracted from this text:
{json.dumps(triples_payload, ensure_ascii=False, indent=2)}

For EACH of the above triples, output exactly one JSON object:
{{
  "head": "...",
  "relation": "...",
  "tail": "...",
  "mechanism": "...",
  "evidence": "...",
  "confidence": 0.0-1.0
}}

Return ONLY a JSON array. Do not add or remove triples.
"""

        try:
            raw = self.call_llm(prompt)
            data = self.parse_json(raw)
        except Exception as e:
            self.logger.info(f"[MechanismExtraction] LLM call/parse failed for {text_id}: {e}")
            return

        if not isinstance(data, list):
            return

        # (head, relation, tail) -> 最佳机制
        mech_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for item in data:
            if not isinstance(item, dict):
                continue

            h = str(item.get("head", "")).strip()
            r = str(item.get("relation", "")).strip()
            t = str(item.get("tail", "")).strip()
            if not (h and r and t):
                continue

            key = (h.lower(), r.upper(), t.lower())
            mech = str(item.get("mechanism", "")).strip()
            ev = str(item.get("evidence", "")).strip()
            try:
                conf = float(item.get("confidence", 0.0))
            except Exception:
                conf = 0.0

            prev = mech_map.get(key)
            if (prev is None) or (conf > prev.get("confidence", 0.0)):
                mech_map[key] = {
                    "mechanism": mech,
                    "evidence": ev,
                    "confidence": conf,
                }

        # 原地更新 triples
        for t in triples:
            key = (t.head.strip().lower(), t.relation.strip().upper(), t.tail.strip().lower())
            info = mech_map.get(key)
            if not info:
                continue

            mech = info.get("mechanism", "")
            ev = info.get("evidence", "")
            conf = info.get("confidence", 0.0)

            if mech:
                t.mechanism = mech
            if ev:
                t.evidence = ev
            if conf > 0 and conf > getattr(t, "confidence", 0.0):
                t.confidence = conf