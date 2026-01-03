import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from Core.Agent import Agent
from Logger.index import get_global_logger
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


class TruthHypoAgent(Agent):
    """
    专门负责路径级别的真假性/相关性裁定。
    输入：用户 query、抽取的 KG 路径、路径上下文。
    输出：一个标签（正相关/负相关/无关），附带置信度和简要依据。
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        query: str,
        memory: Optional[Any] = None,
    ):
        system_prompt = """
    You are an expert biomedical researcher and information extraction assistant. Your task is to analyze biological entities within a given context and classify the relationship between them.

    There are three distinct sub-tasks. Depending on the entity types provided (Chemical, Gene, Disease), apply the corresponding definition:

    Task 1: Chemical–Gene Interaction
    Context: Pharmacology, toxicology.
    Entities: Head (Chemical) → Tail (Gene/Protein).
    Labels:
    - positive_correlate: chemical increases/activates/upregulates the gene/protein.
    - negative_correlate: chemical decreases/inhibits/downregulates the gene/protein.
    - no_relation: no functional interaction or correlation described.

    Task 2: Disease–Gene Association
    Context: Pathology, etiology.
    Entities: Head (Disease) ↔ Tail (Gene/Protein).
    Labels:
    - stimulate: gene promotes/causes/exacerbates disease, or disease upregulates the gene.
    - inhibit: gene protects/suppresses disease, or disease downregulates the gene.
    - no_relation: no direct pathological or functional link described.

    Task 3: Gene–Gene Regulation
    Context: Molecular biology, signaling pathways.
    Entities: Head (Upstream Gene/Protein) → Tail (Downstream Gene/Protein).
    Labels:
    - positive_correlate: upstream gene activates/promotes/increases downstream activity or expression.
    - negative_correlate: upstream gene represses/inhibits/decreases downstream activity or expression.
    - no_relation: no direct regulatory interaction described.

    Instructions:
    1. Identify the entity types for the given pair.
    2. Select the matching task definition.
    3. Analyze the context (KG path and supporting text) to decide the correct label.
    4. Respond ONLY with valid JSON (no code fences, no extra text) in the format:
    {
      "label": "positive_correlate | negative_correlate | stimulate | inhibit | no_relation",
      "confidence": "the score ranging from 0.0 to 1.0, indicating your confidence in the label",
      "rationale": "brief explanation citing the context",
      "query_answer": "one-sentence direct answer to the user query"
    }
        """
        super().__init__(client, model_name, system_prompt)
        self.logger = get_global_logger()
        self.memory = memory or get_memory()
        self.query = query

    @staticmethod
    def serialize_path(node_path: List[KGEntity], edge_path: List[KGTriple]) -> str:
        parts: List[str] = []
        for i, node in enumerate(node_path):
            parts.append(f"{node.name}:{node.entity_type}")
            if i < len(edge_path):
                edge = edge_path[i]
                parts.append(f"-[{edge.relation}]->")
        return "".join(parts)

    def process(
        self,
    ):
        """Collect all paths+上下文，一次性送入 LLM 批量裁定。"""
        all_paths: Dict[str, List[Any]] = getattr(self.memory, "paths", {}) or {}

        items: List[Dict[str, Any]] = []
        path_lookup: List[Dict[str, Any]] = []  # 保留原始信息方便回填

        for entity, paths in all_paths.items():
            for idx, path in enumerate(paths):
                node_path: List[KGEntity] = path.get("nodes", []) or []
                edge_path: List[KGTriple] = path.get("edges", []) or []
                path_str: str = path.get("path") or ""
                contexts: str = path.get("contexts", "") or ""

                # 累积来源上下文，尽量给 LLM 充分证据
                sources = set()
                for edge in edge_path:
                    if getattr(edge, "source", None):
                        sources.add(edge.source)
                for src in sources:
                    try:
                        contexts += (self.memory.subgraphs[src].meta.get("text", "") + "\n")
                    except Exception:
                        continue

                if not path_str and node_path and edge_path:
                    path_str = self.serialize_path(node_path, edge_path)
                head, tail = self._extract_entities(node_path, path_str)
                items.append(
                    {
                        "entity": entity,
                        "path_index": idx,
                        "path": path_str,
                        "contexts": contexts,
                        "entity_head": head,
                        "entity_tail": tail,
                    }
                )
                path_lookup.append({
                    "entity": entity,
                    "path_index": idx,
                    "path": path_str,
                    "contexts": contexts,
                    "entity_head": head,
                    "entity_tail": tail,
                })


        payload = {
            "task": "batch link truthfulness classification",
            "query": self.query,
            "items": items,
        }

        try:
            raw = self.call_llm(json.dumps(payload, ensure_ascii=False))
            obj = self._parse_response(raw)
            responses=obj
            return responses
        except Exception as e:
            self.logger.warning(f"[TruthHypoAgent] LLM batch call failed: {e}")
            return [
                {
                    **lookup,
                    "raw_label": None,
                    "label": None,
                    "confidence": 0.0,
                    "rationale": "",
                    "query_answer": "",
                    "error": str(e),
                }
                for lookup in path_lookup
            ]

    def _build_prompt_text(
        self,
        head: Tuple[str, str],
        tail: Tuple[str, str],
        contexts: str,
        path_str: str,
    ) -> str:
        head_name, head_type = head
        tail_name, tail_type = tail
        context_block = contexts.strip() or "N/A"
        return (
            f"Entity 1: {head_name} ({head_type})\n"
            f"Entity 2: {tail_name} ({tail_type})\n"
            f"Context: {context_block}\n"
            f"Path: {path_str}\n"
            "Output:"
        )

    def _extract_entities(
        self,
        node_path: List[KGEntity],
        path_str: str,
    ) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        if node_path:
            head = node_path[0]
            tail = node_path[-1]
            return (head.name, head.entity_type), (tail.name, tail.entity_type)

        pattern = re.compile(r"([^:\-\[]+):([^->\[]+)")
        matches = pattern.findall(path_str)
        if matches:
            head_name, head_type = matches[0]
            tail_name, tail_type = matches[-1]
            return (head_name.strip(), head_type.strip()), (tail_name.strip(), tail_type.strip())

        return ("Unknown", "Unknown"), ("Unknown", "Unknown")

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, count=1, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()
        return json.loads(text)

    @staticmethod
    def _normalize_label(label: str) -> Optional[str]:
        key = label.lower().strip()
        mapping = {
            "positive_correlate": "正相关",
            "stimulate": "正相关",
            "positive": "正相关",
            "positive association": "正相关",
            "negative_correlate": "负相关",
            "inhibit": "负相关",
            "negative": "负相关",
            "negative association": "负相关",
            "no_relation": "无关",
            "no relation": "无关",
            "none": "无关",
        }
        return mapping.get(key)
