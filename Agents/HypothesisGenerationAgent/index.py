from typing import List, Dict, Any, Optional
from openai import OpenAI

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


class HypothesisGenerationAgent(Agent):
    """
    基于：
      - 用户 query（主导）
      - PathExtractionAgent 抽取的 KG 路径（仅作参考上下文）

    调用 LLM 生成一批：
      - 机制合理
      - 与 query 高度相关
      - 尽量可验证的 假设（hypotheses）

    ⚠️ 当前版本：LLM 输出为 Markdown 文本，而不是 JSON。
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        query: str,
        memory: Optional[Memory] = None,
        max_paths: int = 10,
        hypotheses_per_path: int = 3,
    ):
        # system prompt：强调“以 query 为中心，路径为辅助；输出 Markdown”
        system_prompt = (
            "You are a biomedical AI4Science assistant.\n"
            "Your PRIMARY goal is to propose plausible, mechanistic, and experimentally testable hypotheses "
            "that address the user's query.\n"
            "You may optionally use the provided knowledge-graph paths as supporting context, "
            "but you are NOT restricted to entities or relations in those paths. "
            "You can and should leverage your broader biomedical knowledge when helpful.\n\n"
            "You MUST answer in GitHub-flavored Markdown, in English.\n"
            "For each path, write a Markdown section containing multiple hypotheses that:\n"
            "- Are centered on the user query.\n"
            "- Optionally reference entities/relations from the path when they strengthen the reasoning.\n"
            "- Are mechanistic and, in principle, experimentally testable.\n\n"
            "For each hypothesis, include:\n"
            "- A level-3 heading (### Short hypothesis title)\n"
            "- A paragraph for the hypothesis statement\n"
            "- A bullet list including:\n"
            "  - Mechanistic explanation\n"
            "  - Experimental suggestion\n"
            "  - Relevance to the user query\n"
            "  - A confidence score between 0.0 and 1.0\n"
            "Do NOT include JSON. Do NOT wrap the output in ``` fences."
        )

        super().__init__(client, model_name, system_prompt)

        self.logger = get_global_logger()
        self.memory: Memory = memory or get_memory()
        self.query = query

        self.max_paths = max_paths
        self.hypotheses_per_path = hypotheses_per_path

    # ---------- 序列化工具：给 LLM 看的轻量结构 ----------

    def _serialize_entity_for_prompt(self, e: KGEntity) -> Dict[str, Any]:
        """把 KGEntity 压成一个简短的 dict，用于给 LLM 看。"""
        if hasattr(e, "serialize") and callable(e.serialize):
            return e.serialize()
        return {
            "id": getattr(e, "entity_id", None),
            "name": getattr(e, "name", None),
            "type": getattr(e, "entity_type", None),
            "normalized_id": getattr(e, "normalized_id", None),
            "description": getattr(e, "description", None),
        }

    def _serialize_triple_for_prompt(self, t: KGTriple) -> Dict[str, Any]:
        """把 KGTriple 压成一个简短的 dict，用于给 LLM 看。"""
        if hasattr(t, "serialize") and callable(t.serialize):
            return t.serialize()
        rel = getattr(t, "relation", None) or getattr(t, "rel_type", None)
        head = getattr(t, "head", None) or getattr(t, "head_id", None)
        tail = getattr(t, "tail", None) or getattr(t, "tail_id", None)
        return {
            "relation_type": rel,
            "head": getattr(head, "name", None) if head is not None else head,
            "tail": getattr(tail, "name", None) if tail is not None else tail,
        }

    # ---------- 构造 prompt（以 query 为主，path 为参考） ----------

    def _build_prompt_for_path(
        self,
        path_idx: int,
        node_path: List[KGEntity],
        edge_path: List[KGTriple],
    ) -> str:
        """
        把 query + 一条路径打包成“自然语言 + 结构化内容”的 prompt，
        要求模型输出 Markdown。
        这里明确说明：query 是主导目标，path 只是 extra context。
        """
        nodes_ser = [self._serialize_entity_for_prompt(e) for e in node_path]
        edges_ser = [self._serialize_triple_for_prompt(tr) for tr in edge_path]

        path_txt = (
            f"Path index: {path_idx}\n\n"
            "Nodes (in order along the path):\n"
        )
        for i, n in enumerate(nodes_ser):
            path_txt += (
                f"- [{i}] name={n.get('name')!r}, "
                f"type={n.get('type')!r}, "
                f"id={n.get('id')!r}, "
                f"normalized_id={n.get('normalized_id')!r}\n"
            )

        path_txt += "\nEdges (relations along the path):\n"
        for i, r in enumerate(edges_ser):
            path_txt += (
                f"- [{i}] relation_type={r.get('relation_type')!r}, "
                f"head={r.get('head')!r}, tail={r.get('tail')!r}\n"
            )

        instruction = (
            "Your task:\n"
            f"- The user query is:\n  {self.query!r}\n\n"
            "- Your PRIMARY objective is to propose hypotheses that directly help answer this query.\n"
            "- You may use the provided path as optional mechanistic context, "
            "but you are free to bring in additional mechanisms, targets, pathways, or experimental designs "
            "beyond what appears in the path.\n"
            "- When appropriate, you are encouraged to connect the query with entities or relations from the path, "
            "but do NOT force every hypothesis to be strictly confined to the path.\n\n"
            "- For this path, propose "
            f"{self.hypotheses_per_path} non-trivial, mechanistic, and experimentally testable hypotheses.\n"
            "- Each hypothesis must:\n"
            "  - Clearly address or illuminate some aspect of the user query.\n"
            "  - Provide a plausible biological mechanism or causal chain.\n"
            "  - Propose at least one concrete experimental design or assay.\n"
            "  - Explain why this hypothesis is relevant to the user query.\n"
            "  - Include a confidence score between 0.0 and 1.0.\n\n"
            "Output format (Markdown ONLY):\n"
            f"- Start with a level-2 heading: `## Path {path_idx} – Hypotheses`\n"
            "- Then, for each hypothesis, use the following template:\n\n"
            "### <Short hypothesis title>\n"
            "<One paragraph with the full hypothesis statement.>\n\n"
            "- Mechanistic explanation: <1–3 sentences>\n"
            "- Experimental suggestion: <1–3 sentences>\n"
            "- Relevance to query: <1–3 sentences>\n"
            "- Confidence: <a number between 0.0 and 1.0>\n\n"
            "Do NOT use JSON. Do NOT wrap your answer in ``` code fences.\n"
        )

        prompt = (
            f"{instruction}\n"
            "Here is the optional path context (you may use it if helpful):\n\n"
            "---------------- PATH CONTEXT BEGIN ----------------\n"
            f"{path_txt}\n"
            "---------------- PATH CONTEXT END ----------------\n"
        )

        return prompt

    def _call_llm_for_path(
        self,
        path_idx: int,
        node_path: List[KGEntity],
        edge_path: List[KGTriple],
    ) -> str:
        """
        针对单条路径调用一次 LLM，返回 **Markdown 字符串**。
        不再做 JSON 解析。
        """
        # 即使 path 很短/很弱，我们也可以照样给一点 context；
        # 如果你希望空 path 直接跳过，这里可以加判断。
        prompt = self._build_prompt_for_path(path_idx, node_path, edge_path)
        try:
            raw = self.call_llm(prompt)
            self.logger.info(
                f"[HypothesisGeneration] path_idx={path_idx} got markdown hypotheses block (len={len(raw)})"
            )
            return raw
        except Exception as e:
            self.logger.warning(
                f"[HypothesisGeneration] path_idx={path_idx} LLM call failed: {e}"
            )
            return ""
    def _collect_path_text_context(
        self,
        path_nodes: List[KGEntity],
        max_total_chars: int = 4000,
    ) -> str:
        """
        根据路径上的 KG 节点，在 Memory 中查找包含这些实体的子图，
        把对应子图 meta 中的文本片段拼接成整体上下文。

        约定：
        - self.memory.subgraphs: Dict[str, Subgraph]
        - Subgraph.entities.all() 返回该子图中的 KGEntity 列表
        - Subgraph.get_meta() 返回 dict，文本优先从 meta["text"] 取
        """
        if not path_nodes:
            return ""

        # 路径中涉及到的实体 id 集合
        path_entity_ids: set[str] = set()
        for entity in path_nodes:
            entity_id = getattr(entity, "entity_id", None) 
            if entity_id:
                path_entity_ids.add(entity_id)

        merged_snippets: List[str] = []
        visited_subgraph_ids: set[str] = set()

        # 遍历所有子图，找到“与路径实体有交集”的子图
        for subgraph_id, subgraph in self.memory.subgraphs.items():
            if subgraph_id in visited_subgraph_ids:
                continue
            subgraph_entities: List[KGEntity] = subgraph.get_entities()
            subgraph_entity_ids: set[str] = set()
            for entity in subgraph_entities:
                entity_id = getattr(entity, "entity_id", None) or getattr(entity, "id", None)
                if entity_id:
                    subgraph_entity_ids.add(entity_id)

            # 没有任何路径实体出现在该子图中，则跳过
            if not path_entity_ids.intersection(subgraph_entity_ids):
                continue

            meta: Dict[str, Any] = subgraph.get_meta() if hasattr(subgraph, "get_meta") else {}
            raw_text: str = ""
            if isinstance(meta, dict):
                # 如果你实际使用的 key 不是 "text"，在这里调整优先级
                raw_text = meta.get("text") or meta.get("chunk") or ""

            if raw_text:
                visited_subgraph_ids.add(subgraph_id)
                merged_snippets.append(
                    f"[Subgraph: {subgraph_id}]\n{raw_text.strip()}"
                )

            # 控制整体长度，避免 prompt 过大
            current_joined = "\n\n---\n\n".join(merged_snippets)
            if len(current_joined) >= max_total_chars:
                break

        return "\n\n---\n\n".join(merged_snippets)
    # ---------- 对外主入口 ----------
    def process(self) -> List[Dict[str, Any]]:
        """
        主流程：
          1）从 Memory 中取出已抽取的路径；
          2）对前 max_paths 条路径逐一调用 LLM 生成假设（Markdown），
             每次都以【用户 query 为主】、【path 为参考】；
          3）返回一个列表，每个元素包含：
              {
                "path_index": int,
                "nodes": [...],
                "edges": [...],
                "markdown": "<hypotheses in markdown>"
              }
        """
        all_paths = getattr(self.memory, "get_extracted_paths", lambda: [])()
        if not all_paths:
            self.logger.warning("[HypothesisGeneration] no extracted paths found in memory.")
            return []

        results: List[Dict[str, Any]] = []

        for idx, p in enumerate(all_paths[: self.max_paths]):
            node_path: List[KGEntity] = p.get("nodes", []) or []
            edge_path: List[KGTriple] = p.get("edges", []) or []

            # 即使 node_path 为空，这里也可以把 path 当“空 context”，直接根据 query 生成；
            # 如果你想跳过空路径，就保留这个判断：
            # if not node_path:
            #     continue

            markdown_block = self._call_llm_for_path(idx, node_path, edge_path)
            results.append(
                {
                    "path_index": idx,
                    "nodes": node_path,
                    "edges": edge_path,
                    "markdown": markdown_block,
                }
            )

        return results