import json
import re
from typing import Dict, List, Tuple, Any, Optional

from openai import OpenAI

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, Subgraph
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from TypeDefinitions.KnowledgeGraphDefinitions.index import KnowledgeGraph


class PathExtractionAgent(Agent):
    """
    PathExtractionAgent

    目标：
    - 从一个局部知识图中，给定起点实体，搜索一条长度不超过 k 的“思维路径”（节点序列 + 三元组序列），
      用于后续科学假设生成。
    - 搜索过程中，对所有候选 child 节点用 LLM 打分，结合 penalty 进行启发式搜索。

    主要公开接口：
    - process(): 从 Memory 中读取 keyword_entity_map，为每个 keyword 的起点实体抽取一条路径，
      并写回 Memory。最后输出一张汇总表，展示每个起点实体对应的路径长度。
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        k: int = 20,
        memory: Optional[Memory] = None,
        query: str = "",
    ):
        system_prompt = (
            "You are a biomedical AI4Science assistant.\n"
            "Your job is to score candidate nodes for extending a local knowledge-graph path, "
            "so that the resulting paths are helpful for generating plausible, novel, and testable "
            "scientific hypotheses for the given user query.\n\n"
            "You will ALWAYS respond with a single JSON object mapping candidate_id (string) to an object:\n"
            "{\n"
            '  \"<candidate_id>\": {\n'
            '    \"score\": float in [0, 1],   // higher means better to extend the path\n'
            '    \"reasons\": [str],          // short bullet-style reasons\n'
            '    \"flags\": [str]             // optional tags, e.g., [\"redundant\", \"off_topic\"]\n'
            "  },\n"
            "  ...\n"
            "}\n\n"
            "DO NOT output markdown code fences. DO NOT output any text outside the JSON object."
        )

        super().__init__(client, model_name, system_prompt)

        self.memory: Memory = memory or get_memory()
        self.logger = get_global_logger()
        self.query = query or ""
        self.key_entities: List[KGEntity] = self.memory.get_key_entities()
        self.knowledge_graph: KnowledgeGraph = KnowledgeGraph(
            self.memory.get_allRealationShip()
        )
        self.k = k

        # 节点惩罚表：entity_id -> penalty（>=0）
        self.node_penalty: Dict[str, float] = {}
        # 惩罚权重：effective_score = raw_score - penalty_weight * penalty
        self.penalty_weight: float = 0.5

    # -------------------------------------------------------------------------
    # 公共主流程
    # -------------------------------------------------------------------------

    def process(self) -> None:
        """
        主入口：
        - 遍历 Memory 中的 keyword_entity_map，
        - 以每个实体为起点搜索路径，
        - 将找到的路径写回 Memory。
        - 最后输出一张简洁的表格日志：每个起点对应的路径长度。
        """
        keyword_entity_map = self.memory.get_keyword_entity_map()
        summary_rows: List[Dict[str, Any]] = []

        for keyword, ent_list in keyword_entity_map.items():
            for ent_data in ent_list:
                if isinstance(ent_data, KGEntity):
                    start_entity = ent_data
                else:
                    start_entity = KGEntity(**ent_data)

                node_path, edge_path = self.find_path_with_edges(
                    start=start_entity,
                    k=self.k,
                    adj=self.knowledge_graph.Graph,
                )

                if node_path:
                    path_len = len(node_path)
                    # 写回 Memory
                    self.memory.add_extracted_path(keyword, node_path, edge_path)
                else:
                    path_len = 0

                summary_rows.append(
                    {
                        "keyword": keyword,
                        "entity_name": getattr(start_entity, "name", "") or "",
                        "entity_id": getattr(start_entity, "entity_id", "") or "",
                        "path_len": path_len,
                    }
                )

        # 统一输出一个总表
        self._log_summary_table(summary_rows)

    # -------------------------------------------------------------------------
    # 路径搜索：DFS + LLM 打分 + penalty 回溯惩罚
    # -------------------------------------------------------------------------

    def find_path_with_edges(
        self,
        start: KGEntity,
        k: int,
        adj: Any,
    ) -> Tuple[List[KGEntity], List[KGTriple]]:
        """
        使用 DFS 进行启发式搜索：
        - 每一步对所有候选 child 用 LLM 统一打分；
        - 分数加上 penalty 调整后作为启发式，按分数排序扩展；
        - 维护全局 best_path（按累积得分）；
        - 当某条路径在未达到 k 长度时就走到死路 -> 认为是失败路径，对路径上的节点增加惩罚。

        返回：
        - best_nodes: List[KGEntity]
        - best_edges: List[KGTriple]
        若没有找到长度 > 2 的合理路径，则返回空列表。
        """
        node_path: List[KGEntity] = [start]
        edge_path: List[KGTriple] = []

        best_nodes: List[KGEntity] = node_path.copy()
        best_edges: List[KGTriple] = edge_path.copy()
        best_score: float = 0.0  # 累积得分

        def dfs(current: KGEntity, current_score: float) -> None:
            nonlocal best_nodes, best_edges, best_score

            # 只要长度 > 1，就可以更新 best（防止只返回孤立起点）
            if len(node_path) > 1 and current_score > best_score:
                best_score = current_score
                best_nodes = node_path.copy()
                best_edges = edge_path.copy()

            # 深度限制：达到 k 就不再向下扩展
            if len(node_path) >= k:
                return

            neighbors = adj.get(current.entity_id, [])
            candidates: List[KGEntity] = []
            candidate_edges: List[KGTriple] = []

            # 收集候选节点（去环）
            for child_stub, relation in neighbors:
                # 兼容 relation.object 为 KGEntity 或 dict
                child_data = relation.object
                if isinstance(child_data, KGEntity):
                    child_node = child_data
                else:
                    child_node = KGEntity(**child_data)

                # 防止简单环路：不允许路径中重复节点
                if any(child_node.entity_id == e.entity_id for e in node_path):
                    continue

                candidates.append(child_node)
                candidate_edges.append(relation)

            # 没有候选节点：如果还没到 k，视为“失败路径”，施加惩罚
            if not candidates:
                if len(node_path) < k:
                    self._penalize_path(node_path)
                return

            # 用 LLM 对所有候选打分
            scores_info = self._score_candidates_with_llm(
                node_path=node_path,
                edge_path=edge_path,
                candidates=candidates,
            )

            # 将 raw_score 与 penalty 结合得到 effective_score
            scored_children: List[Tuple[float, KGEntity, KGTriple]] = []
            for child_node, rel in zip(candidates, candidate_edges):
                info = scores_info.get(child_node.entity_id, {})
                raw_score = float(info.get("score", 0.0))
                penalty = self.node_penalty.get(child_node.entity_id, 0.0)
                effective = raw_score - self.penalty_weight * penalty

                # 允许 effective <= 0 被直接舍弃，避免低质 / 被严重惩罚的节点
                if effective <= 0:
                    continue

                scored_children.append((effective, child_node, rel))

            # 没有合格候选：同样看作“失败路径”
            if not scored_children:
                if len(node_path) < k:
                    self._penalize_path(node_path)
                return

            # 按 effective_score 从高到低扩展
            scored_children.sort(key=lambda x: x[0], reverse=True)

            for effective_score, child_node, rel in scored_children:
                node_path.append(child_node)
                edge_path.append(rel)

                # 向下搜索，累积得分
                dfs(child_node, current_score + effective_score)

                # 回溯
                node_path.pop()
                edge_path.pop()

        # 启动 DFS
        dfs(start, current_score=0.0)

        # 与你原来的逻辑保持一致：如果 best_nodes 太短，就返回空
        if len(best_nodes) <= 2:
            self.logger.debug(
                f"[PathExtraction] No sufficiently long path found from start={start.name} "
                f"(best_len={len(best_nodes)} <= 2)."
            )
            return [], []

        self.logger.debug(
            f"[PathExtraction] Found best path from start={start.name} | "
            f"nodes={len(best_nodes)}, edges={len(best_edges)}, score={best_score:.4f}"
        )
        return best_nodes, best_edges

    # -------------------------------------------------------------------------
    # LLM 打分 & JSON 解析
    # -------------------------------------------------------------------------

    def _score_candidates_with_llm(
        self,
        node_path: List[KGEntity],
        edge_path: List[KGTriple],
        candidates: List[KGEntity],
    ) -> Dict[str, Dict[str, Any]]:
        """
        调用 LLM，对同一层的所有候选 child 节点一次性打分。

        返回：
        {
          "<entity_id>": {
            "score": float in [0, 1],
            "reasons": [...],
            "flags": [...]
          },
          ...
        }
        """
        path_entities = [self._serialize_entity(e) for e in node_path]
        path_edges = [self._serialize_triple(tr) for tr in edge_path]
        candidate_payload = [
            {
                "id": c.entity_id,
                "node": self._serialize_entity(c),
            }
            for c in candidates
        ]

        payload = {
            "task": (
                "Score candidate nodes for extending a knowledge-graph path used for "
                "biomedical hypothesis generation."
            ),
            "query": self.query,
            "current_path": {
                "nodes": path_entities,
                "edges": path_edges,
            },
            "candidates": candidate_payload,
            "decision_criterion": {
                "novelty": (
                    "Does this node introduce non-trivial or under-explored connections or mechanisms?"
                ),
                "relevance": (
                    "Is this node relevant to the user query and the existing path, "
                    "especially regarding mechanisms, targets, delivery, safety, or outcomes?"
                ),
                "mechanistic_value": (
                    "Does this node extend a plausible mechanistic / causal chain "
                    "between upstream and downstream biomedical entities?"
                ),
                "verifiability": (
                    "Could hypotheses involving this node be tested or grounded in experiments "
                    "or analyses (e.g., in vitro, in vivo, or clinical studies)?"
                ),
                "do_not_reject_just_because": [
                    "the candidate label already appears in another node's description text",
                    "the candidate is a mechanism/action term rather than a named entity",
                    "the candidate originates from the same sentence as existing nodes "
                    "(local overlap is expected in extracted subgraphs)",
                ],
            },
            "output_format": (
                "Return ONLY a JSON object mapping candidate 'id' to:\n"
                "{\n"
                '  \"score\": float in [0, 1],\n'
                '  \"reasons\": [str],\n'
                '  \"flags\": [str]\n'
                "}"
            ),
        }

        prompt = json.dumps(payload, ensure_ascii=False)
        try:
            raw = self.call_llm(prompt)
            cleaned = self._strip_json_fences(raw)
            obj = json.loads(cleaned)

            if not isinstance(obj, dict):
                raise ValueError("LLM scoring output is not a JSON object.")

            # 轻量校验：确保每个条目都有 score 字段
            for cid, info in obj.items():
                if not isinstance(info, dict):
                    obj[cid] = {
                        "score": 0.0,
                        "reasons": ["invalid entry"],
                        "flags": ["invalid"],
                    }
                    continue
                if "score" not in info:
                    info["score"] = 0.0
                # 确保类型安全
                try:
                    info["score"] = float(info["score"])
                except Exception:
                    info["score"] = 0.0
                if "reasons" not in info or not isinstance(info["reasons"], list):
                    info["reasons"] = []
                if "flags" not in info or not isinstance(info["flags"], list):
                    info["flags"] = []

            return obj

        except Exception as e:
            self.logger.warning(
                f"[PathExtraction][LLM scoring] failed to parse JSON, error={e}"
            )
            # 出错时保守处理：所有候选给 0 分，避免乱扩展
            fallback = {
                c.entity_id: {
                    "score": 0.0,
                    "reasons": ["fallback 0 score"],
                    "flags": ["llm_error"],
                }
                for c in candidates
            }
            return fallback

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """
        去掉可能出现的 ```json / ``` 包裹。
        """
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text

    # -------------------------------------------------------------------------
    # 惩罚机制
    # -------------------------------------------------------------------------

    def _penalize_path(self, nodes: List[KGEntity], amount: float = 1.0) -> None:
        """
        当一条路径被判定为“失败路径”（在未达到 k 长度时走到死路）时，
        对路径上的节点增加 penalty，使之后遇到这些节点时有效得分降低。
        """
        for node in nodes:
            eid = node.entity_id
            self.node_penalty[eid] = self.node_penalty.get(eid, 0.0) + amount

        # 用 debug，避免刷屏
        self.logger.debug(
            "[PathExtraction] Penalized path: "
            + " -> ".join(f"{n.name}({n.entity_id})" for n in nodes)
        )

    # -------------------------------------------------------------------------
    # 序列化工具
    # -------------------------------------------------------------------------

    @staticmethod
    def _serialize_entity(e: KGEntity) -> Dict[str, Any]:
        return {
            "entity_id": getattr(e, "entity_id", None),
            "name": getattr(e, "name", None),
            "type": getattr(e, "entity_type", None),
            "normalized_id": getattr(e, "normalized_id", None),
            "aliases": getattr(e, "aliases", None),
            "description": getattr(e, "description", None),
        }

    @staticmethod
    def _serialize_triple(t: KGTriple) -> Dict[str, Any]:
        rel_type = getattr(t, "relation", None) or getattr(t, "rel_type", None)
        head = getattr(t, "head", None) or getattr(t, "head_id", None)
        tail = getattr(t, "tail", None) or getattr(t, "tail_id", None)

        subj = getattr(t, "subject", None)
        obj = getattr(t, "object", None)

        return {
            "relation_type": rel_type,
            "head": head,
            "tail": tail,
            "subject": subj,
            "object": obj,
            "source": getattr(t, "source", None),
            "mechanism": getattr(t, "mechanism", None),
            "evidence": getattr(t, "evidence", None),
        }

    # -------------------------------------------------------------------------
    # 汇总表输出
    # -------------------------------------------------------------------------

    def _log_summary_table(self, rows: List[Dict[str, Any]]) -> None:
        """
        在日志中输出一个简单的 ASCII 表：
        | Keyword | Entity | EntityID | PathLen |
        """
        if not rows:
            self.logger.info("[PathExtraction] No paths extracted for any entity.")
            return

        headers = ["Keyword", "Entity", "EntityID", "PathLen"]

        def _safe_str(x: Any) -> str:
            return str(x) if x is not None else ""

        # 计算每列宽度
        col_widths = [
            max(len(headers[0]), max(len(_safe_str(r["keyword"])) for r in rows)),
            max(len(headers[1]), max(len(_safe_str(r["entity_name"])) for r in rows)),
            max(len(headers[2]), max(len(_safe_str(r["entity_id"])) for r in rows)),
            max(len(headers[3]), max(len(_safe_str(r["path_len"])) for r in rows)),
        ]

        def _fmt_row(cols: List[str]) -> str:
            return (
                "| "
                + " | ".join(
                    c.ljust(w) for c, w in zip(cols, col_widths)
                )
                + " |"
            )

        # 构造表格
        sep_line = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        lines = [sep_line, _fmt_row(headers), sep_line]

        for r in rows:
            line = _fmt_row(
                [
                    _safe_str(r["keyword"]),
                    _safe_str(r["entity_name"]),
                    _safe_str(r["entity_id"]),
                    _safe_str(r["path_len"]),
                ]
            )
            lines.append(line)
        lines.append(sep_line)

        table_str = "\n".join(lines)
        self.logger.info("[PathExtraction] Path length summary:\n" + table_str)