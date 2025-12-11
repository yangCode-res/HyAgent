from typing import List, Dict, Any, Optional
from openai import OpenAI
import json

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


class HypothesisGenerationAgent(Agent):
    """
    基于：
      - 用户 query
      - PathExtractionAgent 抽取的 KG 路径（节点 + 边）

    调用 LLM 生成一批：
      - 机制合理
      - 与 query 相关
      - 尽量可验证的 假设（hypotheses）
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
        # system prompt：主要定义“你是谁 + 要做什么 + 输出格式”
        system_prompt = """
        You are a biomedical AI4Science assistant.
        Given a user query and a mechanistic path extracted from a biomedical knowledge graph,
        your task is to propose plausible, mechanistic, and experimentally testable hypotheses
        that leverage the entities and relations along this path.
        Your another task is to according to several given generated hypotheses and their contexts,
        generate a more comprehensive hypothesis.
        The input will be a JSON payload containing the user query and a single KG path, along with the context of the path.
        The input format is as follows:
        Task 1:
        {
        "task": "generate mechanistic, testable biomedical hypotheses based on a KG path",
        "query": "user query string",
        "path_index": int,
        "path": "string",it will be given in the format of entity1-[relation1]->entity2-[relation2]->entity3...
        "contexts": "string"
        }
        Task 2:
        {
        "task": "generate a more comprehensive hypothesis based on several given hypotheses and their contexts",
        "query": "user query string",
        "given_hypotheses": [
            {
            "hypothesis": "full hypothesis statement",
            "context": "string"
            }
        ]
        }
        You should respond ONLY with valid JSON in the following format:

        {
        "hypotheses": [
            {
            "title": "short hypothesis title",
            "hypothesis": "full hypothesis statement",
            "mechanism_explanation": "how the entities/relations in the path support this hypothesis",
            "experimental_suggestion": "a concise, concrete experimental idea to test it",
            "relevance_to_query": "why this hypothesis is relevant to the user query",
            "confidence": 0.0
            }
        ]
        }

        Do not include any text outside the JSON response.
        """


        super().__init__(client, model_name, system_prompt)

        self.logger = get_global_logger()
        self.memory: Memory = memory or get_memory()
        self.query = query

        self.max_paths = max_paths
        self.hypotheses_per_path = hypotheses_per_path

    # ---------- 工具函数：序列化实体 / 三元组，方便喂给 LLM ----------


    def _build_prompt_for_path(
        self,
        path_idx: int,
        node_path: List[KGEntity],
        edge_path: List[KGTriple],
        hypotheses:Optional[List[Dict[str,Any]]]=None,
        task:int=1,
    ) -> str:
        """
        把 query + 一条路径打包成 JSON prompt 发给 LLM。
        """
        if(task==1){
            payload = {
                "task": "generate mechanistic, testable biomedical hypotheses based on a KG path",
                "query": self.query,
                "path_index": path_idx,
                "path": {
                    "nodes": [e.serialize() for e in node_path],
                    "edges": [tr.serialize() for tr in edge_path],
                },
                "constraints": {
                    "use_path": "Hypotheses should explicitly leverage entities and relations along the path.",
                    "mechanistic": "Explain the mechanism or causal chain implied by the path.",
                    "testable": "Each hypothesis should be experimentally testable in principle.",
                    "novelty": "Prefer non-trivial or non-obvious combinations over purely textbook restatement.",
                    "relevance": "Hypotheses must be relevant to the user query.",
                },
                "generation_config": {
                    "max_hypotheses": self.hypotheses_per_path,
                },
                "output_format": {
                    "hypotheses": [
                        {
                            "title": "short hypothesis title",
                            "hypothesis": "full hypothesis statement",
                            "mechanism_explanation": "how path entities/relations support it",
                            "experimental_suggestion": "how to test it",
                            "relevance_to_query": "why this matters for the query",
                            "confidence": 0.0,
                        }
                    ]
                },
                "instruction": (
                    "Return ONLY valid JSON with a top-level key 'hypotheses'. "
                    "Do not include any natural language outside JSON."
                ),
            }
        }

        return json.dumps(payload, ensure_ascii=False)

    def _call_llm_for_path(
        self,
        path_idx: int,
        node_path: List[KGEntity],
        edge_path: List[KGTriple],
    ) -> List[Dict[str, Any]]:
        """
        针对单条路径调用一次 LLM，返回解析好的 hypothesis 列表。
        """
        if not node_path:
            return []

        prompt = self._build_prompt_for_path(path_idx, node_path, edge_path)
        try:
            raw = self.call_llm(prompt)
            obj = json.loads(raw)
            hyps = obj.get("hypotheses", [])
            if not isinstance(hyps, list):
                self.logger.warning(
                    f"[HypothesisGeneration] path_idx={path_idx} LLM returned invalid 'hypotheses' type: {type(hyps)}"
                )
                return []
            self.logger.info(
                f"[HypothesisGeneration] path_idx={path_idx} got {len(hyps)} hypotheses from LLM."
            )
            return hyps
        except Exception as e:
            self.logger.warning(
                f"[HypothesisGeneration] path_idx={path_idx} LLM call/parse failed: {e}"
            )
            return []

    # ---------- 对外主入口 ----------
    def process(self) -> List[Dict[str, Any]]:
        """
        主流程：
          1）从 Memory 中取出已抽取的路径；
             路径格式为：
             {
               "core_entity":List[Path],
            }
             其中 Path 格式为：
             {
               "nodes": List[KGEntity],
               "edges": List[KGTriple],
            }
            即三元组列表；
          2）对每个实体的 max_paths 路径逐一调用 LLM 生成假设；
          3）返回一个字典，每个元素包含：
              {
                {"path_index": int,
                "nodes": [...],
                "edges": [...],
                "hypotheses": [...]}
              }
          之后你可以视情况把这些结果再写回 Memory。
        """
        all_paths = self.memory.get_extracted_paths()
        if not all_paths:
            self.logger.warning("[HypothesisGeneration] no extracted paths found in memory.")
            return []

        results: List[Dict[str, Any]] = []

        for idx, p in enumerate(all_paths[: self.max_paths]):
            node_path: List[KGEntity] = p.get("nodes", []) or []
            edge_path: List[KGTriple] = p.get("edges", []) or []

            if not node_path:
                continue

            hyps = self._call_llm_for_path(idx, node_path, edge_path)
            results.append(
                {
                    "path_index": idx,
                    "nodes": node_path,
                    "edges": edge_path,
                    "hypotheses": hyps,
                }
            )

        # TODO：如你需要，可以在这里把 results 写回 Memory，如：
        # if hasattr(self.memory, "add_generated_hypotheses"):
        #     self.memory.add_generated_hypotheses(self.query, results)

        return results