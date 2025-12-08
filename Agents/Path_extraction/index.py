from typing import Dict, List, Tuple, Callable, Any, Optional
from numpy import tri
from openai import OpenAI
from sympy import false
from pprint import pprint
from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, Subgraph
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from TypeDefinitions.KnowledgeGraphDefinitions.index import KnowledgeGraph
import json
class PathExtractionAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str,memory:Optional[Memory]=None):
        self.system_prompt = (
            "You are a biomedical AI4Science assistant. "
            "Your job is to decide whether extending a knowledge-graph path "
            "with a candidate node is HELPFUL for generating plausible, "
            "novel and verifiable scientific hypotheses for a given query. "
            "Always respond with JSON: {\"accept\": true/false, \"reason\": \"...\"}."
        )
        super().__init__(client,model_name,self.system_prompt)
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
        self.query='What are the latest advancements in CRISPR-Cas9 gene editing technology for treating genetic disorders?'
        self.keyEntitys:List[KGEntity]=self.memory.get_key_entities()
        self.knowledgeGraph:KnowledgeGraph=KnowledgeGraph(self.memory.get_allRealationShip())
        # self.knowledgeGraph.init()
    def find_path_with_edges(
        self,
        start:KGEntity,
        k:int,
        adj:Any,
        is_valid:Callable[[Any], bool],
    ) -> Optional[Tuple[List[KGEntity], List[KGTriple]]]:
        node_path: List[KGEntity] = []
        edge_path: List[KGTriple] = []
         # 当前已找到的“最佳路径”（最长）
        best_nodes: List[KGEntity] = []
        best_edges: List[KGTriple] = []
        node_path.append(start)
        best_nodes = node_path.copy()   # 至少有起点
        best_edges = edge_path.copy()
         # 如果 k == 1，直接返回起点即可
        if k == 1:
            return best_nodes, best_edges
        def dfs(current:KGEntity) -> bool:
            nonlocal best_nodes, best_edges
            current_id = current.entity_id
            # 每次进入一个新节点，都尝试更新当前“最长路径”
            if len(node_path) > len(best_nodes):
                best_nodes = node_path.copy()
                best_edges = edge_path.copy()

            if len(node_path) == k:
                return True
            neighbors = adj.get(current_id, [])
            for child, relation in neighbors:
                child_data=relation.object
                child_node=KGEntity(**child_data)
                edge_path.append(relation)
                node_path.append(child_node)
                node_for_llm = node_path[:-1].copy()
                edge_for_llm = edge_path[:-1].copy()
                print('this is valid',node_for_llm)
                print('this is child node',child_node)
                if is_valid(child_node, node_for_llm, edge_for_llm):
                    if dfs(child_node): 
                        return True
                node_path.pop()
                edge_path.pop()
            return False
        
        found_full = dfs(start)

        if found_full:
            # 找到了一条长度恰好为 k 的路径，此时 best_nodes / best_edges 其实就是这条
            self.logger.info(
                f"[PathExtraction] Found full path of length {k} "
                f"(nodes={len(best_nodes)}, edges={len(best_edges)})."
            )
            return best_nodes, best_edges
        else:
            # 没有长度为 k 的路径，返回搜索过程中遇到的最长路径
            self.logger.info(
                f"[PathExtraction] No path of length {k} found; "
                f"return longest path length={len(best_nodes)}."
            )
            return best_nodes, best_edges
    def is_valid(
        self,
        child: KGEntity,
        node_path: List[KGEntity],
        edge_path: List[KGTriple],
    ) -> bool:
        def serialize_entity(e: KGEntity) -> Dict[str, Any]:
            return {
                "entity_id": e.entity_id,
                "name": getattr(e, "name", None),
                "type": getattr(e, "entity_type", None),
                "normalized_id": getattr(e, "normalized_id", None),
                "aliases": getattr(e, "aliases", None),
                "description": getattr(e, "description", None),
            }
        def serialize_triple(t: KGTriple) -> Dict[str, Any]:
            # 兼容你关系里的字段名（head / tail / relation + subject / object）
            rel_type = getattr(t, "relation", None) or getattr(t, "rel_type", None)
            head = getattr(t, "head", None) or getattr(t, "head_id", None)
            tail = getattr(t, "tail", None) or getattr(t, "tail_id", None)

            # subject / object 是你 JSON 里的完整实体信息
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
          # 当前路径序列（含 child，因为在 dfs 里已经 append 了）
        path_entities = [serialize_entity(e) for e in node_path]
        path_edges = [serialize_triple(tr) for tr in edge_path]
        # 候选节点（虽然已经在 path 里，但单独拿出来再强调一下）
        candidate_entity = serialize_entity(child)
        payload = {
            "task": "decide whether to keep extending a knowledge-graph path for hypothesis generation",
            "query": self.query,
            "current_path": {
                "nodes": path_entities,
                "edges": path_edges,
            },
            "candidate_extension": {
                "node": candidate_entity,
                "position": "path_suffix",
            },
            "decision_criterion": {
                "novelty": (
                    "Does this node help introduce potentially novel or less-trivial connections? "
                    "Even if its wording already appears in the description of an existing node, "
                    "it can still be useful if it acts as an explicit mechanism node that connects "
                    "different upstream/downstream entities or clarifies causal direction."
                ),
                "relevance": (
                    "Is this node relevant to the query about CRISPR-Cas9 and genetic disorders "
                    "or closely related biomedical mechanisms, targets, delivery strategies, "
                    "safety profiles, or therapeutic outcomes?"
                ),
                "mechanistic_value": (
                    "Does this node help form or extend a mechanistic / causal chain "
                    "(e.g., link gene-editing tools, molecular targets, delivery systems, "
                    "phenotypes, or safety effects)?"
                ),
                "verifiability": (
                    "Could hypotheses using this node be testable or grounded in experiment "
                    "(e.g., wet-lab validation, in vitro/in vivo models, or clinical studies)?"
                ),
                "do_not_reject_just_because": [
                    "the candidate label or phrase already appears in the description text of an existing node",
                    "the candidate is a mechanism/action term (e.g., 'genetically modifies') rather than a named entity",
                    "the candidate comes from the same or adjacent sentence as the current node (this is expected in local KG extraction)"
                ],
            },
            "instruction": (
                "Carefully read the query, the current path, and the candidate node. "
                "You are working with a local knowledge graph extracted from nearby sentences, "
                "so overlapping wording between node labels and descriptions is normal.\n\n"
                "ACCEPT the candidate if it helps structure the mechanism, connects different entities, "
                "clarifies causal direction, or can support concrete, testable hypotheses, "
                "even when the phrase already appears in a description.\n\n"
                "ONLY reject when the node is truly redundant (e.g., exact self-loop with no new role, "
                "no new entity, and no new mechanistic link) or clearly off-topic.\n\n"
                "Always respond ONLY with JSON: {\"accept\": true/false, \"reason\": \"...\"}."
            ),
        }

        prompt = json.dumps(payload, ensure_ascii=False)
        try:
            raw = self.call_llm(prompt)
            obj = json.loads(raw)
            accept = obj.get("accept")
            reason = obj.get("reason", "")

            self.logger.info(
                f"[PathExtraction][LLM is_valid] child={child.name} "
                f"accept={accept}, reason={reason}"
            )

            # 只在 accept 是布尔值时采用，否则保守返回 False
            if isinstance(accept, bool):
                return accept
            return False

        except Exception as e:
            # 解析失败 / LLM 出错 -> 保守一点，直接判 False，避免乱扩路径
            self.logger.warning(
                f"[PathExtraction][LLM is_valid] parse failed, child={child.name}, error={e}"
            )
            return False
        prompt = json.dumps(payload, ensure_ascii=False)
        return True
    def process(self):
        k=5
        keyEntityPath, keyTripePath = self.find_path_with_edges(
            self.keyEntitys[0], 
            k=k, 
            adj=self.knowledgeGraph.Graph, 
            is_valid=self.is_valid
        )
        print(f"✅ 返回路径长度：{len(keyEntityPath)}（目标长度 k={k}）")
        print("✅ 路径节点：")
        pprint(keyEntityPath)
        print("✅ 路径边：")
        pprint(keyTripePath)

 



