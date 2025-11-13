from openai import OpenAI
from Core.Agent import Agent
from Memory.index import Memory, Subgraph
from Logger.index import get_global_logger
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from transformers import AutoTokenizer, AutoModel
from Store.index import get_memory
import numpy as np
import torch
from Config.index import BioBertPath
from typing import Dict, List, Tuple, Any, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
Embedding = List[float]
class AlignmentTripleAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str,memory:Optional[Memory]=None):
        self.system_prompt=""""""""
        super().__init__(client,model_name,self.system_prompt)
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
        self.subgraph_entity_embeddings: Dict[str, Dict[str, Embedding]] = {}
        self.biobert_dir = BioBertPath
        self.biobert_model=None
        self.biobert_tokenizer=None
        self._load_biobert()
        self.subgraph_adj: Dict[str, Tuple[np.ndarray, Dict[str, int]]] = {}
        self.subgraph_hypergraphs: Dict[str, Dict[str, Any]] = {}
        self.entity_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.subgraph_entities: Dict[str, Dict[str, KGEntity]] = {}
    def process(self) -> None:
        for sg_id, sg in self.memory.subgraphs.items():
            ent_embeds: Dict[str, Embedding] = {}
            for ent in sg.entities.all():
                text = ent.description or ent.name or ent.normalized_id
                embedding = self._encode_text(text)  # 返回 List[float] 或 np.ndarray
                ent_embeds[ent.entity_id] = embedding
                ent_map[ent.entity_id] = ent  # 记录实体对象
            # 这里的 sg 在类型系统里就是 Subgraph
            self.subgraph_entity_embeddings[sg_id] = ent_embeds
            id2idx, adj = self.build_adj_for_subgraph(sg)
            self.subgraph_adj[sg_id] = (adj, id2idx)
            H, center_ids, hyperedge_embeds = self.build_hypergraph_for_subgraph(
                sg, id2idx, adj, ent_embeds
            )
            self.subgraph_hypergraphs[sg_id] = {
                "H": H,                      # n × m incidence matrix
                "center_ids": center_ids,    # len = m
                "hyperedge_embeddings": hyperedge_embeds,  # m × d (如果 embedding 齐全)
            }
            # 简单打印检查一下
            self.logger.info(
                f"[Adjacency] subgraph={sg_id}, |V|={adj.shape[0]}, |E|={int(adj.sum())}"
            )
            self.logger.info(
                f"[Hypergraph] subgraph={sg_id}, |V|={H.shape[0]}, |E_h|={H.shape[1]}"
            )
        self.propagate_embeddings_with_hypergraph(alpha=0.5)
        self.build_entity_alignment(sim_threshold=0.9, top_k=10)
        # 用 LLM 做精过滤（并行）
        self.refine_alignment_with_llm(max_workers=8)
        for src_sg_id, ent_map in self.entity_alignment.items():
            for src_eid, matches in ent_map.items():
                print(src_sg_id, src_eid)
                for m in matches:
                    print("  ->", m["target_subgraph"], m["target_entity"], m["similarity"])   
    def _llm_filter_for_entity(
        self,
        src_sg_id: str,
        src_eid: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        对“一个源实体 + 多个候选实体”调用一次 LLM：
        - 组装源实体 / 候选实体的 name / type / description / text context
        - 请 LLM 输出 JSON: {"keep": [候选下标列表]}
        - 返回被保留的候选（原始 dict 子集）
        """
        # 1. 取源实体对象
        src_ent_obj = self.subgraph_entities.get(src_sg_id, {}).get(src_eid, None)
        if src_ent_obj is None:
            # 没有实体对象就没法提供文本，直接保留原始候选（或者直接返回空，看你需求）
            return candidates

        src_text = self._build_entity_text_context(src_ent_obj)

        # 2. 组装候选实体信息
        cand_infos = []
        for idx, cand in enumerate(candidates):
            tgt_sg_id = cand["target_subgraph"]
            tgt_eid = cand["target_entity"]
            sim = cand.get("similarity", 0.0)

            tgt_ent_obj = self.subgraph_entities.get(tgt_sg_id, {}).get(tgt_eid, None)
            if tgt_ent_obj is None:
                continue

            tgt_text = self._build_entity_text_context(tgt_ent_obj)

            cand_infos.append(
                {
                    "idx": idx,
                    "target_subgraph": tgt_sg_id,
                    "target_entity": tgt_eid,
                    "similarity": sim,
                    "text": tgt_text,
                }
            )

        if not cand_infos:
            return []

        # 3. 构造 prompt
        system_prompt = (
            "You are an expert biomedical entity alignment model. "
            "Given one source entity and a list of candidate target entities from "
            "other subgraphs, decide which candidates refer to the SAME real-world "
            "biomedical concept as the source entity (e.g., same disease, same drug).\n\n"
            "Respond strictly in JSON format: {\"keep\": [list of candidate_idx]}.\n"
            "Only include indices for candidates that are very likely to be the same entity. "
            "If none match, return an empty list."
        )

        # user 内容里给结构化信息
        user_payload = {
            "source": {
                "subgraph_id": src_sg_id,
                "entity_id": src_eid,
                "text": src_text,
            },
            "candidates": [
                {
                    "candidate_idx": ci["idx"],
                    "target_subgraph": ci["target_subgraph"],
                    "target_entity": ci["target_entity"],
                    "similarity": ci["similarity"],
                    "text": ci["text"],
                }
                for ci in cand_infos
            ],
        }

        user_content = (
            "Here is the source entity and its candidate target entities.\n\n"
            "Decide which candidates are the same real-world biomedical entity as the source.\n"
            "Return JSON only, no extra text.\n\n"
            f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            keep_idxs = set(data.get("keep", []))
        except Exception as e:
            self.logger.warning(
                f"[LLMAlign] LLM call/parse failed for ({src_sg_id}, {src_eid}): {e}"
            )
            # 如果 LLM 出错，可以选择“全保留”或者“全丢弃”，这里示例为全保留
            return candidates

        # 4. 根据 keep 下标过滤
        kept_candidates: List[Dict[str, Any]] = []
        for ci in cand_infos:
            if ci["idx"] in keep_idxs:
                kept_candidates.append(
                    {
                        "target_subgraph": ci["target_subgraph"],
                        "target_entity": ci["target_entity"],
                        "similarity": ci["similarity"],
                    }
                )

        return kept_candidates
        def refine_alignment_with_llm(self, max_workers: int = 8) -> None:
            """
            使用 LLM 对粗召回的实体对齐结果做精过滤（并行调用）：
            - 单次 LLM 调用对应“一个源实体 + 它的若干候选实体”
            - 让 LLM 决定哪些候选是真正同一实体
            - 最终更新 self.entity_alignment（结构保持不变）
            """
            jobs = []  # (src_sg_id, src_eid, candidates)

            for src_sg_id, ent_map in self.entity_alignment.items():
                for src_eid, candidates in ent_map.items():
                    if not candidates:
                        continue
                    jobs.append((src_sg_id, src_eid, candidates))

            if not jobs:
                self.logger.info("[LLMAlign] No candidates to refine, skip.")
                return

            refined_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

            self.logger.info(
                f"[LLMAlign] Start LLM refinement, #entities={len(jobs)}, max_workers={max_workers}"
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key = {}
                for (src_sg_id, src_eid, candidates) in jobs:
                    fut = executor.submit(
                        self._llm_filter_for_entity,
                        src_sg_id,
                        src_eid,
                        candidates,
                    )
                    future_to_key[fut] = (src_sg_id, src_eid)

                for fut in as_completed(future_to_key):
                    src_sg_id, src_eid = future_to_key[fut]
                    try:
                        kept = fut.result()  # List[Dict] (过滤后的候选)
                    except Exception as e:
                        self.logger.warning(
                            f"[LLMAlign] LLM refine failed for ({src_sg_id}, {src_eid}): {e}"
                        )
                        kept = []

                    if kept:
                        refined_alignment.setdefault(src_sg_id, {})[src_eid] = kept

            self.entity_alignment = refined_alignment
            self.logger.info(
                f"[LLMAlign] Done. #subgraphs={len(self.entity_alignment)}"
            )
    def build_entity_alignment(self, sim_threshold: float = 0.7, top_k: int = 10) -> None:
        """
        实体对齐：
        - 遍历每个子图的每个实体（作为源实体）
        - 对于每一个“其他子图”，计算该源实体与该子图所有实体的余弦相似度
        - 对每个目标子图，保留相似度 >= sim_threshold 的 Top-K 实体
        - 结果保存在 self.entity_alignment 中，不写文件

        self.entity_alignment 结构：
        {
            src_sg_id: {
                src_entity_id: [
                    {
                        "target_subgraph": tgt_sg_id,
                        "target_entity": tgt_entity_id,
                        "similarity": float
                    },
                    ...
                ]
            },
            ...
        }
        """
        # 1. 先把每个子图的实体 embedding 组织成矩阵并预归一化，方便做余弦相似度
        #    normalized_embeddings: {sg_id: (entity_ids: List[str], emb_matrix: np.ndarray [n, d])}
        normalized_embeddings: Dict[str, Tuple[List[str], np.ndarray]] = {}

        for sg_id, ent_embeds in self.subgraph_entity_embeddings.items():
            ids: List[str] = []
            vecs: List[np.ndarray] = []

            for eid, emb in ent_embeds.items():
                v = np.asarray(emb, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm == 0.0:
                    continue
                v = v / norm
                ids.append(eid)
                vecs.append(v)

            mat = np.stack(vecs, axis=0)   # [n, d]
            normalized_embeddings[sg_id] = (ids, mat)

        alignment_result: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        # 2. 遍历每个源子图 / 源实体，跟其他子图做对齐
        for src_sg_id, (src_ids, src_mat) in normalized_embeddings.items():
            sg_align: Dict[str, List[Dict[str, Any]]] = {}

            # 遍历源子图里的每个实体
            for i_src, src_eid in enumerate(src_ids):
                src_vec = src_mat[i_src : i_src + 1, :]  # shape [1, d]
                matches_for_entity: List[Dict[str, Any]] = []

                # 跟所有其他子图对齐
                for tgt_sg_id, (tgt_ids, tgt_mat) in normalized_embeddings.items():
                    if tgt_sg_id == src_sg_id:
                        continue
                    if tgt_mat.size == 0:
                        continue

                    # 余弦相似度：因为已经 L2 norm 过，直接点乘
                    # sims: [n_tgt]
                    sims = (tgt_mat @ src_vec.T).reshape(-1)  # [n_tgt]

                    # 从大到小排序
                    idx_sorted = np.argsort(-sims)

                    kept_count = 0
                    for idx in idx_sorted:
                        sim = float(sims[idx])
                        if sim < sim_threshold:
                            # 因为已经按从大到小排，后面的都更小，可以直接 break
                            break

                        matches_for_entity.append(
                            {
                                "target_subgraph": tgt_sg_id,
                                "target_entity": tgt_ids[idx],
                                "similarity": sim,
                            }
                        )
                        kept_count += 1
                        if kept_count >= top_k:
                            break

                if matches_for_entity:
                    sg_align[src_eid] = matches_for_entity

            alignment_result[src_sg_id] = sg_align
            self.logger.info(
                f"[EntityAlign] subgraph={src_sg_id}, aligned_entities={len(sg_align)}"
            )

        self.entity_alignment = alignment_result
        self.logger.info(
            f"[EntityAlign] Done. #subgraphs={len(self.entity_alignment)}"
        )
    def propagate_embeddings_with_hypergraph(self, alpha: float = 0.5):
        """
        使用超图 (H) 对每个子图里的实体 embedding 做一次
        无参数卷积传播，并更新 self.subgraph_entity_embeddings。

        alpha: 残差权重，最终 X_final = alpha * X + (1 - alpha) * X_propagated
        """
        for sg_id, hyper_info in self.subgraph_hypergraphs.items():
            H = hyper_info["H"]              # [N, M] numpy
            # 普通图信息
            adj, id2idx = self.subgraph_adj.get(sg_id, (None, None))
            ent_embeds = self.subgraph_entity_embeddings.get(sg_id, {})
            # 1. 组织成一个 [N, d] 的矩阵 X，按 id2idx 的顺序
            n = len(id2idx)
            # 拿一个样本向量确定维度
            sample_vec = next(iter(ent_embeds.values()))
            d = sample_vec.shape[0]
            X = np.zeros((n, d), dtype=np.float32)
            for eid, idx in id2idx.items():
                vec = ent_embeds.get(eid, None)
                X[idx] = np.array(vec, dtype=np.float32)
            # 2. 计算节点度 / 超边度
            # H: [N, M]
            dv = H.sum(axis=1, keepdims=True)  # [N, 1]
            de = H.sum(axis=0, keepdims=True)  # [1, M]
            dv[dv == 0] = 1.0
            de[de == 0] = 1.0

            # 3. 无参数的超图卷积: Node -> Hyperedge -> Node
            # Node -> Hyperedge
            X_norm = X / dv                    # [N, d]
            Xe = H.T @ X_norm                  # [M, d]
            Xe = Xe / de.T                     # [M, d]

            # Hyperedge -> Node
            X_propagated = H @ Xe              # [N, d]

            # 4. 残差融合：保持一点原始语义
            X_final = alpha * X + (1.0 - alpha) * X_propagated

            # 5. 写回 self.subgraph_entity_embeddings
            updated_ent_embeds: Dict[str, Embedding] = {}
            for eid, idx in id2idx.items():
                updated_ent_embeds[eid] = X_final[idx]
            self.subgraph_entity_embeddings[sg_id] = updated_ent_embeds

            self.logger.info(
                f"[HypergraphProp] subgraph={sg_id}, alpha={alpha}, "
                f"updated {len(updated_ent_embeds)} entity embeddings."
            )
    def build_hypergraph_for_subgraph(
        self,
        subgraph: Subgraph,
        id2idx: Dict[str, int],
        adj: np.ndarray,
        ent_embeds: Dict[str, Embedding],
    ):
        """
        基于实体图 (adj) 构建“以实体为中心的超图”：
        - 每个实体 i 作为一个中心，形成一个超边 e_i
        - e_i 包含: i 本身 + 所有与 i 有边的邻居节点
        返回:
            H: np.ndarray, shape = [n_nodes, n_hyperedges]
            center_ids: List[str], 每个超边的中心实体 id
            hyperedge_embeds: np.ndarray[m, d] (如果有 embedding，否则 None)
        """
        n = adj.shape[0]
        # 反向索引 idx -> entity_id
        idx2id = {idx: eid for eid, idx in id2idx.items()}
        hyperedges: List[List[int]] = []
        center_ids: List[str] = []

        for center_idx in range(n):
            center_eid = idx2id[center_idx]
            # 找到所有与 center_idx 相连的邻居
            neighbor_idxs = np.nonzero(adj[center_idx])[0].tolist()
            # 超边 = 中心 + 邻居（去重）
            nodes = [center_idx] + neighbor_idxs
            nodes = sorted(set(nodes))
            hyperedges.append(nodes)
            center_ids.append(center_eid)
        m = len(hyperedges)
        H = np.zeros((n, m), dtype=np.float32)
        for e_idx, nodes in enumerate(hyperedges):
            H[nodes, e_idx] = 1.0
        # 给每个超边一个初始 embedding：使用中心实体的 embedding
        # 超边数 m，假设 embedding 维度 d
        hyperedge_embeds = None
        if ent_embeds:
            # 拿一个示例向量推断维度
            sample_vec = next(iter(ent_embeds.values()))
            d = sample_vec.shape[0] if hasattr(sample_vec, "shape") else len(sample_vec)
            hyperedge_embeds = np.zeros((m, d), dtype=np.float32)

            for e_idx, center_eid in enumerate(center_ids):
                vec = ent_embeds.get(center_eid, None)
                hyperedge_embeds[e_idx] = np.array(vec, dtype=np.float32)

        return H, center_ids, hyperedge_embeds
    def build_adj_for_subgraph(self, subgraph: Subgraph):
        """
        给一个子图（Memory 里的 Subgraph 对象），返回：
        - id2idx: entity_id -> 行列索引
        - adj: 邻接矩阵 (numpy.ndarray, shape = [n_entities, n_entities])
        """
        # 1. 所有实体，构建 entity_id -> idx
        entities = subgraph.get_entities()   # 这里应该是 KGEntity 的列表
        relations = subgraph.get_relations() # 这里应该是 KGTriple 的列表

        id2idx: Dict[str, int] = {}
        for idx, ent in enumerate(entities):
            eid = ent.get_id()
            id2idx[eid] = idx

        n = len(id2idx)
        adj = np.zeros((n, n), dtype=int)
        # 2. 遍历关系，用 subject/object 里的 entity_id 建边
        for rel in relations:
            subj = rel.get_subject()   # 预期是 KGEntity 或 None
            obj  = rel.get_object()
            # 有些 triple 的 subject / object 可能是 None（比如你 JSON 里看到的 null），要先过滤掉
            if subj is None or obj is None:
                continue
            subj=KGEntity.from_dict(subj)
            obj=KGEntity.from_dict(obj)
            head_id = subj.get_id()
            tail_id = obj.get_id()

            i = id2idx[head_id]
            j = id2idx[tail_id]

            adj[i, j] += 1
            adj[j, i] += 1
        return id2idx, adj
            
    def _encode_text(self, text: str):
        if not self.biobert_model or not self.biobert_tokenizer:
            self.logger.info(f"[EntityNormalize][BioBERT] model or tokenizer not loaded")
            return None
        with torch.no_grad():
            inputs = self.biobert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            outputs = self.biobert_model(**inputs)
            vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return vec
    def _load_biobert(self) -> None:
        try:
            self.biobert_tokenizer = AutoTokenizer.from_pretrained(
                    self.biobert_dir,
                    local_files_only=True,
                )
            self.biobert_model = AutoModel.from_pretrained(
                    self.biobert_dir,
                    local_files_only=True,
                )
            self.biobert_model.eval()
        except Exception as e:
            self.logger.info(f"[EntityNormalize][BioBERT] load failed ({e}), skip similarity-based suggestions.")
    