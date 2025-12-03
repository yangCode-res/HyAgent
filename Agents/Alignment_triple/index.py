import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer

from Config.index import BioBertPath
from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, Subgraph
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple

Embedding = List[float]
"""
实体对齐三元组 Agent。
基于子图中的实体嵌入和文本描述，识别不同子图中表示同一实体的实体对齐三元组。
输入:无（从内存中获取子图和实体信息）
输出:无（将识别的实体对齐三元组存储到内存中的对齐存储中）
调用入口：agent.process()
"""
class AlignmentTripleAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str,memory:Optional[Memory]=None):
        self.system_prompt = """
You are an expert in biomedical knowledge graph entity alignment.

You will receive a single JSON string as user input, with fields such as:
- "source_subgraph", "source_entity_id", "source_entity_name", "source_entity_text"
- "target_subgraph", "target_subgraph_text"
- "candidates": a list of objects { "id": ..., "name": ... }
- "instruction": a natural language description of the task

Your task:
1. Parse the JSON input.
2. Compare the source entity with all candidate entities from the target subgraph.
3. Decide which candidates refer to the SAME real-world biomedical entity as the source entity.

Output format (VERY IMPORTANT):
- You MUST respond with STRICT JSON only.
- The JSON must have exactly one top-level key "keep".
- "keep" must be a list of candidate ids (strings) that should be kept.
- Example: {"keep": ["cand1", "cand3"]}

Rules:
- If no candidate should be aligned with the source entity, return {"keep": []}.
- Do NOT add any other keys, text, comments, or explanations.
- Do NOT change, rename, or invent candidate ids.
- The response must be valid JSON and parseable by a standard JSON parser.
"""
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
            if sg.entities.all()==[]:
                self.logger.info(f"AlignmentTripleAgent: Subgraph {sg_id} has no entities, skipping.")
                continue
            if sg.get_relations()==[]:
                self.logger.info(f"AlignmentTripleAgent: Subgraph {sg_id} has no relationships, skipping.")
                continue
            ent_embeds: Dict[str, Embedding] = {}
            ent_map: Dict[str, KGEntity] = {}
            for ent in sg.entities.all():
                text = ent.description or ent.name or ent.normalized_id
                embedding = self._encode_text(text)  # 返回 List[float] 或 np.ndarray
                ent_embeds[ent.entity_id] = embedding
                ent_map[ent.entity_id] = ent  # 记录实体对象
            # 这里的 sg 在类型系统里就是 Subgraph
            self.subgraph_entity_embeddings[sg_id] = ent_embeds
            self.subgraph_entities[sg_id] = ent_map
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
        self.build_entity_alignment(sim_threshold=0.95, top_k=5)
        # 用 LLM 做精过滤（并行）
        # for src_sg_id, ent_map in self.entity_alignment.items():
        #     for src_eid, matches in ent_map.items():
        #         print(src_sg_id, src_eid)
        #         for m in matches:
        #             print("  ->", m["target_subgraph"], m["target_entity"], m["similarity"])   
    def _llm_filter_for_one_pair(
        self,
        src_sg_id: str,
        src_eid: str,
        tgt_sg_id: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        对一个 (源子图, 源实体, 目标子图) 的候选列表，调一次 LLM：
        让 LLM 判断哪些 target_entity 真正是“同一实体”，返回保留的子集。
        """
        # 1. 取源实体文本
        src_entity = self._find_entity_in_subgraph(src_sg_id, src_eid)
        src_text =self.memory.subgraphs.get(src_sg_id).get_meta().get("text", "")
        # print('this is src_text',self.memory.subgraphs.get(src_sg_id).get_meta().get("text", ""))
        # 2. 取每个候选实体的文本
        tgt_items = []
        tgt_text = self.memory.subgraphs.get(tgt_sg_id).get_meta().get("text", "")
        for cand in candidates:
            tgt_eid = cand["target_entity"]
            tgt_entity = self._find_entity_in_subgraph(tgt_sg_id, tgt_eid)
            tgt_items.append(
                {
                    "id": tgt_eid,
                    "name":tgt_entity.get_name(),
                }
            )
        # 3. 组织 prompt
        user_payload = {
            "source_subgraph": src_sg_id,
            "source_entity_id": src_eid,
            "source_entity_text": src_text,
            "target_subgraph": tgt_sg_id,
            "source_entity_name":src_entity.get_name(),
            "candidates": [
                {
                    "id": t["id"],
                    "name": t["name"],
                }
                for t in tgt_items
            ],
            "target_subgraph_text": tgt_text,
            "instruction": (
                "Read the source entity and the candidate entities. "
                "Decide which candidates refer to the SAME real-world biomedical entity "
                "as the source. Return a JSON object with a single key 'keep', "
                "whose value is a list of candidate ids to keep. "
                "Example: {\"keep\": [\"cand1\", \"cand3\"]}."
            ),
        }

        response=self.call_llm(prompt=json.dumps(user_payload, ensure_ascii=False))
        # print('this is response',response)
        content=self.parse_json(response)
        # print('this is content',content)
        # 5. 解析 JSON，只保留被 LLM 选中的候选
        keep_ids: List[str] = []
        try:
            # obj = json.loads(content)
            keep_ids=[str(x) for x in content]
        except Exception as e:
            self.logger.warning(
                f"[EntityAlign-LLM] parse JSON failed for ({src_sg_id}, {src_eid}, {tgt_sg_id}): "
                f"raw_content={content!r}, error={e}"
            )
            # 解析失败就视为不保留任何候选
            return []
        id_set = set(keep_ids)
        kept: List[Dict[str, Any]] = []
        for cand in candidates:
            if cand["target_entity"] in id_set:
                # 可以顺带打个标记，说明是 LLM 通过的
                kept.append(
                    {
                        "target_subgraph": tgt_sg_id,
                        "target_entity": cand["target_entity"],
                        "similarity": cand["similarity"],
                        "llm_agree": True,
                    }
                )

        return kept
    def _run_llm_alignment_parallel(
        self,
        candidate_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]],
        max_workers: int = 8,
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        输入：candidate_alignment（只经过余弦筛选）
        输出：refined_alignment（经过 LLM 精筛）
        并行粒度：每个 (src_sg_id, src_eid, tgt_sg_id) 单独调一次 LLM。
        """
        # 组装 LLM job 列表
        jobs: List[Tuple[str, str, str, List[Dict[str, Any]]]] = []

        for src_sg_id, ent_map in candidate_alignment.items():
            for src_eid, matches in ent_map.items():
                # 按 target_subgraph 重新分组：一个 job 对应一个目标子图
                by_tgt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for m in matches:
                    tgt_sg_id = m["target_subgraph"]
                    by_tgt[tgt_sg_id].append(m)

                for tgt_sg_id, cand_list in by_tgt.items():
                    # 过滤掉空的
                    if not cand_list:
                        continue
                    jobs.append((src_sg_id, src_eid, tgt_sg_id, cand_list))

        self.logger.info(f"[EntityAlign-LLM] total jobs={len(jobs)}")

        # 收集结果：键是 (src_sg_id, src_eid)
        merged_results: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

        if not jobs:
            return {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(
                    self._llm_filter_for_one_pair,
                    src_sg_id,
                    src_eid,
                    tgt_sg_id,
                    cand_list,
                ): (src_sg_id, src_eid, tgt_sg_id)
                for (src_sg_id, src_eid, tgt_sg_id, cand_list) in jobs
            }

            for future in as_completed(future_to_job):
                src_sg_id, src_eid, tgt_sg_id = future_to_job[future]
                try:
                    kept = future.result()  # List[Dict]
                except Exception as e:
                    self.logger.warning(
                        f"[EntityAlign-LLM] job ({src_sg_id}, {src_eid}, {tgt_sg_id}) "
                        f"failed: {e}"
                    )
                    kept = []

                if kept:
                    merged_results[(src_sg_id, src_eid)].extend(kept)

        # 转回 {src_sg_id: {src_eid: [..]}} 结构
        refined_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for (src_sg_id, src_eid), lst in merged_results.items():
            refined_alignment.setdefault(src_sg_id, {})[src_eid] = lst

        return refined_alignment
    def build_entity_alignment(self, sim_threshold: float = 0.7, top_k: int = 10,
                               max_workers: int = 8) -> None:
        """
        第一步：用余弦相似度在所有子图之间做候选对齐；
        第二步：对每个 (源子图, 源实体, 目标子图) 调一次 LLM 做精筛；
        最终结果写入 self.entity_alignment。
        """
        # ---------- 1. 归一化每个子图的实体向量 ----------
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


            mat = np.stack(vecs, axis=0)  # [n, d]
            normalized_embeddings[sg_id] = (ids, mat)

        # ---------- 2. 余弦相似度候选：candidate_alignment ----------
                # ---------- 2. 余弦相似度候选：candidate_alignment ----------
        candidate_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        # 选“第一个子图”作为 anchor（中心子图）
        if not normalized_embeddings:
            return
        anchor_sg_id = next(iter(normalized_embeddings.keys()))
        anchor_ids, anchor_mat = normalized_embeddings[anchor_sg_id]

        sg_align: Dict[str, List[Dict[str, Any]]] = {}

        # 只让 anchor_sg_id 作为 source，向其他子图对齐
        for i_src, src_eid in enumerate(anchor_ids):
            src_vec = anchor_mat[i_src : i_src + 1, :]  # [1, d]
            matches_for_entity: List[Dict[str, Any]] = []

            for tgt_sg_id, (tgt_ids, tgt_mat) in normalized_embeddings.items():
                # 不和自己对齐
                if tgt_sg_id == anchor_sg_id:
                    continue
                if tgt_mat.size == 0:
                    continue

                sims = (tgt_mat @ src_vec.T).reshape(-1)  # [n_tgt]
                idx_sorted = np.argsort(-sims)

                kept_count = 0
                for idx in idx_sorted:
                    sim = float(sims[idx])
                    if sim < sim_threshold:
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

        # candidate_alignment 里只会有一个 source：anchor_sg_id
        candidate_alignment[anchor_sg_id] = sg_align
        self.logger.info(
            f"[EntityAlign-Candidate] anchor_subgraph={anchor_sg_id}, "
            f"candidate_entities={len(sg_align)}"
        )
        # ---------- 3. LLM 并行精筛（关键：每个 target 子图单独调一次） ----------
        refined_alignment = self._run_llm_alignment_parallel(
            candidate_alignment,
            max_workers=max_workers,
        )
        self.entity_alignment = refined_alignment
        self.memory.alignments.save_from_alignment_dict(refined_alignment)
        self.logger.info(
            f"[EntityAlign-LLM] Done. #subgraphs={len(self.entity_alignment)}"
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
    def _find_entity_in_subgraph(self, sg_id: str, entity_id: str) -> Optional[KGEntity]:
        """
        在 Memory 里的某个 subgraph 中按 entity_id 找实体。
        如果你的 Subgraph 已经有 get_entity(entity_id) 之类的方法，可以直接改用。
        """
        sg: Subgraph = self.memory.subgraphs.get(sg_id)
        if sg is None:
            return None
        for ent in sg.entities.all():
            if getattr(ent, "entity_id", None) == entity_id:
                return ent
        return None