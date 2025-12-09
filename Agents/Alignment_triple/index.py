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
        # 子图 -> (adj, id2idx)
        self.subgraph_adj: Dict[str, Tuple[np.ndarray, Dict[str, int]]] = {}
        # 子图 -> 超图信息
        self.subgraph_hypergraphs: Dict[str, Dict[str, Any]] = {}
        # 对齐结果：{anchor_subgraph: {anchor_entity_id: [ {...}, ... ]}}
        self.entity_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        # 子图 -> {entity_id -> KGEntity}
        self.subgraph_entities: Dict[str, Dict[str, KGEntity]] = {}
        
    def process(self) -> None:
        for sg_id, sg in self.memory.subgraphs.items():
            #如果关系和实体为空则跳过
            if sg.entities.all()==[]:
                self.logger.info(f"AlignmentTripleAgent: Subgraph {sg_id} has no entities, skipping.")
                continue
            if sg.get_relations()==[]:
                self.logger.info(f"AlignmentTripleAgent: Subgraph {sg_id} has no relationships, skipping.")
                continue
            #实体嵌入
            ent_embeds: Dict[str, Embedding] = {}
            #实体map
            ent_map: Dict[str, KGEntity] = {}
            #录制子图实体的embedding
            for ent in sg.entities.all():
                text = ent.description or ent.name or ent.normalized_id
                embedding = self._encode_text(text)  # 返回 List[float] 或 np.ndarray
                ent_embeds[ent.entity_id] = embedding
                ent_map[ent.entity_id] = ent  # 记录实体对象
            # 这里的 sg 在类型系统里就是 Subgraph
           
            # 记录：子图实体 embedding / 实体 map
            self.subgraph_entity_embeddings[sg_id] = ent_embeds
            self.subgraph_entities[sg_id] = ent_map
            # 邻接
            id2idx, adj = self.build_adj_for_subgraph(sg)
            self.subgraph_adj[sg_id] = (adj, id2idx)
            #创建超图
            H, center_ids, hyperedge_embeds = self.build_hypergraph_for_subgraph(
                sg, id2idx, adj, ent_embeds
            )
            #保存超图embedding
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
        #聚合超图
        self.propagate_embeddings_with_hypergraph(alpha=0.5)
        #对齐
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

        # 2. 目标子图文本 + 候选实体
        tgt_text = self.memory.subgraphs.get(tgt_sg_id).get_meta().get("text", "")
        tgt_items = []
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
            "source_entity_name": src_entity.get_name() if src_entity else "",
            "target_subgraph": tgt_sg_id,
            "target_subgraph_text": tgt_text,
            "candidates": tgt_items,
            "instruction": (
                "Read the source entity and the candidate entities. "
                "Decide which candidates refer to the SAME real-world biomedical entity "
                "as the source. Return a JSON object with a single key 'keep', "
                "whose value is a list of candidate ids to keep. "
                "Example: {\"keep\": [\"cand1\", \"cand3\"]}."
            ),
        }

        response = self.call_llm(prompt=json.dumps(user_payload, ensure_ascii=False))
        self.logger.debug(
            f"[EntityAlign-LLM] raw response for ({src_sg_id}, {src_eid}, {tgt_sg_id}) = {response[:400]!r}"
        )

        # 5. 解析 JSON，只保留被 LLM 选中的候选
        keep_ids: List[str] = []
        try:
            parsed = self.parse_json(response)
            if isinstance(parsed, dict):
                keep_ids = [str(x) for x in parsed.get("keep", [])]
            elif isinstance(parsed, list):
                # 极端情况：LLM 直接给了一个 id 列表
                keep_ids = [str(x) for x in parsed]
            else:
                keep_ids = []
        except Exception as e:
            self.logger.warning(
                f"[EntityAlign-LLM] parse JSON failed for ({src_sg_id}, {src_eid}, {tgt_sg_id}): "
                f"raw_content={response!r}, error={e}"
            )
            return []

        id_set = set(keep_ids)
        kept: List[Dict[str, Any]] = []
        for cand in candidates:
            if cand["target_entity"] in id_set:
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
    def build_entity_alignment(
        self,
        sim_threshold: float = 0.7,
        top_k: int = 10,
        max_workers: int = 8,
    ) -> None:
        """
        顺序融合对齐：

        1）选“第一个子图”为 anchor / canonical；
        2）把它视为“融合子图 F”，其实体作为全局 canonical；
        3）依次遍历剩余子图 S_i：
            - 动态看 |F| 和 |S_i| 的大小：
              * 如果 |F| <= |S_i|：用 F 的实体做 source；
              * 否则用 S_i 的实体做 source；
            - 做一次余弦相似度候选筛选；
            - 调 _run_llm_alignment_parallel 对这一对(F, S_i)做精筛；
            - 把对齐结果统一映射到 anchor_sg_id 的 canonical 实体上；
            - 同时用 S_i 实体的 embedding 对 canonical embedding 做简单融合（0.5 * old + 0.5 * new）。
        4）最终 self.entity_alignment 只用 anchor_sg_id 做 key，结构：
           {anchor_sg_id: {canonical_eid: [ {target_subgraph, target_entity, similarity, llm_agree}, ... ]}}
        """
        # 辅助函数：把 raw embedding dict 变成 (ids, normalized_mat)
        def _normalize_embeds(embeds: Dict[str, Embedding]) -> Tuple[List[str], np.ndarray]:
            ids: List[str] = []
            vecs: List[np.ndarray] = []
            for eid, emb in embeds.items():
                v = np.asarray(emb, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm == 0.0:
                    continue
                v = v / norm
                ids.append(eid)
                vecs.append(v)
            if not ids:
                return [], np.zeros((0, 1), dtype=np.float32)
            mat = np.stack(vecs, axis=0)
            return ids, mat

        if not self.subgraph_entity_embeddings:
            return

        # 1）选 anchor / canonical 子图
        all_sg_ids = list(self.subgraph_entity_embeddings.keys())
        anchor_sg_id = all_sg_ids[0]
        self.logger.info(f"[EntityAlign] anchor_subgraph={anchor_sg_id}")

        # canonical 的“raw” embedding，会在后面被融合更新
        canonical_raw_embeds: Dict[str, Embedding] = dict(self.subgraph_entity_embeddings[anchor_sg_id])

        # 对齐结果（统一以 anchor_sg_id 为 key）
        global_alignment: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            anchor_sg_id: {}
        }

        # 2）依次用其余子图做顺序对齐
        for tgt_sg_id in all_sg_ids[1:]:
            if tgt_sg_id not in self.subgraph_entity_embeddings:
                continue
            tgt_raw_embeds = self.subgraph_entity_embeddings[tgt_sg_id]
            if not tgt_raw_embeds:
                continue

            # 当前轮次的 canonical / target 的归一化向量
            canon_ids, canon_mat = _normalize_embeds(canonical_raw_embeds)
            tgt_ids, tgt_mat = _normalize_embeds(tgt_raw_embeds)

            if len(canon_ids) == 0 or len(tgt_ids) == 0:
                continue

            self.logger.info(
                f"[EntityAlign] step align: F(anchor={anchor_sg_id}, |F|={len(canon_ids)}) "
                f"<-> S({tgt_sg_id}, |S|={len(tgt_ids)})"
            )

            # 决定谁当 source（小的当 source）
            source_is_canonical = len(canon_ids) <= len(tgt_ids)

            candidate_alignment_step: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

            if source_is_canonical:
                # F 作为 source，对每个 canonical 实体去 S_i 里找候选
                src_sg_id = anchor_sg_id
                ent_map: Dict[str, List[Dict[str, Any]]] = {}

                for i, src_eid in enumerate(canon_ids):
                    src_vec = canon_mat[i : i + 1, :]  # [1, d]
                    sims = (tgt_mat @ src_vec.T).reshape(-1)  # [n_tgt]
                    idx_sorted = np.argsort(-sims)

                    matches_for_entity: List[Dict[str, Any]] = []
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
                        ent_map[src_eid] = matches_for_entity

                if ent_map:
                    candidate_alignment_step[src_sg_id] = ent_map

            else:
                # S_i 作为 source，对每个 S_i 的实体去 F 里找候选
                src_sg_id = tgt_sg_id
                ent_map: Dict[str, List[Dict[str, Any]]] = {}

                for j, src_eid in enumerate(tgt_ids):
                    src_vec = tgt_mat[j : j + 1, :]  # [1, d]
                    sims = (canon_mat @ src_vec.T).reshape(-1)  # [n_canon]
                    idx_sorted = np.argsort(-sims)

                    matches_for_entity: List[Dict[str, Any]] = []
                    kept_count = 0
                    for idx in idx_sorted:
                        sim = float(sims[idx])
                        if sim < sim_threshold:
                            break
                        matches_for_entity.append(
                            {
                                "target_subgraph": anchor_sg_id,
                                "target_entity": canon_ids[idx],
                                "similarity": sim,
                            }
                        )
                        kept_count += 1
                        if kept_count >= top_k:
                            break

                    if matches_for_entity:
                        ent_map[src_eid] = matches_for_entity

                if ent_map:
                    candidate_alignment_step[src_sg_id] = ent_map

            if not candidate_alignment_step:
                self.logger.info(
                    f"[EntityAlign] step ({anchor_sg_id} vs {tgt_sg_id}) has no cosine candidates above threshold."
                )
                continue

            # ---------- 调一次 LLM 对这一对 (F, S_i) 做精筛 ----------
            refined_step = self._run_llm_alignment_parallel(
                candidate_alignment_step,
                max_workers=max_workers,
            )

            if not refined_step:
                self.logger.info(
                    f"[EntityAlign-LLM] step ({anchor_sg_id} vs {tgt_sg_id}) LLM kept nothing."
                )
                continue

            # ---------- 把 refined_step 统一映射到 anchor_sg_id 的 canonical 上，并融合 embedding ----------
            if source_is_canonical:
                # refined_step: {anchor_sg_id: {canonical_eid: [ {tgt_sg_id, tgt_eid}, ... ]}}
                per_src = refined_step.get(anchor_sg_id, {})
                for canon_eid, kept_list in per_src.items():
                    if not kept_list:
                        continue
                    # 累积对齐结果（anchor_sg_id -> canonical_eid）
                    global_alignment[anchor_sg_id].setdefault(canon_eid, []).extend(kept_list)

                    # 用 S_i 的 embedding 融合更新 canonical embedding
                    old_vec = np.asarray(canonical_raw_embeds.get(canon_eid), dtype=np.float32)
                    for m in kept_list:
                        tgt_eid = m["target_entity"]
                        tgt_vec = np.asarray(tgt_raw_embeds.get(tgt_eid), dtype=np.float32)
                        if tgt_vec.size == 0 or old_vec.size == 0:
                            continue
                        old_vec = 0.5 * old_vec + 0.5 * tgt_vec
                    canonical_raw_embeds[canon_eid] = old_vec

            else:
                # refined_step: {tgt_sg_id: {src_eid_in_Si: [ {target_subgraph=anchor, target_entity=canon_eid}, ... ]}}
                per_src = refined_step.get(tgt_sg_id, {})
                for src_eid_in_S, kept_list in per_src.items():
                    if not kept_list:
                        continue
                    for m in kept_list:
                        canon_eid = m["target_entity"]
                        # 反向存储：canonical_eid -> (tgt_sg_id, src_eid_in_S)
                        inv_match = {
                            "target_subgraph": tgt_sg_id,
                            "target_entity": src_eid_in_S,
                            "similarity": m.get("similarity", 0.0),
                            "llm_agree": m.get("llm_agree", True),
                        }
                        global_alignment[anchor_sg_id].setdefault(canon_eid, []).append(inv_match)

                        # 融合 embedding: canonical <- S_i 实体
                        old_vec = np.asarray(canonical_raw_embeds.get(canon_eid), dtype=np.float32)
                        src_vec = np.asarray(tgt_raw_embeds.get(src_eid_in_S), dtype=np.float32)
                        if src_vec.size == 0 or old_vec.size == 0:
                            continue
                        old_vec = 0.5 * old_vec + 0.5 * src_vec
                        canonical_raw_embeds[canon_eid] = old_vec

        # 最终结果
        self.entity_alignment = global_alignment
        # 可选：也把融合后的 canonical embedding 回写到 subgraph_entity_embeddings[anchor_sg_id]
        self.subgraph_entity_embeddings[anchor_sg_id] = canonical_raw_embeds

        # 写回 Memory 的对齐存储
        self.memory.alignments.save_from_alignment_dict(global_alignment)
        self.logger.info(
            f"[EntityAlign-LLM] Done. anchor_subgraph={anchor_sg_id}, "
            f"#canonical_entities={len(global_alignment.get(anchor_sg_id, {}))}"
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