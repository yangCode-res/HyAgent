# Agents/Entity_normalize/index.py

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, Subgraph
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity

logger = get_global_logger()

# 控制台颜色（如果重定向到文件，只是普通字符串，不影响）
ANSI_RESET = "\033[0m"
ANSI_CYAN = "\033[96m"    # LLM 相关
ANSI_GREEN = "\033[92m"   # 汇总信息

"""
实体归一化 Agent。
基于已有的子图实体，进行实体归一化合并，减少冗余节点。
输入: 无（从内存中获取子图实体）
输出: 无（将归一化合并后的实体更新回内存的子图）
调用入口：agent.process()
归一化流程包括三步：
1.规则归一化（同子图 + 同类型，确定性字符串匹配）
2.BioBERT 相似度候选（同子图 + 同类型，基于 description/name）
3.LLM 裁决合并（候选批次并行请求，合并操作串行执行）
"""

class EntityNormalizationAgent(Agent):
    """
    子图级实体归一化 Agent

    三步流程：

    1）规则归一化（同子图 + 同类型，确定性字符串匹配）
    2）BioBERT 相似度候选（同子图 + 同类型，基于 description/name）
    3）LLM 裁决合并（候选批次并行请求，合并操作串行执行）
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        biobert_dir: str = "/home/nas2/path/models/biobert-base-cased-v1.1",
        sim_threshold: float = 0.94,
    ):
        system_prompt = """
You are a specialized Entity Normalization Agent for biomedical literature.

You are given:
- Local subgraphs built from biomedical texts.
- Entities with:
    - id, type, name, normalized_id (may be N/A),
    - aliases,
    - description: a short definition-like summary.
- High-similarity candidate pairs proposed by a BioBERT encoder.

Your job in the final step:
For each candidate pair, decide whether they refer to the SAME underlying biomedical entity
(should be merged as one node in a knowledge graph) or are DISTINCT but related entities.

Guidelines:
1. Only answer based on the given fields (name, aliases, type, description, normalized_id).
2. Prefer MERGE when:
   - Names and descriptions clearly refer to the same concept / synonym / abbreviation / variant.
   - Or one is a more specific surface form of the other, but not meaningfully distinct in KG granularity.
3. Prefer NO_MERGE when:
   - One is a cause, regulator, or downstream effect of the other.
   - One is a disease and the other is a biomarker, pathway, phenotype, mechanism, or drug.
   - They occupy clearly different roles (e.g., hyperglycemia vs atherosclerosis).
   - Types conflict in a meaningful way.
4. Be precise and conservative. Do NOT merge just because text is similar or they co-occur.
5. Output STRICT JSON ONLY, no comments or extra text.
"""
        super().__init__(client, model_name, system_prompt)
        self.logger = get_global_logger()

        self.sim_threshold = float(sim_threshold)

        # BioBERT（本地加载，仅用于相似度）
        self.biobert_dir = biobert_dir
        self.biobert_tokenizer = None
        self.biobert_model = None
        self._load_biobert()

    # ===================== 对外入口 =====================

    def process(self, memory: Memory) -> None:
        """
        对所有 subgraph 执行：
        1）规则归一化；
        2）用 BioBERT 生成候选对；
        3）并行 LLM 裁决；
        带总体进度条。
        """
        if not memory.subgraphs:
            logger.info("[EntityNormalize] no subgraphs found in memory, skip.")
            return

        total_before = 0
        total_after = 0
        total_llm_merged = 0

        subgraph_items = list(memory.subgraphs.items())

        for sg_id, sg in tqdm(
            subgraph_items,
            desc="EntityNormalize | subgraphs",
            unit="sg"
        ):
            before = len(sg.entities.all())
            if before == 0:
                continue

            # 1) 精确规则合并
            merged_rule = self._normalize_subgraph_entities(sg)
            after_rule = len(sg.entities.all())

            # 2) BioBERT 候选对
            candidates: List[Dict[str, Any]] = []
            if self.biobert_model is not None:
                candidates = self._collect_biobert_candidate_pairs(sg)
            else:
                logger.info(
                    f"[EntityNormalize][BioBERT] subgraph={sg_id} skipped: BioBERT model not loaded."
                )

            # 3) LLM 裁决并合并（内部带 batch 进度条）
            llm_merged = 0
            if candidates:
                logger.info(
                    f"{ANSI_CYAN}[EntityNormalize][LLM] subgraph={sg_id} "
                    f"received {len(candidates)} candidate pairs from BioBERT, "
                    f"sending to LLM in parallel...{ANSI_RESET}"
                )
                llm_merged = self._llm_decide_and_merge(sg, candidates)
            else:
                logger.info(
                    f"[EntityNormalize][LLM] subgraph={sg_id} no candidates passed to LLM."
                )

            after_all = len(sg.entities.all())

            total_before += before
            total_after += after_all
            total_llm_merged += llm_merged

            logger.info(
                f"{ANSI_GREEN}[EntityNormalize] subgraph={sg_id} "
                f"entities_before={before} "
                f"after_rule={after_rule} "
                f"rule_merged={merged_rule} "
                f"llm_merged={llm_merged} "
                f"entities_after_all={after_all}{ANSI_RESET}"
            )

        logger.info(
            f"{ANSI_GREEN}[EntityNormalize] done. total_before={total_before}, "
            f"total_after={total_after}, "
            f"total_delta={total_before - total_after}, "
            f"total_llm_merged={total_llm_merged}{ANSI_RESET}"
        )

    # ===================== 子图内：规则合并 =====================

    def _normalize_subgraph_entities(self, sg: Subgraph) -> int:
        entities: List[KGEntity] = sg.entities.all()
        if len(entities) <= 1:
            return 0

        idx_to_ent: Dict[int, KGEntity] = {i: e for i, e in enumerate(entities)}

        # 按类型分组
        type_to_indices: Dict[str, List[int]] = {}
        for idx, ent in idx_to_ent.items():
            etype = ent.entity_type or "Unknown"
            type_to_indices.setdefault(etype, []).append(idx)

        merged_count = 0
        for etype, indices in type_to_indices.items():
            if len(indices) <= 1:
                continue
            merged_count += self._merge_by_string_keys_within_type(sg, idx_to_ent, indices)

        return merged_count

    def _merge_by_string_keys_within_type(
        self,
        sg: Subgraph,
        idx_to_ent: Dict[int, KGEntity],
        indices: List[int],
    ) -> int:
        if len(indices) <= 1:
            return 0

        parent = {i: i for i in indices}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        norm_map: Dict[str, int] = {}

        for idx in indices:
            ent = idx_to_ent[idx]

            surfaces: List[str] = []
            if ent.name:
                surfaces.append(ent.name)
            if ent.aliases:
                surfaces.extend(ent.aliases)

            for s in surfaces:
                norm = _normalize_str(s)
                if not norm:
                    continue

                if norm in norm_map:
                    union(idx, norm_map[norm])
                else:
                    norm_map[norm] = idx

        # 收集连通块
        comp: Dict[int, List[int]] = {}
        for idx in indices:
            root = find(idx)
            comp.setdefault(root, []).append(idx)

        merged_count = 0

        # 每个连通块合并
        for _, group in comp.items():
            if len(group) <= 1:
                continue

            leader_idx = self._choose_leader(idx_to_ent, group)
            leader = idx_to_ent[leader_idx]

            alias_set = set(_safe_list(leader.aliases))
            leader_norm = _normalize_str(leader.name)

            for idx in group:
                if idx == leader_idx:
                    continue

                ent = idx_to_ent[idx]

                if (not _has_valid_norm_id(leader)) and _has_valid_norm_id(ent):
                    leader.normalized_id = ent.normalized_id

                if ent.name:
                    alias_set.add(ent.name.strip())
                for a in _safe_list(ent.aliases):
                    if a:
                        alias_set.add(a.strip())

                if ent.entity_id in sg.entities.by_id:
                    del sg.entities.by_id[ent.entity_id]

                merged_count += 1

            cleaned_aliases: List[str] = []
            for a in alias_set:
                if not a:
                    continue
                if _normalize_str(a) == leader_norm:
                    continue
                cleaned_aliases.append(a)
            leader.aliases = sorted(set(cleaned_aliases), key=lambda x: x.lower())

        return merged_count

    def _choose_leader(
        self,
        idx_to_ent: Dict[int, KGEntity],
        group: List[int],
    ) -> int:
        """
        在实体簇里选代表：
        1) 有 normalized_id 优先；
        2) 名字更长优先；
        3) 别名更多优先。
        """
        best_idx = group[0]
        best_ent = idx_to_ent[best_idx]

        def score(e: KGEntity) -> Tuple[int, int, int]:
            has_id = 1 if _has_valid_norm_id(e) else 0
            name_len = len(e.name or "")
            alias_count = len(e.aliases or [])
            return has_id, name_len, alias_count

        best_score = score(best_ent)
        for idx in group[1:]:
            ent = idx_to_ent[idx]
            sc = score(ent)
            if sc > best_score:
                best_idx = idx
                best_ent = ent
                best_score = sc

        return best_idx

    # ===================== BioBERT 候选收集 =====================

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
            logger.info(f"[EntityNormalize][BioBERT] loaded from {self.biobert_dir}")
        except Exception as e:
            self.biobert_tokenizer = None
            self.biobert_model = None
            logger.info(
                f"[EntityNormalize][BioBERT] load failed ({e}), skip similarity-based suggestions."
            )

    def _encode_text(self, text: str):
        if not self.biobert_model or not self.biobert_tokenizer:
            return None
        if not text:
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

    def _get_ent_text(self, ent: KGEntity) -> str:
        desc = getattr(ent, "description", None)
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
        if isinstance(ent.name, str) and ent.name.strip():
            return ent.name.strip()
        return ""

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _collect_biobert_candidate_pairs(self, sg: Subgraph) -> List[Dict[str, Any]]:
        entities: List[KGEntity] = sg.entities.all()
        if len(entities) <= 1 or not self.biobert_model:
            return []

        type_to_ents: Dict[str, List[KGEntity]] = {}
        for e in entities:
            etype = e.entity_type or "Unknown"
            type_to_ents.setdefault(etype, []).append(e)

        all_candidates: List[Dict[str, Any]] = []

        for etype, ents in type_to_ents.items():
            if len(ents) <= 1:
                continue

            texts: List[str] = []
            vecs: List[np.ndarray] = []
            for ent in ents:
                txt = self._get_ent_text(ent)
                texts.append(txt)
                vecs.append(self._encode_text(txt) if txt else None)

            n = len(ents)
            for i in range(n):
                vi = vecs[i]
                if vi is None:
                    continue
                for j in range(i + 1, n):
                    vj = vecs[j]
                    if vj is None:
                        continue

                    sim = self._cosine(vi, vj)
                    if sim < self.sim_threshold:
                        continue

                    ea, eb = ents[i], ents[j]
                    all_candidates.append(
                        {
                            "subgraph_id": sg.id,
                            "entity_type": etype,
                            "similarity": float(sim),
                            "ent_a_id": ea.entity_id,
                            "ent_b_id": eb.entity_id,
                            "ent_a_name": ea.name,
                            "ent_b_name": eb.name,
                            "ent_a_normalized_id": getattr(ea, "normalized_id", "N/A"),
                            "ent_b_normalized_id": getattr(eb, "normalized_id", "N/A"),
                            "ent_a_aliases": list(getattr(ea, "aliases", []) or []),
                            "ent_b_aliases": list(getattr(eb, "aliases", []) or []),
                            "ent_a_description": self._get_ent_text(ea),
                            "ent_b_description": self._get_ent_text(eb),
                        }
                    )

        if all_candidates:
            logger.info(
                f"[EntityNormalize][BioBERT] subgraph={sg.id} "
                f"collected_candidates>={self.sim_threshold}: {len(all_candidates)}"
            )
        return all_candidates

    # ===================== LLM 裁决（并行查询 + 进度条，串行合并） =====================

    def _llm_decide_and_merge(
        self,
        sg: Subgraph,
        candidates: List[Dict[str, Any]],
        batch_size: int = 40,
    ) -> int:
        if not candidates:
            return 0

        # 构造 batch prompts
        batches: List[List[Dict[str, Any]]] = []
        prompts: List[str] = []
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i: i + batch_size]
            batches.append(batch)

            prompt_obj = {
                "subgraph_id": sg.id,
                "instructions": (
                    "For each candidate pair, decide if they are the SAME entity (MERGE) "
                    "or DIFFERENT entities (NO_MERGE). "
                    "Follow the guidelines in the system prompt. "
                    "Return ONLY a JSON array. "
                    "Each item must be:\n"
                    "{\n"
                    '  \"ent_a_id\": \"...\",\n'
                    '  \"ent_b_id\": \"...\",\n'
                    '  \"decision\": \"merge\" or \"no_merge\",\n'
                    '  \"reason\": \"short explanation\"\n'
                    "}\n"
                ),
                "candidates": batch,
            }
            prompts.append(str(prompt_obj))

        num_batches = len(prompts)
        if num_batches == 0:
            return 0

        results: List[Any] = [None] * num_batches
        max_workers = min(8, num_batches)

        # 并行请求 + 批次进度条
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
            total=num_batches,
            desc=f"LLM merge | sg={sg.id}",
            unit="batch"
        ) as pbar:
            future_to_idx = {
                executor.submit(self.call_llm_json_safe, prompts[idx]): idx
                for idx in range(num_batches)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    data = fut.result()
                except Exception as e:
                    logger.info(
                        f"[EntityNormalize][LLM] subgraph={sg.id} batch_{idx} call failed: {e}"
                    )
                    data = []
                results[idx] = data
                pbar.update(1)

        # 串行应用合并决策
        merged_count = 0

        for batch_idx, raw in enumerate(results):
            if not raw:
                continue
            if not isinstance(raw, List):
                logger.info(
                    f"[EntityNormalize][LLM] subgraph={sg.id} "
                    f"batch_{batch_idx} invalid LLM output type: {type(raw)}"
                )
                continue

            for item in raw:
                try:
                    ent_a_id = str(item.get("ent_a_id", "")).strip()
                    ent_b_id = str(item.get("ent_b_id", "")).strip()
                    decision = str(item.get("decision", "")).strip().lower()
                except Exception:
                    continue

                if decision != "merge":
                    continue
                if not ent_a_id or not ent_b_id or ent_a_id == ent_b_id:
                    continue

                ea = sg.entities.by_id.get(ent_a_id)
                eb = sg.entities.by_id.get(ent_b_id)
                if ea is None or eb is None:
                    continue  # 可能之前已被合并删除

                leader, removed = self._merge_two_entities(sg, ea, eb)
                if removed:
                    merged_count += 1
                    logger.info(
                        f"{ANSI_CYAN}[EntityNormalize][LLM] subgraph={sg.id} "
                        f"MERGED {removed.entity_id} -> {leader.entity_id} | "
                        f"{removed.name} -> {leader.name}{ANSI_RESET}"
                    )

        return merged_count

    def _merge_two_entities(
        self,
        sg: Subgraph,
        ea: KGEntity,
        eb: KGEntity,
    ) -> Tuple[KGEntity, KGEntity | None]:
        idx_to_ent = {0: ea, 1: eb}
        leader_idx = self._choose_leader(idx_to_ent, [0, 1])
        leader = idx_to_ent[leader_idx]
        follower = eb if leader is ea else ea

        if follower.entity_id == leader.entity_id:
            return leader, None

        if (not _has_valid_norm_id(leader)) and _has_valid_norm_id(follower):
            leader.normalized_id = follower.normalized_id

        alias_set = set(_safe_list(leader.aliases))
        if follower.name:
            alias_set.add(follower.name.strip())
        for a in _safe_list(follower.aliases):
            if a:
                alias_set.add(a.strip())

        leader_norm = _normalize_str(leader.name)
        cleaned_aliases = []
        for a in alias_set:
            if not a:
                continue
            if _normalize_str(a) == leader_norm:
                continue
            cleaned_aliases.append(a)
        leader.aliases = sorted(set(cleaned_aliases), key=lambda x: x.lower())

        if follower.entity_id in sg.entities.by_id:
            del sg.entities.by_id[follower.entity_id]

        return leader, follower

    # ===================== LLM 调用封装 =====================

    def call_llm_json_safe(self, content: Any) -> Any:
        """
        调用 LLM 并解析为 JSON 数组。用于多线程环境，只做读取，不修改共享状态。
        """
        if isinstance(content, (dict, list)):
            prompt = (
                "You will receive a JSON-like object describing candidate entity pairs.\n"
                "Respond ONLY with a JSON array as specified.\n\n"
                f"{content}"
            )
        else:
            prompt = str(content)

        raw = self.call_llm(prompt)
        try:
            data = self.parse_json(raw)
        except Exception as e:
            logger.info(f"[EntityNormalize][LLM] parse_json failed: {e}")
            return []
        return data


# ===================== 小工具函数 =====================

def _normalize_str(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def _safe_list(x):
    return x if isinstance(x, list) else []


def _has_valid_norm_id(e: KGEntity) -> bool:
    nid = getattr(e, "normalized_id", None)
    return bool(nid) and str(nid).strip().upper() != "N/A"