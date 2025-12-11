import math
import json
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from Store.index import get_memory
from Config.index import BioBertPath
from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity


class KeywordEntitySearchAgent(Agent):
    """
    功能：
      1）用 BioBERT 在 Memory 全局实体中检索与多个 keyword 最相近的实体（相似度 Top-M 作为候选池）
      2）把这 M 个候选丢给大模型，由大模型在候选中选出最多 K 个最匹配的实体
      3）结果写入 memory.keyword_entity_map[keyword] = [KGEntity, ...]

    说明：
      - 相似度是 keyword 与实体的 [name + aliases] 多个 surface 中的最大值
      - 日志：
          * 过程阶段只在警告/出错时打印 warning/error
          * 最后一次性以表格形式输出所有关键词的匹配结果
    """

    def __init__(
        self,
        client,
        model_name: str,
        keywords: List[str],            # ⭐ 多个关键词
        memory: Optional[Memory] = None,
        top_k_default: int = 3,         # ⭐ 每个 keyword 最终保留多少个实体
        candidate_pool_size: int = 10,  # ⭐ 每个 keyword 先取多少个 BioBERT 候选交给 LLM
    ):
        system_prompt = (
            "You are a biomedical entity-linking agent. "
            "Given a query keyword and a list of candidate entities from a knowledge graph, "
            "you must choose up to K candidates that best match the keyword.\n\n"
            "Always respond with STRICT JSON of the form:\n"
            "{\n"
            '  \"entity_ids\": [\"<id1>\", \"<id2>\", \"...\"]\n'
            "}\n"
            "You MUST:\n"
            "- Return at least 1 id if there is any reasonable match.\n"
            "- NEVER return more than K ids.\n"
            "- Do not include any extra fields, comments, or text outside the JSON."
        )
        super().__init__(client, model_name, system_prompt)

        self.logger = get_global_logger()
        self.memory: Memory = memory or get_memory()

        self.keywords: List[str] = keywords
        self.top_k_default: int = top_k_default
        self.candidate_pool_size: int = candidate_pool_size

        self.biobert_dir = BioBertPath
        self.biobert_model = None
        self.biobert_tokenizer = None

        # 全局实体索引
        self.entities: Dict[str, KGEntity] = {}
        # 每个实体对应多个 surface：(surface_text, embedding)
        self.entity_surfaces: Dict[str, List[Tuple[str, np.ndarray]]] = {}

        self._load_biobert()
        self._build_entity_index()

        # 为了配合你要求的“无中间 info 日志”，这里不打印构建完成信息

    # ---------------- BioBERT 相关 ----------------
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
            # 这是致命问题，打 error
            self.logger.error(
                f"[KeywordSearch][BioBERT] load failed ({e}), keyword search will not work properly."
            )
            self.biobert_model = None
            self.biobert_tokenizer = None

    @torch.no_grad()
    def _encode_text(self, text: str) -> Optional[np.ndarray]:
        if not self.biobert_model or not self.biobert_tokenizer:
            return None
        text = (text or "").strip()
        if not text:
            return None
        inputs = self.biobert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        outputs = self.biobert_model(**inputs)
        vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return vec

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0 or math.isnan(norm):
            return vec
        return vec / norm

    # ---------------- 建索引（name + aliases 多 surface） ----------------
    def _build_entity_index(self) -> None:
        """
        遍历内存中的关系，把 subject 视为 KGEntity，
        为每个实体记录：
          - self.entities[eid] = KGEntity
          - self.entity_surfaces[eid] = [(surface_text, emb_norm), ...]
            其中 surface_text = name 或 alias
        """
        triples = self.memory.relations.all()
        seen_ids = set()
        for tri in triples:
            node = getattr(tri, "subject", None)
            if node is None:
                continue
            if not isinstance(node, KGEntity):
                ent = KGEntity(**node)
            else:
                ent = node
            eid = getattr(ent, "entity_id", None)
            if not eid or eid in seen_ids:
                continue

            # 收集 surfaces: name + aliases
            surfaces: List[str] = []
            name = getattr(ent, "name", None)
            if isinstance(name, str) and name.strip():
                surfaces.append(name.strip())
            aliases = getattr(ent, "aliases", None) or []
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    surfaces.append(a.strip())

            if not surfaces:
                continue  # 没有任何有效文本，不建索引

            surface_vecs: List[Tuple[str, np.ndarray]] = []
            for s in surfaces:
                emb = self._encode_text(s)
                if emb is None:
                    continue
                emb_norm = self._l2_normalize(emb)
                surface_vecs.append((s, emb_norm))

            if not surface_vecs:
                continue  # 所有 surface 都没成功编码

            seen_ids.add(eid)
            self.entities[eid] = ent
            self.entity_surfaces[eid] = surface_vecs

        # 不在这里打印 info，避免过程日志刷屏

    # ---------------- 针对单个 keyword 的 BioBERT Top-K 检索 ----------------
    def _search_top_k_for_keyword(
        self,
        keyword: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[KGEntity, float, str, str]]:
        """
        返回：
          [(KGEntity, best_sim, best_surface_text, surface_source), ...]
        其中 best_sim 是 keyword 与该实体 [name + aliases] 多个 surface 中最大相似度；
        surface_source 取值：'name' / 'alias' / 'unknown'
        """
        if top_k is None:
            top_k = self.candidate_pool_size

        if not self.entity_surfaces:
            self.logger.warning("[KeywordSearch] entity_surfaces is empty, return [].")
            return []

        q_vec = self._encode_text(keyword)
        if q_vec is None:
            self.logger.warning(f"[KeywordSearch] failed to encode keyword='{keyword}'")
            return []

        q_vec = self._l2_normalize(q_vec)

        scored: List[Tuple[str, float, str, str]] = []
        for eid, surfaces in self.entity_surfaces.items():
            best_sim = -1.0
            best_surface_text = ""
            for surface_text, svec in surfaces:
                if svec is None:
                    continue
                sim = float(np.dot(q_vec, svec))
                if sim > best_sim:
                    best_sim = sim
                    best_surface_text = surface_text

            if best_sim < 0.0:
                continue

            ent = self.entities.get(eid)
            if ent is None:
                continue

            # 判断这个 surface 来自 name 还是 alias
            source = "unknown"
            if isinstance(ent.name, str) and best_surface_text == ent.name.strip():
                source = "name"
            elif best_surface_text in (ent.aliases or []):
                source = "alias"

            scored.append((eid, best_sim, best_surface_text, source))

        # 按相似度排序，取前 top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top_scored = scored[:top_k]

        results: List[Tuple[KGEntity, float, str, str]] = []
        for eid, sim, best_surface_text, source in top_scored:
            ent = self.entities.get(eid)
            if ent is None:
                continue
            results.append((ent, sim, best_surface_text, source))

        return results

    # ---------------- LLM 在候选中选若干个实体 id ----------------
    def _llm_disambiguate(
        self,
        keyword: str,
        candidates: List[Dict[str, Any]],
        max_return: Optional[int] = None,
    ) -> Optional[List[str]]:
        """
        让 LLM 在候选中选出若干个实体 id（最多 max_return 个）。
        返回 entity_id 列表；失败则返回 None。
        """
        if not candidates:
            return None

        if max_return is None:
            max_return = self.top_k_default

        payload = {
            "keyword": keyword,
            "top_k": max_return,
            "candidates": candidates,
        }
        prompt = json.dumps(payload, ensure_ascii=False)

        raw = self.call_llm(prompt)

        try:
            obj = json.loads(raw)
        except Exception as e:
            self.logger.warning(
                f"[KeywordSearch][LLM] parse JSON failed: raw={raw!r}, error={e}"
            )
            return None

        ids = obj.get("entity_ids")
        if not isinstance(ids, list):
            return None

        cleaned: List[str] = []
        for x in ids:
            if isinstance(x, str) and x.strip():
                cleaned.append(x.strip())

        if not cleaned:
            return None

        # 去重并截断到 max_return
        uniq: List[str] = []
        seen = set()
        for eid in cleaned:
            if eid not in seen:
                uniq.append(eid)
                seen.add(eid)
            if len(uniq) >= max_return:
                break

        return uniq or None

    # ---------------- 对外接口：多 keyword，每个 keyword 返回 K 个实体 ----------------
    def process(
        self,
    ) -> Tuple[
        Dict[str, List[KGEntity]],
        Dict[str, List[float]],
        Dict[str, List[Tuple[KGEntity, float, str, str]]],
    ]:
        """
        返回：
          kw2best_entities: {keyword: [KGEntity, ...]}
          kw2best_scores:   {keyword: [float, ...]}        # 与 best_entities 对应
          kw2candidates:    {keyword: [(KGEntity, score, best_surface, source), ...]}
        并在最后输出一张总表格日志（info），过程不刷 info，只在异常时打 warning/error。
        """
        kw2best_entities: Dict[str, List[KGEntity]] = {}
        kw2best_scores: Dict[str, List[float]] = {}
        kw2candidates: Dict[str, List[Tuple[KGEntity, float, str, str]]] = {}

        # 用于最后汇总日志的一行一行记录
        # 列：Keyword | Rank | EntityID | EntityName | MatchedSurface | SurfaceType | Similarity
        table_rows: List[List[Any]] = []

        for kw in self.keywords:
            kw = (kw or "").strip()
            if not kw:
                continue

            # 1) BioBERT 相似度检索；候选池大小 = candidate_pool_size
            candidates = self._search_top_k_for_keyword(
                kw,
                top_k=self.candidate_pool_size,
            )
            kw2candidates[kw] = candidates

            if not candidates:
                # 这个 keyword 没有任何候选，跳过但记录空
                kw2best_entities[kw] = []
                kw2best_scores[kw] = []
                continue

            # 构造给 LLM 的候选信息
            llm_cands: List[Dict[str, Any]] = []
            # 用来后面查找 best_surface 的索引：eid -> (sim, best_surface, source)
            eid2surface_info: Dict[str, Tuple[float, str, str]] = {}

            for ent, sim, best_surface, source in candidates:
                eid = ent.entity_id
                eid2surface_info[eid] = (sim, best_surface, source)
                llm_cands.append(
                    {
                        "entity_id": eid,
                        "name": getattr(ent, "name", "") or "",
                        "description": getattr(ent, "description", "") or "",
                        "best_surface": best_surface,
                        "surface_type": source,  # name / alias / unknown
                        "similarity_hint": f"{sim:.4f}",
                    }
                )

            # 2) 让 LLM 在候选里选 若干个（最多 top_k_default 个）
            chosen_ids = self._llm_disambiguate(
                kw,
                llm_cands,
                max_return=self.top_k_default,
            )

            best_entities: List[KGEntity] = []
            best_scores: List[float] = []

            if chosen_ids is None:
                # LLM 挂了，就退回 BioBERT Top-K（这里用 top_k_default，而不是 candidate_pool_size）
                fallback = candidates[: self.top_k_default]
                for ent, sim, _, _ in fallback:
                    best_entities.append(ent)
                    best_scores.append(sim)
            else:
                # 在候选中按 LLM 顺序对齐
                id_to_ent: Dict[str, KGEntity] = {ent.entity_id: ent for ent, _, _, _ in candidates}

                for eid in chosen_ids:
                    ent = id_to_ent.get(eid)
                    if ent is None:
                        continue
                    sim, _, _ = eid2surface_info.get(eid, (0.0, "", "unknown"))
                    best_entities.append(ent)
                    best_scores.append(sim)

                # 如果 LLM 给出的 id 都不在候选里，则退回 top-1
                if not best_entities:
                    ent, sim, _, _ = candidates[0]
                    best_entities = [ent]
                    best_scores = [sim]

            kw2best_entities[kw] = best_entities
            kw2best_scores[kw] = best_scores

            # 写入 memory：一个 keyword 对应 K 个 best_entities
            self.memory.add_keyword_entities(kw, best_entities)

            # 同时平铺到全局 key_entities 里
            self.memory.add_key_entities(best_entities)

            # 收集表格行（每个 keyword × 每个选中的实体都一行）
            for rank, ent in enumerate(best_entities, start=1):
                eid = ent.entity_id
                name = getattr(ent, "name", "") or ""
                sim, best_surface, source = eid2surface_info.get(eid, (0.0, "", "unknown"))
                table_rows.append(
                    [
                        kw,
                        rank,
                        eid,
                        name,
                        best_surface,
                        source,          # name / alias / unknown
                        f"{sim:.4f}",
                    ]
                )

        # -------- 最终汇总日志：表格形式（只打一条 info） --------
        if table_rows:
            headers = [
                "Keyword",
                "Rank",
                "EntityID",
                "EntityName",
                "MatchedSurface",
                "SurfaceType",
                "Similarity",
            ]

            # 计算每一列的宽度
            col_widths = [len(h) for h in headers]
            for row in table_rows:
                for i, cell in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

            def _fmt_row(row_vals: List[Any]) -> str:
                parts = []
                for i, cell in enumerate(row_vals):
                    s = str(cell)
                    parts.append(s.ljust(col_widths[i]))
                return " | ".join(parts)

            header_line = _fmt_row(headers)
            sep_line = "-+-".join("-" * w for w in col_widths)
            body_lines = [_fmt_row(r) for r in table_rows]

            table_str = "\n".join([header_line, sep_line] + body_lines)

            self.logger.info(
                "\n[KeywordSearch] Summary of keyword → entities mapping:\n"
                + table_str
            )

        return kw2best_entities, kw2best_scores, kw2candidates