import math
import json
from typing import List, Dict, Tuple, Optional

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
      1）用 BioBERT 在 Memory 全局实体中检索与 keyword 最相近的实体（相似度 Top-N 候选池）
      2）把这 N 个候选丢给大模型，由大模型在候选中选出 若干个 最匹配的实体（最多 K 个）

    参数：
      - top_k_default: 最终想返回的实体数量 K
      - candidate_pool_size: 提供给 LLM 选择的候选池大小 N（如果为 None，则默认为 max(K, 10)）

    用法示例：
        agent = KeywordEntitySearchAgent(
            client=openai_client,
            model_name="gpt-4o-mini",
            keyword="Src family kinase",
            memory=get_memory(),
            top_k_default=3,          # 最终返回 3 个
            candidate_pool_size=15,   # 让 LLM 从 15 个候选里选
        )
        best_ents, best_scores, candidates = agent.process()
    """

    def __init__(
        self,
        client,
        model_name: str,
        keyword: str,
        memory: Optional[Memory] = None,
        top_k_default: int = 3,             # 最终返回 K 个
        candidate_pool_size: Optional[int] = None,  # BioBERT 候选池 N
    ):
        # 先保存 K，用于 system_prompt 里描述「最多返回 K 个」
        self.top_k_default = top_k_default

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
            f"- NEVER return more than K={top_k_default} ids.\n"
            "- Do not include any extra fields, comments, or text outside the JSON."
        )
        super().__init__(client, model_name, system_prompt)

        self.logger = get_global_logger()
        self.memory: Memory = memory or get_memory()
        self.keyword = keyword
        self.biobert_dir = BioBertPath
        self.biobert_model = None
        self.biobert_tokenizer = None
        self.entities: Dict[str, KGEntity] = {}
        self.entity_embeddings: Dict[str, np.ndarray] = {}

        # ------- 新增：候选池大小 N -------
        if candidate_pool_size is None:
            # 默认给 LLM 的候选池比 K 大一点（至少 10）
            candidate_pool_size = max(top_k_default, 10)
        # 保证候选池大小 >= K
        if candidate_pool_size < top_k_default:
            candidate_pool_size = top_k_default
        self.candidate_pool_size = int(candidate_pool_size)
        # ---------------------------------

        # 索引统计信息，用于最终 summary 日志
        self._index_total_entities: int = 0
        self._index_with_embeddings: int = 0

        self._load_biobert()
        self._build_entity_index()

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
            # 成功加载不打 log，保持过程静默
        except Exception as e:
            # 真正的错误仍然报出来
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
        try:
            inputs = self.biobert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            outputs = self.biobert_model(**inputs)
            vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            return vec
        except Exception as e:
            # 编码失败算异常，打 error
            self.logger.error(
                f"[KeywordSearch][BioBERT] encode_text failed for text='{text[:50]}...' ({e})"
            )
            return None

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0.0 or math.isnan(norm):
            return vec
        return vec / norm

    # ---------------- 建索引 ----------------
    def _build_entity_index(self) -> None:
        triples = self.memory.relations.all()
        seen_ids = set()
        count_ok = 0
        for tri in triples:
            node = getattr(tri, "subject", None)
            if node is None:
                continue
            ent = KGEntity(**node)
            eid = getattr(ent, "entity_id", None)
            if not eid or eid in seen_ids:
                continue
            seen_ids.add(eid)
            # 记录到本地实体索引
            self.entities[eid] = ent
            # 选一个文本用于 BioBERT 编码：name > description > normalized_id
            text = getattr(ent, "name", None)
            if text is None:
                continue
            emb = self._encode_text(text)
            if emb is None:
                continue
            emb_norm = self._l2_normalize(emb)
            self.entity_embeddings[eid] = emb_norm
            count_ok += 1

        # 只保存统计信息，不打印过程 log
        self._index_total_entities = len(self.entities)
        self._index_with_embeddings = count_ok

    # ---------------- BioBERT 检索 Top-K（候选池） ----------------
    def _search_top_k(
        self, keyword: str, top_k: Optional[int] = None
    ) -> List[Tuple[KGEntity, float]]:
        """
        用 BioBERT 检索 Top-K 候选（这里 K 一般是 candidate_pool_size）
        """
        if top_k is None:
            top_k = self.candidate_pool_size
        if not self.entity_embeddings:
            # 没有任何 embedding，算异常
            self.logger.error(
                "[KeywordSearch] entity_embeddings is empty, cannot perform keyword search."
            )
            return []
        q_vec = self._encode_text(keyword)
        if q_vec is None:
            self.logger.error(
                f"[KeywordSearch] failed to encode keyword='{keyword}', cannot perform search."
            )
            return []
        q_vec = self._l2_normalize(q_vec)
        scores: List[Tuple[str, float]] = []
        for eid, evec in self.entity_embeddings.items():
            if evec is None:
                continue
            sim = float(np.dot(q_vec, evec))
            scores.append((eid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_k]
        results: List[Tuple[KGEntity, float]] = []
        for eid, sim in top_scores:
            ent = self.entities.get(eid)
            if ent is not None:
                results.append((ent, sim))
        return results

    # ---------------- LLM 在候选中选 若干个 ----------------
    def _llm_disambiguate(
        self,
        keyword: str,
        candidates: List[Dict[str, str]],
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

        # 用基类里的 LLM 调用
        raw = self.call_llm(prompt)

        try:
            obj = json.loads(raw)
        except Exception as e:
            # 解析失败算异常，打 error
            self.logger.error(
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

    # ---------------- Summary 日志格式化 ----------------
    def _log_summary_table(
        self,
        keyword: str,
        candidates: List[Tuple[KGEntity, float]],
        best_entities: List[KGEntity],
        best_scores: List[float],
        decision_source: str,
    ) -> None:
        """
        在所有计算结束后，以表格形式打印一次优雅的 summary 日志。
        """
        lines: List[str] = []
        sep_main = "=" * 95
        sep_sub = "-" * 95

        lines.append(sep_main)
        lines.append("[KeywordSearch] Summary")
        lines.append(sep_sub)
        lines.append(f"Keyword               : {keyword}")
        lines.append(f"Final Top-K (return)  : {self.top_k_default}")
        lines.append(f"Candidate Pool Size N : {self.candidate_pool_size}")
        lines.append(
            f"Index Entities         : {self._index_total_entities} "
            f"(with embeddings: {self._index_with_embeddings})"
        )
        lines.append(f"BioBERT Candidates    : {len(candidates)}")
        lines.append(f"Selected Entities      : {len(best_entities)}")
        lines.append(f"Decision Source        : {decision_source}")
        lines.append(sep_sub)

        if not best_entities:
            lines.append("No entities selected.")
            lines.append(sep_main)
            self.logger.info("\n" + "\n".join(lines))
            return

        # 表头
        col_rank = "Rank"
        col_id = "Entity ID"
        col_name = "Name"
        col_sim = "Sim"

        # 设定宽度
        w_rank = 6
        w_id = 20
        w_name = 40
        w_sim = 8

        header = (
            f"{col_rank:^{w_rank}} | {col_id:^{w_id}} | "
            f"{col_name:^{w_name}} | {col_sim:^{w_sim}}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        def _truncate(s: str, max_len: int) -> str:
            s = s or ""
            if len(s) <= max_len:
                return s
            if max_len <= 3:
                return s[:max_len]
            return s[: max_len - 3] + "..."

        for idx, (ent, score) in enumerate(zip(best_entities, best_scores), start=1):
            eid = getattr(ent, "entity_id", "") or ""
            name = getattr(ent, "name", "") or ""
            eid_short = _truncate(eid, w_id)
            name_short = _truncate(name, w_name)
            sim_str = f"{score:.4f}"

            line = (
                f"{idx:^{w_rank}} | {eid_short:<{w_id}} | "
                f"{name_short:<{w_name}} | {sim_str:^{w_sim}}"
            )
            lines.append(line)

        lines.append(sep_main)

        # 最终只打一条 info，把整个表格输出
        self.logger.info("\n" + "\n".join(lines))

    # ---------------- 对外接口：process() 返回 K 个实体 ----------------
    def process(
        self,
    ) -> Tuple[List[KGEntity], List[float], List[Tuple[KGEntity, float]]]:
        """
        不传任何参数：
          - 使用 __init__ 里的 self.keyword
          - 先用 BioBERT 找 N 个候选（candidate_pool_size）
          - 再用 LLM 从候选中选 若干个（最多 K 个）

        返回：
          best_entities: [KGEntity, ...] 选中的实体列表（可能 < K，失败则为空列表）
          best_scores:   [float, ...]    对应实体的 BioBERT 相似度（与 best_entities 对应）
          candidates:    [(KGEntity, score), ...] 原始 BioBERT 候选，方便调试
        """
        kw = self.keyword

        # 1) BioBERT 相似度检索（返回候选池 N）
        candidates = self._search_top_k(kw, top_k=self.candidate_pool_size)
        if not candidates:
            best_entities: List[KGEntity] = []
            best_scores: List[float] = []
            # 输出 summary 日志
            self._log_summary_table(
                keyword=kw,
                candidates=[],
                best_entities=best_entities,
                best_scores=best_scores,
                decision_source="no_candidates",
            )
            return best_entities, best_scores, []

        # 整理成给 LLM 用的候选信息
        llm_cands: List[Dict[str, str]] = []
        for ent, sim in candidates:
            llm_cands.append(
                {
                    "entity_id": ent.entity_id,
                    "name": getattr(ent, "name", "") or "",
                    "description": getattr(ent, "description", "") or "",
                    "similarity_hint": f"{sim:.4f}",
                }
            )

        # 2) 让 LLM 在候选里选 若干个（最多 K 个）
        chosen_ids = self._llm_disambiguate(
            kw, llm_cands, max_return=self.top_k_default
        )

        best_entities: List[KGEntity] = []
        best_scores: List[float] = []
        decision_source: str

        if chosen_ids is None:
            # LLM 挂了就退回 BioBERT Top-K（在候选池里再取前 K）
            decision_source = "fallback_biobert_topk"
            for ent, sim in candidates[: self.top_k_default]:
                best_entities.append(ent)
                best_scores.append(sim)
        else:
            # 在候选中按 LLM 顺序对齐
            id_to_item = {ent.entity_id: (ent, sim) for ent, sim in candidates}
            for eid in chosen_ids:
                item = id_to_item.get(eid)
                if item is None:
                    continue
                ent, sim = item
                best_entities.append(ent)
                best_scores.append(sim)

            # 如果 LLM 返回的 id 都不在候选里，则退回 BioBERT Top-1
            if not best_entities:
                ent, sim = candidates[0]
                best_entities = [ent]
                best_scores = [sim]
                decision_source = "fallback_biobert_top1"
            else:
                decision_source = "llm_filtered"

        # 把选中的实体都记录到 memory 的 key entity 里
        for ent in best_entities:
            self.memory.add_key_entity(ent)

        # 最终 summary 表格 log
        self._log_summary_table(
            keyword=kw,
            candidates=candidates,
            best_entities=best_entities,
            best_scores=best_scores,
            decision_source=decision_source,
        )

        return best_entities, best_scores, candidates