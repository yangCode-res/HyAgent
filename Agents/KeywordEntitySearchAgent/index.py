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
      1）用 BioBERT 在 Memory 全局实体中检索与 keyword 最相近的实体（相似度 Top-K）
      2）把这 K 个候选丢给大模型，由大模型在候选中选出 1 个最匹配的实体

    用法示例：
        agent = KeywordEntitySearchAgent(
            client=openai_client,
            model_name="gpt-4o-mini",
            keyword="Src family kinase",
            memory=get_memory(),
            top_k_default=10,
        )
        best_ent, best_score, candidates = agent.process()
    """

    def __init__(
        self,
        client,
        model_name: str,
        keyword: str,
        memory: Optional[Memory] = None,
        top_k_default: int = 10,
    ):
        # ✅ 修改后的 system_prompt：专门做“实体链接 + 候选消歧”
        system_prompt = (
            "You are a biomedical entity-linking agent. "
            "Given a query keyword and a list of candidate entities from a knowledge graph, "
            "you must choose EXACTLY ONE candidate that best matches the keyword. "
            "Always respond with JSON: {\"entity_id\": \"<id>\"}."
        )
        super().__init__(client, model_name, system_prompt)

        self.logger = get_global_logger()
        self.memory: Memory = memory or get_memory()
        self.top_k_default = top_k_default
        self.keyword = keyword  # ✅ 关键词在 init 里传进来，以后都用 self.keyword

        self.biobert_dir = BioBertPath
        self.biobert_model = None
        self.biobert_tokenizer = None

        # entity_id -> KGEntity
        self.entities: Dict[str, KGEntity] = {}
        # entity_id -> np.ndarray (归一化后的 embedding)
        self.entity_embeddings: Dict[str, np.ndarray] = {}

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
            self.logger.info(f"[KeywordSearch][BioBERT] loaded from {self.biobert_dir}")
        except Exception as e:
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

    # ---------------- 建索引 ----------------
    def _build_entity_index(self) -> None:
        if not self.biobert_model:
            self.logger.warning("[KeywordSearch] BioBERT not available, skip index building.")
            return

        ents = list(self.memory.entities.all())
        self.logger.info(f"[KeywordSearch] building index for {len(ents)} entities...")

        count_ok = 0
        for ent in ents:
            self.entities[ent.entity_id] = ent

            text = None
            name = getattr(ent, "name", None)
            desc = getattr(ent, "description", None)
            norm_id = getattr(ent, "normalized_id", None)

            if name and str(name).strip():
                text = str(name)
            elif desc and str(desc).strip():
                text = str(desc)
            elif norm_id and str(norm_id).strip():
                text = str(norm_id)
            else:
                continue

            emb = self._encode_text(text)
            if emb is None:
                continue

            emb_norm = self._l2_normalize(emb)
            self.entity_embeddings[ent.entity_id] = emb_norm
            count_ok += 1

        self.logger.info(
            f"[KeywordSearch] index built: {len(self.entities)} entities, "
            f"{count_ok} got BioBERT embeddings."
        )

    # ---------------- BioBERT 检索 Top-K ----------------
    def _search_top_k(self, keyword: str, top_k: Optional[int] = None) -> List[Tuple[KGEntity, float]]:
        if top_k is None:
            top_k = self.top_k_default

        if not self.entity_embeddings:
            self.logger.warning("[KeywordSearch] entity_embeddings is empty, return [].")
            return []

        q_vec = self._encode_text(keyword)
        if q_vec is None:
            self.logger.warning(f"[KeywordSearch] failed to encode keyword='{keyword}'")
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

        if results:
            self.logger.info(
                f"[KeywordSearch] keyword='{keyword}' -> top1='{results[0][0].get_name()}' "
                f"(sim={results[0][1]:.4f})"
            )
        else:
            self.logger.info(f"[KeywordSearch] keyword='{keyword}' -> no results")

        return results

    # ---------------- LLM 在候选中选 1 个 ----------------
    def _llm_disambiguate(
        self,
        keyword: str,
        candidates: List[Dict[str, str]],
    ) -> Optional[str]:
        if not candidates:
            return None

        payload = {
            "keyword": keyword,
            "candidates": candidates,
        }
        prompt = json.dumps(payload, ensure_ascii=False)

        # 这里用你基类里的 LLM 调用
        raw = self.call_llm(prompt)

        try:
            obj = json.loads(raw)
            eid = obj.get("entity_id")
            if isinstance(eid, str) and eid.strip():
                return eid.strip()
        except Exception as e:
            self.logger.warning(
                f"[KeywordSearch][LLM] parse JSON failed: raw={raw!r}, error={e}"
            )
            return None

        return None

    # ---------------- 对外接口：process() 不再收参数 ----------------
    def process(self) -> Tuple[Optional[KGEntity], Optional[float], List[Tuple[KGEntity, float]]]:
        """
        不传任何参数：
          - 使用 __init__ 里的 self.keyword
          - 先用 BioBERT 找 Top-K
          - 再用 LLM 从候选中选 1 个
        返回：
          best_entity: 选中的 KGEntity（失败则为 None）
          best_score: 该实体的 BioBERT 相似度（失败则为 None）
          candidates: [(KGEntity, score), ...] 原始候选，方便调试
        """
        kw = self.keyword

        # 1) BioBERT 相似度检索
        candidates = self._search_top_k(kw, top_k=self.top_k_default)
        if not candidates:
            return None, None, []

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

        # 2) 让 LLM 在候选里选 1 个
        chosen_eid = self._llm_disambiguate(kw, llm_cands)

        if chosen_eid is None:
            # LLM 挂了就退回 Top-1
            best_ent, best_score = candidates[0]
            self.logger.info(
                f"[KeywordSearch] LLM disambiguation failed, fallback to top-1: "
                f"{best_ent.get_name()} (sim={best_score:.4f})"
            )
            return best_ent, best_score, candidates

        # 在候选中找这个 entity_id
        best_ent = None
        best_score = None
        for ent, sim in candidates:
            if ent.entity_id == chosen_eid:
                best_ent = ent
                best_score = sim
                break

        # 如果 LLM 给的 id 不在候选里，同样退回 Top-1
        if best_ent is None:
            best_ent, best_score = candidates[0]
            self.logger.info(
                f"[KeywordSearch] LLM chose unknown id={chosen_eid}, fallback to top-1: "
                f"{best_ent.get_name()} (sim={best_score:.4f})"
            )
        else:
            self.logger.info(
                f"[KeywordSearch] keyword='{kw}', LLM chose: "
                f"{best_ent.get_name()} (id={best_ent.entity_id}, sim={best_score:.4f})"
            )

        return best_ent, best_score, candidates