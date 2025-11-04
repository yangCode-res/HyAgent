# shared_memory.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Iterable, Callable, Any, Tuple
import threading
import uuid
import json

# ---- 你的 KGEntity 定义（可直接复用你已有的）----
@dataclass
class KGEntity:
    """
    Represents a canonical entity in the knowledge graph.
    """
    entity_id: str
    entity_type: str = "Unknown"
    name: str = ""
    normalized_id: str = "N/A"
    aliases: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.name} ({self.entity_type})"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KGEntity":
        return cls(**data)

# ---------- 全局共享记忆池（线程安全 / 可扩展） ----------
class GlobalMemory:
    """
    一个全局共享的内存池：
      - 线程安全（RLock）
      - 面向实体的多索引（id / normalized_id / name/alias）
      - 自动去重合并（基于 normalized_id 或名称/别名重叠）
      - 可持久化到 JSON
      - 可扩展：为关系等其他对象预留命名空间
    """
    _instance = None
    _lock_singleton = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock_singleton:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 实体存储
        self._entities_by_id: Dict[str, KGEntity] = {}
        # 辅助索引
        self._idx_norm: Dict[str, str] = {}              # normalized_id -> entity_id
        self._idx_name: Dict[str, str] = {}              # lower(name) -> entity_id
        self._idx_alias: Dict[str, str] = {}             # lower(alias) -> entity_id
        # 将来扩展用：比如关系、三元组等
        self._namespaces: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ---------- 工具 ----------
    @staticmethod
    def _norm_key(s: str) -> str:
        return (s or "").strip().lower()

    @staticmethod
    def _new_id() -> str:
        return f"ent:{uuid.uuid4().hex[:16]}"

    # ---------- 对外 API：实体 CRUD ----------
    def upsert_entity(self, e: KGEntity) -> KGEntity:
        """
        插入或合并一个实体：
          - 若 normalized_id 存在并已被索引，则合并到该实体
          - 否则若名称/别名命中已存在实体（忽略大小写），则合并
          - 否则新建
        返回“池中最终的实体”。
        """
        with self._lock:
            # 1) 命中 normalized_id 优先
            if e.normalized_id and e.normalized_id != "N/A":
                existed_id = self._idx_norm.get(self._norm_key(e.normalized_id))
                if existed_id and existed_id in self._entities_by_id:
                    return self._merge_into(self._entities_by_id[existed_id], e)

            # 2) 名称/别名命中
            keys = []
            if e.name:
                keys.append(self._idx_name.get(self._norm_key(e.name)))
            for a in e.aliases:
                k = self._idx_alias.get(self._norm_key(a))
                if k:
                    keys.append(k)
            keys = [k for k in keys if k]

            if keys:
                # 命中第一条即可（也可设计更复杂的多重合并）
                return self._merge_into(self._entities_by_id[keys[0]], e)

            # 3) 新建
            if not e.entity_id:
                e.entity_id = self._new_id()
            self._entities_by_id[e.entity_id] = e
            self._index_entity(e)
            return e

    def get_entity(self, entity_id: str) -> Optional[KGEntity]:
        with self._lock:
            return self._entities_by_id.get(entity_id)

    def find_by_normalized(self, normalized_id: str) -> Optional[KGEntity]:
        with self._lock:
            eid = self._idx_norm.get(self._norm_key(normalized_id))
            return self._entities_by_id.get(eid) if eid else None

    def find_by_name(self, name_or_alias: str) -> Optional[KGEntity]:
        with self._lock:
            key = self._norm_key(name_or_alias)
            eid = self._idx_name.get(key) or self._idx_alias.get(key)
            return self._entities_by_id.get(eid) if eid else None

    def list_entities(
        self,
        predicate: Optional[Callable[[KGEntity], bool]] = None
    ) -> List[KGEntity]:
        with self._lock:
            it = self._entities_by_id.values()
            return [e for e in it if (predicate(e) if predicate else True)]

    def remove_entity(self, entity_id: str) -> bool:
        with self._lock:
            e = self._entities_by_id.pop(entity_id, None)
            if not e:
                return False
            # 清理索引
            if e.normalized_id and e.normalized_id != "N/A":
                self._idx_norm.pop(self._norm_key(e.normalized_id), None)
            self._idx_name.pop(self._norm_key(e.name), None)
            for a in e.aliases:
                self._idx_alias.pop(self._norm_key(a), None)
            return True

    def clear(self):
        with self._lock:
            self._entities_by_id.clear()
            self._idx_norm.clear()
            self._idx_name.clear()
            self._idx_alias.clear()
            self._namespaces.clear()

    # ---------- 合并 & 索引 ----------
    def _merge_into(self, base: KGEntity, incoming: KGEntity) -> KGEntity:
        """
        将 incoming 的信息合并入 base，并更新索引。
        合并策略（简单实用）：
          - entity_type：优先保留更具体（非 Unknown）的
          - name：保留更长的可读名称（或已有）
          - normalized_id：优先非 N/A
          - aliases：并集去重；把 base.name 与 incoming.name 互相纳入别名（避免丢失）
        """
        changed = False

        # type
        if base.entity_type == "Unknown" and incoming.entity_type != "Unknown":
            base.entity_type = incoming.entity_type
            changed = True

        # name：保留更长的可读名称
        cand_name = incoming.name or ""
        if cand_name and len(cand_name) > len(base.name or ""):
            # 原 name 进 aliases
            if base.name and self._norm_key(base.name) != self._norm_key(cand_name):
                base.aliases.append(base.name)
            base.name = cand_name
            changed = True

        # normalized_id
        if (not base.normalized_id or base.normalized_id == "N/A") and \
           (incoming.normalized_id and incoming.normalized_id != "N/A"):
            base.normalized_id = incoming.normalized_id
            changed = True

        # aliases 合并
        pool = {self._norm_key(a): a for a in base.aliases}
        # 把双方名称作为别名互补
        if incoming.name and self._norm_key(incoming.name) != self._norm_key(base.name):
            pool.setdefault(self._norm_key(incoming.name), incoming.name)
        if base.name and self._norm_key(base.name) != self._norm_key(incoming.name or ""):
            pool.setdefault(self._norm_key(base.name), base.name)

        for a in incoming.aliases:
            pool.setdefault(self._norm_key(a), a)
        base.aliases = sorted(set(pool.values()), key=lambda s: s.lower())

        if changed:
            # 重新索引（先清除旧索引，再建新索引）
            self._reindex_entity(base)

        return base

    def _index_entity(self, e: KGEntity) -> None:
        if e.normalized_id and e.normalized_id != "N/A":
            self._idx_norm[self._norm_key(e.normalized_id)] = e.entity_id
        if e.name:
            self._idx_name[self._norm_key(e.name)] = e.entity_id
        for a in e.aliases:
            self._idx_alias[self._norm_key(a)] = e.entity_id

    def _reindex_entity(self, e: KGEntity) -> None:
        # 为简单起见：先从三个索引中清掉“指向该实体”的旧键，再重建
        # （也可维护反向索引优化，这里保持简洁）
        keys_norm = [k for k, v in self._idx_norm.items() if v == e.entity_id]
        keys_name = [k for k, v in self._idx_name.items() if v == e.entity_id]
        keys_alias = [k for k, v in self._idx_alias.items() if v == e.entity_id]
        for k in keys_norm:  self._idx_norm.pop(k, None)
        for k in keys_name:  self._idx_name.pop(k, None)
        for k in keys_alias: self._idx_alias.pop(k, None)
        self._index_entity(e)

    # ---------- 持久化 ----------
    def save_json(self, path: str) -> None:
        with self._lock, open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entities": [e.to_dict() for e in self._entities_by_id.values()],
                    "namespaces": self._namespaces,  # 保留扩展数据
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def load_json(self, path: str, merge: bool = True) -> None:
        with self._lock, open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ents = data.get("entities", [])
        ns = data.get("namespaces", {})
        if not merge:
            self.clear()
        for d in ents:
            self.upsert_entity(KGEntity.from_dict(d))
        # 合并或覆盖命名空间
        self._namespaces.update(ns)

    # ---------- 扩展命名空间（示例：关系等） ----------
    def put_in_namespace(self, ns: str, key: str, value: Any) -> None:
        with self._lock:
            space = self._namespaces.setdefault(ns, {})
            space[key] = value

    def get_from_namespace(self, ns: str, key: str) -> Any:
        with self._lock:
            return self._namespaces.get(ns, {}).get(key)

# 便捷的全局实例（推荐直接 import 使用）
memory = GlobalMemory()