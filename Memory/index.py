# shared_memory_min.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid, json
from HyAgent.TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity  # 保留你给的导入路径
from tqdm import tqdm
from pathlib import Path
from typing import Union
# ---------- 数据对象 ----------
@dataclass
class KGRelation:
    rel_id: str = ""                    # 留空自动生成
    head_id: str = ""                   # 实体ID
    rel_type: str = ""                  # 关系类型
    tail_id: str = ""                   # 实体ID
    props: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------- 独立存储：实体 ----------
class EntityStore:
    def __init__(self):
        self.by_id: Dict[str, KGEntity] = {}
        self.idx_normid: Dict[str, str] = {}  # normalized_id.lower() -> entity_id
        self.idx_name: Dict[str, str] = {}    # norm(name) -> entity_id

    def _nid(self) -> str: return f"ent:{uuid.uuid4().hex[:12]}"
    def _rid(self) -> str: return f"rel:{uuid.uuid4().hex[:12]}"
    def _key(self, s: str) -> str: return (s or "").strip().lower()

    def upsert(self, e: KGEntity) -> KGEntity:
        """
        规则：
        - 完全忽略来稿里的 entity_id（防止 name==id 的历史设计影响），统一由本存储分配。
        - 先按 normalized_id 合并；否则按规范化 name 合并；否则新建并分配 ent:xxxxxx。
        """
        # 忽略外部传入的 entity_id
        e.entity_id = ""

        norm = self._key(e.name)

        # 1) 先用 normalized_id 合并
        if e.normalized_id and e.normalized_id != "N/A":
            k = self._key(e.normalized_id)
            if k in self.idx_normid:
                return self._merge(self.by_id[self.idx_normid[k]], e)

        # 2) 再用规范化 name 合并
        if norm and norm in self.idx_name:
            return self._merge(self.by_id[self.idx_name[norm]], e)

        # 3) 新建：分配全新 ID，并建立索引
        new_id = self._nid()
        e.entity_id = new_id
        self.by_id[new_id] = e
        if e.normalized_id and e.normalized_id != "N/A":
            self.idx_normid[self._key(e.normalized_id)] = new_id
        if norm:
            self.idx_name[norm] = new_id
        return e

    def _merge(self, base: KGEntity, inc: KGEntity) -> KGEntity:
        # 类型：用更具体的
        if base.entity_type == "Unknown" and inc.entity_type != "Unknown":
            base.entity_type = inc.entity_type

        # 名称：更长更可读则升级主名，并把旧主名并入别名
        if inc.name and len(inc.name) > len(base.name):
            if base.name:
                base.aliases.append(base.name)
            base.name = inc.name

        # 本体ID：优先非 N/A
        if base.normalized_id == "N/A" and inc.normalized_id and inc.normalized_id != "N/A":
            base.normalized_id = inc.normalized_id
            self.idx_normid[self._key(base.normalized_id)] = base.entity_id

        # 别名并集（去重，大小写不敏感）
        pool = {self._key(a): a for a in base.aliases}
        for a in ([inc.name] if inc.name else []) + (inc.aliases or []):
            if a:
                pool.setdefault(self._key(a), a)
        base.aliases = sorted(pool.values(), key=str.lower)

        # 重新索引规范名（主名可能变化）
        norm = self._key(base.name)
        if norm:
            self.idx_name[norm] = base.entity_id

        return base

    def find_by_norm(self, name_or_alias: str) -> Optional[KGEntity]:
        k = self._key(name_or_alias)
        eid = self.idx_name.get(k)
        return self.by_id.get(eid) if eid else None
    # 在 EntityStore 类中加入
    def upsert_many(self, entities: List[KGEntity]) -> List[KGEntity]:
        return [self.upsert(e) for e in tqdm(entities)]
    def find_by_normalized_id(self, normalized_id: str) -> Optional[KGEntity]:
        k = self._key(normalized_id)
        eid = self.idx_normid.get(k)
        return self.by_id.get(eid) if eid else None

    def all(self) -> List[KGEntity]:
        return list(self.by_id.values())

# ---------- 独立存储：关系 ----------
class RelationStore:
    def __init__(self):
        self.by_id: Dict[str, KGRelation] = {}

    def _rid(self) -> str: return f"rel:{uuid.uuid4().hex[:12]}"

    def add(self, r: KGRelation) -> KGRelation:
        if not r.rel_id: r.rel_id = self._rid()
        self.by_id[r.rel_id] = r
        return r

    def all(self) -> List[KGRelation]:
        return list(self.by_id.values())

# ---------- 全局共享记忆池（解耦聚合） ----------
class Memory:
    def __init__(self):
        self.entities = EntityStore()
        self.relations = RelationStore()

    def upsert_many_entities(self, entities: List[KGEntity]) -> List[KGEntity]:
        """转发到 EntityStore，并兼容 dict 输入"""
        self.entities.upsert_many(entities)
    def dump_json(self, dirpath: str = ".") -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dirp = Path(dirpath)
        dirp.mkdir(parents=True, exist_ok=True)  # 自动建目录

        path = dirp / f"memory-{ts}.json"
        data = {
            "entities": [e.to_dict() for e in self.entities.all()],
            "relations": [r.to_dict() for r in self.relations.all()],
            "meta": {"generated_at": ts}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(path)

# 模块级全局实例：各 Agent 直接 `from shared_memory_min import memory`
memory = Memory()