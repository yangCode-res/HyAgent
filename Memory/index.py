# shared_memory_min.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid, json
from pathlib import Path
from tqdm import tqdm

from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity  # 保留你给的导入路径


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

    def _nid(self) -> str:
        return f"ent:{uuid.uuid4().hex[:12]}"

    def _key(self, s: str) -> str:
        return (s or "").strip().lower()

    def upsert(self, e: KGEntity) -> KGEntity:
        """
        规则：
        - 忽略外部传入的 entity_id，统一由本存储分配。
        - 先按 normalized_id 合并；否则按规范化 name 合并；否则新建 ent:xxxxxx。
        """
        e.entity_id = ""  # 清掉外部ID
        norm = self._key(e.name)

        # 1) 用 normalized_id 合并
        if e.normalized_id and e.normalized_id != "N/A":
            k = self._key(e.normalized_id)
            if k in self.idx_normid:
                return self._merge(self.by_id[self.idx_normid[k]], e)

        # 2) 用规范化 name 合并
        if norm and norm in self.idx_name:
            return self._merge(self.by_id[self.idx_name[norm]], e)

        # 3) 新建：分配 ID + 建索引
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

        # 名称：更长更可读则升级主名，并把旧主名变为别名
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

    def find_by_normalized_id(self, normalized_id: str) -> Optional[KGEntity]:
        k = self._key(normalized_id)
        eid = self.idx_normid.get(k)
        return self.by_id.get(eid) if eid else None

    def upsert_many(self, entities: List[KGEntity]) -> List[KGEntity]:
        return [self.upsert(e) for e in tqdm(entities)]

    def all(self) -> List[KGEntity]:
        return list(self.by_id.values())


# ---------- 独立存储：关系 ----------
class RelationStore:
    def __init__(self):
        self.by_id: Dict[str, KGRelation] = {}

    def _rid(self) -> str:
        return f"rel:{uuid.uuid4().hex[:12]}"

    def add(self, r: KGRelation) -> KGRelation:
        if not r.rel_id:
            r.rel_id = self._rid()
        self.by_id[r.rel_id] = r
        return r

    def all(self) -> List[KGRelation]:
        return list(self.by_id.values())


# ---------- 子图 ----------
class Subgraph:
    def __init__(
        self,
        subgraph_id: str,                 # 你来传这个ID（字符串）
        name: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.id = subgraph_id            # 子图唯一标识（外部提供）
        self.name = name
        self.meta = dict(meta or {})
        self.entities = EntityStore()
        self.relations = RelationStore()

    # 子图内操作
    def upsert_entity(self, e: KGEntity) -> KGEntity:
        return self.entities.upsert(e)

    def upsert_many_entities(self, ents: List[KGEntity]) -> List[KGEntity]:
        return self.entities.upsert_many(ents)

    def add_relation(self, r: KGRelation) -> KGRelation:
        return self.relations.add(r)

    def add_relations(self, rs: List[KGRelation]) -> List[KGRelation]:
        return [self.relations.add(r) for r in rs]

    def find_by_norm(self, name_or_alias: str) -> Optional[KGEntity]:
        return self.entities.find_by_norm(name_or_alias)

    def find_by_normalized_id(self, nid: str) -> Optional[KGEntity]:
        return self.entities.find_by_normalized_id(nid)

    # 导出
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "meta": self.meta,
            "entities": [e.to_dict() for e in self.entities.all()],
            "relations": [r.to_dict() for r in self.relations.all()],
        }

    def to_json(self, dirpath: str = ".") -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        p = Path(dirpath)
        p.mkdir(parents=True, exist_ok=True)
        path = p / f"subgraph-{self.id or self.name or 'noname'}-{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return str(path)

    # 合并到全局 Memory
    def merge_into(self, mem: "Memory") -> Dict[str, str]:
        """
        把当前子图合并进全局 memory：
        - 实体通过 mem.entities.upsert 去重/合并
        - 返回: 子图实体旧ID -> 全局实体ID 的映射，用于对齐关系
        """
        id_map: Dict[str, str] = {}

        # 1) 合并实体
        for e in self.entities.all():
            old_id = e.entity_id
            e_copy = KGEntity(**e.to_dict())
            merged = mem.entities.upsert(e_copy)
            if old_id:
                id_map[old_id] = merged.entity_id

        # 2) 合并关系（用映射替换 head/tail）
        for r in self.relations.all():
            head = id_map.get(r.head_id, r.head_id)
            tail = id_map.get(r.tail_id, r.tail_id)
            mem.relations.add(KGRelation(
                rel_type=r.rel_type,
                head_id=head,
                tail_id=tail,
                props=dict(r.props or {})
            ))

        # 3) 可选：把子图本身登记到全局（如果你希望 Memory 记住这个子图）
        if self.id:
            mem.register_subgraph(self)

        return id_map


# ---------- 全局共享记忆池 ----------
class Memory:
    def __init__(self):
        self.entities = EntityStore()
        self.relations = RelationStore()
        # 新增：子图注册表
        self.subgraphs: Dict[str, Subgraph] = {}

    def upsert_many_entities(self, entities: List[KGEntity]) -> List[KGEntity]:
        return self.entities.upsert_many(entities)

    # 子图管理：按 ID 注册 & 获取
    def register_subgraph(self, sg: Subgraph) -> None:
        """
        将子图登记到全局索引中。
        要求 sg.id 是非空字符串（由你在创建 Subgraph 时传入）。
        """
        if not sg.id:
            return
        self.subgraphs[sg.id] = sg

    def get_subgraph(self, sg_id: str) -> Optional[Subgraph]:
        """根据子图ID获取子图；如果不存在则返回 None。"""
        return self.subgraphs.get(sg_id)

    def dump_json(self, dirpath: str = ".") -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dirp = Path(dirpath)
        dirp.mkdir(parents=True, exist_ok=True)

        path = dirp / f"memory-{ts}.json"
        data = {
            "entities": [e.to_dict() for e in self.entities.all()],
            "relations": [r.to_dict() for r in self.relations.all()],
            "subgraphs": list(self.subgraphs.keys()),
            "meta": {"generated_at": ts},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(path)


# 模块级全局实例：各 Agent 直接 `from shared_memory_min import memory`
memory = Memory()