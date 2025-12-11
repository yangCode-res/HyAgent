# shared_memory_min.py
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
# 按你的项目结构保留导入
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple

# ===================== 简单对齐数据结构 =====================

@dataclass
class SimpleAlignment:
    """
    一条最简单的实体对齐记录：
    - src_subgraph / src_entity: 源子图 + 源实体ID
    - tgt_subgraph / tgt_entity: 目标子图 + 目标实体ID
    """
    src_subgraph: str
    src_entity: str
    tgt_subgraph: str
    tgt_entity: str


# ===================== 实体存储 =====================

class EntityStore:
    """
    负责全局 / 子图内实体去重与索引。

    规则：
    - 忽略外部传入的 entity_id，由本类统一分配 ent:xxxxxx。
    - 优先按 normalized_id 合并，其次按规范化 name 合并。
    - 维护:
        - by_id: id -> KGEntity
        - idx_normid: normalized_id.lower() -> id
        - idx_name: norm(name).lower() -> id
    """
    def __init__(self):
        self.by_id: Dict[str, KGEntity] = {}
        self.idx_normid: Dict[str, str] = {}
        self.idx_name: Dict[str, str] = {}

    def _nid(self) -> str:
        return f"ent:{uuid.uuid4().hex[:12]}"

    def _key(self, s: str) -> str:
        return (s or "").strip().lower()

    def upsert(self, e: KGEntity) -> KGEntity:
        # 统一忽略外部 entity_id，避免历史设计冲突
        e.entity_id = ""

        norm = self._key(e.name)

        # 1) 尝试用 normalized_id 合并
        if e.normalized_id and e.normalized_id != "N/A":
            k = self._key(e.normalized_id)
            if k in self.idx_normid:
                return self._merge(self.by_id[self.idx_normid[k]], e)

        # 2) 尝试用规范化名称合并
        if norm and norm in self.idx_name:
            return self._merge(self.by_id[self.idx_name[norm]], e)

        # 3) 新建实体
        new_id = self._nid()
        e.entity_id = new_id
        self.by_id[new_id] = e

        if e.normalized_id and e.normalized_id != "N/A":
            self.idx_normid[self._key(e.normalized_id)] = new_id
        if norm:
            self.idx_name[norm] = new_id

        return e

    def update(self, entities: List[KGEntity]):
        for entity in entities:
            former_entity = self.by_id.get(entity.entity_id)
            if former_entity:
                if former_entity == entity:
                    continue
                former_entity.name = entity.name
                former_entity.entity_type = entity.entity_type
                self.by_id[entity.entity_id] = former_entity

    def _merge(self, base: KGEntity, inc: KGEntity) -> KGEntity:
        # 类型：prefer 更具体
        if base.entity_type == "Unknown" and inc.entity_type != "Unknown":
            base.entity_type = inc.entity_type

        # 名称：更长更可读 -> 升级为主名，旧主名入别名
        if inc.name and len(inc.name) > len(base.name):
            if base.name:
                base.aliases.append(base.name)
            base.name = inc.name

        # 本体ID：优先非 N/A
        if base.normalized_id == "N/A" and inc.normalized_id and inc.normalized_id != "N/A":
            base.normalized_id = inc.normalized_id
            self.idx_normid[self._key(base.normalized_id)] = base.entity_id

        # 别名并集（大小写不敏感去重）
        pool = {self._key(a): a for a in base.aliases}
        for a in ([inc.name] if inc.name else []) + (inc.aliases or []):
            if a:
                pool.setdefault(self._key(a), a)
        base.aliases = sorted(pool.values(), key=str.lower)

        # 主名可能变化了，更新 name 索引
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


# ===================== 关系存储 =====================

class RelationStore:
    """
    by_head:通过查询头部实体获取三元组列表
    by_relation:通过查询关系类型获取三元组列表
    by_tail:通过查询尾部实体获取三元组列表
    """
    def __init__(self):
        self.triples: List[KGTriple] = []
        self.by_head: Dict[str, List[KGTriple]] = {}
        self.by_relation: Dict[str, List[KGTriple]] = {}
        self.by_tail: Dict[str, List[KGTriple]] = {}

    def _rid(self) -> str:
        return f"rel:{uuid.uuid4().hex[:12]}"

    def add(self, triple: KGTriple):
        """插入一个三元组"""
        self.triples.append(triple)

        # 这里按照 KGTriple 的属性名来取；你项目里如果是 rel_type/head_id/tail_id，
        # 可以在 KGTriple 里做 property 映射。
        relation = getattr(triple, "relation", None) or getattr(triple, "rel_type", None)
        head = getattr(triple, "head", None) or getattr(triple, "head_id", None)
        tail = getattr(triple, "tail", None) or getattr(triple, "tail_id", None)

        if relation is not None:
            self.by_relation.setdefault(relation, []).append(triple)
        if head is not None:
            self.by_head.setdefault(head, []).append(triple)
        if tail is not None:
            self.by_tail.setdefault(tail, []).append(triple)

    def add_many(self, triples: List[KGTriple]):
        for triple in triples:
            self.add(triple)

    def find_Triple_by_head_and_tail(self, head: str, tail: str) -> Optional[KGTriple]:
        """通过头实体和尾实体查找三元组"""
        head_triples = self.by_head.get(head, [])
        for triple in head_triples:
            t = getattr(triple, "tail", None) or getattr(triple, "tail_id", None)
            if t == tail:
                return triple
        return None

    def all(self) -> List[KGTriple]:
        return self.triples

    def reset(self):
        """重置relationstore"""
        self.triples.clear()
        self.by_head.clear()
        self.by_relation.clear()
        self.by_tail.clear()


# ===================== 对齐关系存储（极简版） =====================

class AlignmentStore:
    """
    只存“实体对实体”的对齐结果，不带ID、不带相似度：
    - 每条记录是一个 SimpleAlignment
    - 主索引：by_source[(src_subgraph, src_entity)] -> List[SimpleAlignment]
    """
    def __init__(self):
        self.by_source: Dict[tuple, List[SimpleAlignment]] = {}

    def add(
        self,
        src_subgraph: str,
        src_entity: str,
        tgt_subgraph: str,
        tgt_entity: str,
    ) -> None:
        """
        添加一条对齐边；如果已经存在则不重复添加。
        """
        key = (src_subgraph, src_entity)
        lst = self.by_source.setdefault(key, [])
        # 去重：相同 tgt_subgraph + tgt_entity 就不再加
        for rec in lst:
            if rec.tgt_subgraph == tgt_subgraph and rec.tgt_entity == tgt_entity:
                return
        lst.append(SimpleAlignment(
            src_subgraph=src_subgraph,
            src_entity=src_entity,
            tgt_subgraph=tgt_subgraph,
            tgt_entity=tgt_entity,
        ))

    def save_from_alignment_dict(
        self,
        alignment_dict: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ) -> None:
        """
        直接接收 AlignmentTripleAgent.build_entity_alignment 的 refined_alignment：
        {
          src_sg_id: {
            src_eid: [
              {
                "target_subgraph": ...,
                "target_entity": ...,
                ...
              }, ...
            ]
          }
        }
        只取子图 + 实体ID 四个字段，其他信息忽略。
        """
        for src_sg_id, ent_map in alignment_dict.items():
            for src_eid, matches in ent_map.items():
                for m in matches:
                    tgt_sg_id = m.get("target_subgraph")
                    tgt_eid = m.get("target_entity")
                    if not tgt_sg_id or not tgt_eid:
                        continue
                    self.add(
                        src_subgraph=src_sg_id,
                        src_entity=src_eid,
                        tgt_subgraph=tgt_sg_id,
                        tgt_entity=tgt_eid,
                    )

    # -------- 查询接口 --------

    def get_for_source(
        self,
        src_subgraph: str,
        src_entity: str,
    ) -> List[SimpleAlignment]:
        """给定 (src_subgraph, src_entity) 查询所有目标实体。"""
        return list(self.by_source.get((src_subgraph, src_entity), []))

    def all(self) -> List[SimpleAlignment]:
        """返回所有对齐边（扁平列表）"""
        result: List[SimpleAlignment] = []
        for lst in self.by_source.values():
            result.extend(lst)
        return result

    # -------- 序列化 / 反序列化，用于 Memory.dump_json --------

    def to_list(self) -> List[Dict[str, Any]]:
        """
        导出为 list[dict] 方便 dump_json：
        [
          {
            "src_subgraph": ...,
            "src_entity": ...,
            "tgt_subgraph": ...,
            "tgt_entity": ...
          },
          ...
        ]
        """
        return [asdict(rec) for rec in self.all()]

    def from_list(self, data: List[Dict[str, Any]]) -> None:
        """
        从 list[dict] 恢复；会清空当前内容。
        """
        self.by_source.clear()
        for d in data or []:
            self.add(
                src_subgraph=d["src_subgraph"],
                src_entity=d["src_entity"],
                tgt_subgraph=d["tgt_subgraph"],
                tgt_entity=d["tgt_entity"],
            )


# ===================== 子图 =====================

class Subgraph:
    """
    子图：
    - 拥有自己的 EntityStore / RelationStore
    - 用 subgraph_id 标识（你在创建时传入）
    - 可单独导出，也可 merge_into 全局 Memory

    NOTICE:subgraph_id的格式为 PMID_索引 的格式 用于标识文章_段落
    """
    def __init__(
        self,
        subgraph_id: str,
        name: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.id = subgraph_id
        self.name = name
        self.meta = dict(meta or {})
        self.entities = EntityStore()
        self.relations = RelationStore()

    # 子图内操作
    def upsert_entity(self, e: KGEntity) -> KGEntity:
        return self.entities.upsert(e)

    def upsert_many_entities(self, ents: List[KGEntity]) -> List[KGEntity]:
        return self.entities.upsert_many(ents)

    def add_relation(self, r: KGTriple) -> KGTriple:
        return self.relations.add(r)

    def add_relations(self, rs: List[KGTriple]) -> List[KGTriple]:
        return self.relations.add_many(rs)

    def get_meta(self) -> Dict[str, Any]:
        return self.meta

    def get_relations(self) -> List[KGTriple]:
        return self.relations.all()

    def get_entities(self) -> List[KGEntity]:
        return self.entities.all()

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
        把当前子图内容合并进全局 memory：
        - 实体通过 mem.entities.upsert 去重/合并
        - 关系使用合并后的全局实体ID重写 head/tail
        - 返回: {子图旧实体ID -> 全局实体ID}
        - 同时将该子图注册到 mem.subgraphs（保存子图视图）
        """
        id_map: Dict[str, str] = {}

        # 1) 合并实体
        for e in self.entities.all():
            old_id = e.entity_id
            e_copy = KGEntity(**e.to_dict())  # 避免直接改动子图内对象
            merged = mem.entities.upsert(e_copy)
            if old_id:
                id_map[old_id] = merged.entity_id

        # 2) 合并关系
        for r in self.relations.all():
            # 这里按照你 KGTriple 的字段来取；示例用 rel_type/head_id/tail_id
            head = id_map.get(getattr(r, "head_id", None) or getattr(r, "head", None),
                              getattr(r, "head_id", None) or getattr(r, "head", None))
            tail = id_map.get(getattr(r, "tail_id", None) or getattr(r, "tail", None),
                              getattr(r, "tail_id", None) or getattr(r, "tail", None))
            rel_type = getattr(r, "rel_type", None) or getattr(r, "relation", None)

            mem.relations.add(KGTriple(
                rel_type=rel_type,
                head_id=head,
                tail_id=tail,
                props=dict(getattr(r, "props", {}) or {}),
            ))

        # 3) 记录子图本身（保持原局部视图）
        if self.id:
            mem.register_subgraph(self)

        return id_map


# ===================== 全局共享记忆池 =====================
# ===================== 关键实体存储（简单列表） =====================

class KeyEntityStore:
    """
    用于存储“关键实体”的简单列表：
    - 不做去重合并，不维护任何索引
    - 只提供追加、批量追加、获取全部、清空等基础操作
    """
    def __init__(self):
        self.entities: List[KGEntity] = []

    def add(self, e: KGEntity) -> None:
        """追加一个关键实体"""
        self.entities.append(e)

    def add_many(self, ents: List[KGEntity]) -> None:
        """批量追加关键实体"""
        self.entities.extend(ents)

    def all(self) -> List[KGEntity]:
        """返回所有关键实体（原始列表）"""
        return list(self.entities)

    def reset(self) -> None:
        """清空关键实体列表"""
        self.entities.clear()
class Memory:
    """
    全局共享记忆池：
    - 持有一个全局 EntityStore / RelationStore
    - 注册多个 Subgraph（以字符串ID索引）
    - 支持导出统一快照 JSON（含全局 + 子图内部详细内容）
    """
    def __init__(self):
        self.entities = EntityStore()
        self.relations = RelationStore()
        self.subgraphs: Dict[str, Subgraph] = {}
        # 新增：极简实体对齐存储
        self.alignments = AlignmentStore()
        # 新增：关键实体列表存储
        self.key_entities = KeyEntityStore()
        self.keyword_entity_map: Dict[str, List[KGEntity]] = {}
        self._extracted_paths: List[dict] = []
        self.entity_id_mapping_path: Optional[str] = None
        
    def add_extracted_path(
        self,
        node_path: List[KGEntity],
        edge_path: List[KGTriple],
    ) -> None:
        """
        把一次抽取到的路径存起来，节点和边都直接以对象形式存。
        """
        self._extracted_paths.append(
            {
                "nodes": list(node_path),   # 做一份拷贝，避免后面被修改
                "edges": list(edge_path),
            }
        )
    def get_extracted_paths(self) -> List[dict]:
        """
        返回所有已存的路径记录，每条记录为 {'nodes': [...], 'edges': [...]}。
        """
        return list(self._extracted_paths)
    def upsert_many_entities(self, entities: List[KGEntity]) -> List[KGEntity]:
        return self.entities.upsert_many(entities)

    # 子图管理
    def register_subgraph(self, sg: Subgraph) -> None:
        if not sg.id:
            return
        self.subgraphs[sg.id] = sg

    def get_subgraph(self, sg_id: str) -> Optional[Subgraph]:
        return self.subgraphs.get(sg_id)
    
    def remove_subgraph(self, sg_id: str) -> None:
        if sg_id in self.subgraphs:
            del self.subgraphs[sg_id]
    def add_key_entity(self, e: KGEntity) -> None:
        self.key_entities.add(e)
    def add_key_entities(self, ents: List[KGEntity]) -> None:
        self.key_entities.add_many(ents)
    def get_key_entities(self) -> List[KGEntity]:
        return self.key_entities.all()
    def get_allRealationShip(self)-> List[KGTriple]:
        return self.relations.all()
    def add_keyword_entities(self, keyword: str, ents: List[KGEntity]) -> None:
        """
        覆盖式设置某个 keyword 对应的实体列表。
        """
        keyword = (keyword or "").strip()
        if not keyword:
            return
        self.keyword_entity_map[keyword] = list(ents)

    def append_keyword_entities(self, keyword: str, ents: List[KGEntity]) -> None:
        """
        追加式添加某个 keyword 的实体列表（在原有基础上 extend）。
        """
        keyword = (keyword or "").strip()
        if not keyword:
            return
        self.keyword_entity_map.setdefault(keyword, []).extend(ents)

    def get_keyword_entities(self, keyword: str) -> List[KGEntity]:
        """
        获取某个 keyword 目前映射的实体列表。
        """
        return list(self.keyword_entity_map.get(keyword, []))

    def get_keyword_entity_map(self) -> Dict[str, List[KGEntity]]:
        """
        获取完整的 keyword -> [KGEntity, ...] 映射（浅拷贝）。
        """
        return {k: list(v) for k, v in self.keyword_entity_map.items()}
    # 导出全局快照（包含子图内部）
    def dump_json(self, dirpath: str = ".") -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        dirp = Path(dirpath)
        dirp.mkdir(parents=True, exist_ok=True)

        path = dirp / f"memory-{ts}.json"
        data = {
            "entities": [e.to_dict() for e in self.entities.all()],
            "relations": [r.to_dict() for r in self.relations.all()],
            "subgraphs": {
                sg_id: sg.to_dict()
                for sg_id, sg in self.subgraphs.items()
            },
            # 新增：对齐结果
            "alignments": self.alignments.to_list(),
              # 新增：关键实体列表
            "key_entities": [e.to_dict() for e in self.key_entities.all()],
            "keyword_entity_map": {
                kw: [e.to_dict() for e in ents]
                for kw, ents in self.keyword_entity_map.items()
            },
            "meta": {"generated_at": ts,"entity_id_mapping_path": self.entity_id_mapping_path,},
            "paths": [
            {
                "nodes": [n.to_dict() for n in p.get("nodes", [])],
                "edges": [e.to_dict() for e in p.get("edges", [])],
            }
            for p in getattr(self, "_extracted_paths", [])
        ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(path)


def load_memory_from_json(path_or_data: Any) -> Memory:
    """
    从 Memory.dump_json() 导出的快照恢复为一个新的 Memory 实例。
    - path_or_data: 文件路径(str/Path) 或 已解析的 dict
    - 不修改当前全局 memory 变量，返回新的 Memory 对象
    - 保留原始 entity_id，不走 upsert；同时重建各类倒排索引
    """
    # ==== 小工具（局部作用域，避免增加全局改动） ====
    def _coerce_entity(ed: Any) -> KGEntity:
        if isinstance(ed, KGEntity):
            return ed
        # 优先 from_dict
        try:
            from_dict = getattr(KGEntity, "from_dict", None)
            if callable(from_dict):
                return from_dict(ed)
        except Exception:
            pass
        # 回退：直接解包
        return KGEntity(**ed)

    def _coerce_triple(rd: Any) -> KGTriple:
        if isinstance(rd, KGTriple):
            return rd
        # 优先 from_dict
        try:
            from_dict = getattr(KGTriple, "from_dict", None)
            if callable(from_dict):
                return from_dict(rd)
        except Exception:
            pass
        # 回退：尝试直接解包；不行就做字段映射
        try:
            return KGTriple(**rd)
        except Exception:
            mapped = dict(rd)
            # 字段别名兼容
            if "relation" in mapped and "rel_type" not in mapped:
                mapped["rel_type"] = mapped["relation"]
            if "head" in mapped and "head_id" not in mapped:
                mapped["head_id"] = mapped["head"]
            if "tail" in mapped and "tail_id" not in mapped:
                mapped["tail_id"] = mapped["tail"]
            # 只保留常见字段
            allowed = {"rel_type", "relation", "head_id", "tail_id", "head", "tail", "props"}
            slim = {k: v for k, v in mapped.items() if k in allowed}
            return KGTriple(**slim)

    # ==== 读取数据 ====
    if isinstance(path_or_data, (str, Path)):
        with open(path_or_data, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(path_or_data, dict):
        data = path_or_data
    else:
        raise TypeError("load_memory_from_json expects a file path (str/Path) or a dict")

    # ==== 构建新的 Memory ====
    mem = Memory()

    # ---- 恢复全局实体 ----
    for ed in data.get("entities", []):
        e = _coerce_entity(ed)
        # 直接写入并重建索引（不走 upsert，避免重分配ID）
        mem.entities.by_id[e.entity_id] = e
        if getattr(e, "normalized_id", None) and e.normalized_id != "N/A":
            mem.entities.idx_normid[mem.entities._key(e.normalized_id)] = e.entity_id
        if getattr(e, "name", None):
            mem.entities.idx_name[mem.entities._key(e.name)] = e.entity_id

    # ---- 恢复全局关系 ----
    for rd in data.get("relations", []):
        r = _coerce_triple(rd)
        mem.relations.triples.append(r)

        rel = getattr(r, "relation", None) or getattr(r, "rel_type", None)
        head = getattr(r, "head", None) or getattr(r, "head_id", None)
        tail = getattr(r, "tail", None) or getattr(r, "tail_id", None)

        if rel is not None:
            mem.relations.by_relation.setdefault(rel, []).append(r)
        if head is not None:
            mem.relations.by_head.setdefault(head, []).append(r)
        if tail is not None:
            mem.relations.by_tail.setdefault(tail, []).append(r)

    # ---- 恢复子图 ----
    for sg_id, sgd in (data.get("subgraphs") or {}).items():
        sg = Subgraph(
            subgraph_id=sgd.get("id", sg_id),
            name=sgd.get("name", ""),
            meta=sgd.get("meta", {}),
        )

        # 子图实体
        for ed in sgd.get("entities", []):
            e = _coerce_entity(ed)
            sg.entities.by_id[e.entity_id] = e
            if getattr(e, "normalized_id", None) and e.normalized_id != "N/A":
                sg.entities.idx_normid[sg.entities._key(e.normalized_id)] = e.entity_id
            if getattr(e, "name", None):
                sg.entities.idx_name[sg.entities._key(e.name)] = e.entity_id

        # 子图关系
        for rd in sgd.get("relations", []):
            r = _coerce_triple(rd)
            sg.relations.triples.append(r)

            rel = getattr(r, "relation", None) or getattr(r, "rel_type", None)
            head = getattr(r, "head", None) or getattr(r, "head_id", None)
            tail = getattr(r, "tail", None) or getattr(r, "tail_id", None)

            if rel is not None:
                sg.relations.by_relation.setdefault(rel, []).append(r)
            if head is not None:
                sg.relations.by_head.setdefault(head, []).append(r)
            if tail is not None:
                sg.relations.by_tail.setdefault(tail, []).append(r)

        mem.register_subgraph(sg)

    # ---- 恢复对齐结果 ----
    align_data = data.get("alignments", [])
    if align_data:
        mem.alignments.from_list(align_data)
     # ---- 恢复关键实体列表 ----
    key_ents_data = data.get("key_entities", [])
    for ed in key_ents_data:
        e = _coerce_entity(ed)
        mem.key_entities.add(e)
    kw_map_data = data.get("keyword_entity_map", {}) or {}
    for kw, ents_list in kw_map_data.items():
        ents: List[KGEntity] = []
        for ed in ents_list or []:
            ents.append(_coerce_entity(ed))
        mem.keyword_entity_map[kw] = ents
    paths_data = data.get("paths", [])
    for pd in paths_data:
        node_ents: List[KGEntity] = []
        edge_triples: List[KGTriple] = []

        for ed in pd.get("nodes", []):
            node_ents.append(_coerce_entity(ed))
        for rd in pd.get("edges", []):
            edge_triples.append(_coerce_triple(rd))

        mem.add_extracted_path(node_ents, edge_triples)
    mem.entity_id_mapping_path = (
        data.get("entity_id_mapping_path")
        or (data.get("meta", {}) or {}).get("entity_id_mapping_path")
    )
    return mem


# 全局实例：所有 Agent 请统一从这里 import
memory = Memory()