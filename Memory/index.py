from __future__ import annotations

from dataclasses import dataclass, field, asdict
from threading import RLock
from typing import Any, Dict, List, Optional
import uuid


# ===== 数据结构定义 =====

@dataclass
class EntityTypeDefinition:
    """
    实体类型定义
    - name: 类型名（如 DRUG、DISEASE）
    - description: 类型的解释
    - examples: 示例（若干字符串）
    - include: 包含/同义/相关词
    """

    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityRecord:
    """
    实体记录
    - id: 唯一标识（默认自动生成）
    - name: 实体名称
    - type_name: 所属实体类型名（如 DRUG、DISEASE）
    - description/examples/include: 语义信息
    """

    name: str
    type_name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    include: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ===== 共享记忆池（可扩展） =====

class SharedMemory:
    """
    共享记忆池（可扩展版本）

    当前实现：
    - 存储实体类型：增删改查
    - 存储实体：按类型组织，支持增删改查

    预留扩展：
    - 关系存储（relations_by_signature 等）
    - 子图存储（subgraphs_by_id 等）
    """

    def __init__(self) -> None:
        self._lock = RLock()

        # 实体类型存储（name -> definition）
        self._entity_types: Dict[str, EntityTypeDefinition] = {}

        # 实体存储（type_name -> {entity_id -> EntityRecord}）
        self._entities_by_type: Dict[str, Dict[str, EntityRecord]] = {}
        # 名称索引（便于查找，同类型下 name/alias 到 entity_id 的映射）
        self._name_index: Dict[str, Dict[str, str]] = {}

        # 预留扩展（不实现，留接口）
        self._relations: Dict[str, Any] = {}
        self._subgraphs: Dict[str, Any] = {}

    # ===== 实体类型：增删改查 =====

    def upsert_entity_type(self, definition: EntityTypeDefinition) -> None:
        with self._lock:
            self._entity_types[definition.name] = definition
            if definition.name not in self._entities_by_type:
                self._entities_by_type[definition.name] = {}
            if definition.name not in self._name_index:
                self._name_index[definition.name] = {}

    def get_entity_type(self, name: str) -> Optional[EntityTypeDefinition]:
        with self._lock:
            return self._entity_types.get(name)

    def list_entity_types(self) -> List[EntityTypeDefinition]:
        with self._lock:
            return list(self._entity_types.values())

    def remove_entity_type(self, name: str) -> bool:
        with self._lock:
            if name in self._entity_types:
                # 同时清理该类型下的实体与索引
                self._entity_types.pop(name, None)
                self._entities_by_type.pop(name, None)
                self._name_index.pop(name, None)
                return True
            return False

    # ===== 实体：增删改查 =====

    def upsert_entity(self, entity: EntityRecord) -> str:
        with self._lock:
            # 确保类型存在
            if entity.type_name not in self._entity_types:
                # 自动创建一个空定义，避免异常（也可改为抛错）
                self._entity_types[entity.type_name] = EntityTypeDefinition(
                    name=entity.type_name,
                    description=f"Auto-created type for {entity.type_name}",
                )
                self._entities_by_type.setdefault(entity.type_name, {})
                self._name_index.setdefault(entity.type_name, {})

            # 写入实体
            bucket = self._entities_by_type.setdefault(entity.type_name, {})
            bucket[entity.id] = entity

            # 更新名称索引（name 与 aliases 都索引）
            name_map = self._name_index.setdefault(entity.type_name, {})
            name_map[entity.name.lower()] = entity.id
            for alias in entity.aliases:
                name_map[alias.lower()] = entity.id

            return entity.id

    def get_entity(self, type_name: str, entity_id: str) -> Optional[EntityRecord]:
        with self._lock:
            return self._entities_by_type.get(type_name, {}).get(entity_id)

    def find_entity_by_name(self, type_name: str, name_or_alias: str) -> Optional[EntityRecord]:
        with self._lock:
            name_map = self._name_index.get(type_name, {})
            entity_id = name_map.get(name_or_alias.lower())
            if not entity_id:
                return None
            return self._entities_by_type.get(type_name, {}).get(entity_id)

    def list_entities(self, type_name: str) -> List[EntityRecord]:
        with self._lock:
            return list(self._entities_by_type.get(type_name, {}).values())

    def remove_entity(self, type_name: str, entity_id: str) -> bool:
        with self._lock:
            bucket = self._entities_by_type.get(type_name)
            if not bucket or entity_id not in bucket:
                return False
            entity = bucket.pop(entity_id)

            # 清理名称索引
            name_map = self._name_index.get(type_name, {})
            name_map.pop(entity.name.lower(), None)
            for alias in entity.aliases:
                name_map.pop(alias.lower(), None)
            return True

    # ===== 导入/导出 =====

    def export_snapshot(self) -> Dict[str, Any]:
        """导出当前内存快照为可序列化的 dict。"""
        with self._lock:
            return {
                "entity_types": [asdict(t) for t in self._entity_types.values()],
                "entities": {
                    t: [asdict(e) for e in self._entities_by_type.get(t, {}).values()]
                    for t in self._entities_by_type.keys()
                },
            }

    def load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """从快照恢复（会覆盖当前内存）。"""
        with self._lock:
            self._entity_types.clear()
            self._entities_by_type.clear()
            self._name_index.clear()

            for t in snapshot.get("entity_types", []):
                definition = EntityTypeDefinition(
                    name=t.get("name"),
                    description=t.get("description") or "",
                    examples=list(t.get("examples") or []),
                    include=list(t.get("include") or []),
                    metadata=dict(t.get("metadata") or {}),
                )
                self.upsert_entity_type(definition)

            entities_by_type = snapshot.get("entities", {})
            for type_name, entities in entities_by_type.items():
                for e in entities:
                    record = EntityRecord(
                        id=e.get("id") or str(uuid.uuid4()),
                        name=e.get("name") or "",
                        type_name=type_name,
                        description=e.get("description"),
                        examples=list(e.get("examples") or []),
                        include=list(e.get("include") or []),
                        aliases=list(e.get("aliases") or []),
                        confidence=e.get("confidence"),
                        metadata=dict(e.get("metadata") or {}),
                    )
                    self.upsert_entity(record)

    # ===== 其他 =====

    def reset(self) -> None:
        """清空所有存储。"""
        with self._lock:
            self._entity_types.clear()
            self._entities_by_type.clear()
            self._name_index.clear()
            self._relations.clear()
            self._subgraphs.clear()


