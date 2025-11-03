from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Agent:
    """
    通用父类 Agent。

    必备属性（对齐 meta_agent 中的规划字段）：
    - template_id: 模板 ID
    - name: Agent 名称
    - responsibility: 该 Agent 的职责说明
    - entity_focus: 关注的实体类型列表（在不同工程中可为字符串或枚举）
    - relation_focus: 关注的关系类型列表（在不同工程中可为字符串或枚举）
    - priority: 该 Agent 的优先级（数值越小，优先级越高）
    """

    template_id: str
    name: str
    responsibility: str
    entity_focus: List[Any] = field(default_factory=list)
    relation_focus: List[Any] = field(default_factory=list)
    priority: int = 1

    # 可选扩展字段（不强依赖于 Agent2 的实现，便于后续扩展）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def configure(
        self,
        *,
        template_id: Optional[str] = None,
        name: Optional[str] = None,
        responsibility: Optional[str] = None,
        entity_focus: Optional[List[Any]] = None,
        relation_focus: Optional[List[Any]] = None,
        priority: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """按需更新 Agent 的配置。"""
        if template_id is not None:
            self.template_id = template_id
        if name is not None:
            self.name = name
        if responsibility is not None:
            self.responsibility = responsibility
        if entity_focus is not None:
            self.entity_focus = list(entity_focus)
        if relation_focus is not None:
            self.relation_focus = list(relation_focus)
        if priority is not None:
            self.priority = int(priority)
        if metadata is not None:
            self.metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典，便于日志/序列化。"""
        return asdict(self)

    # 预留的运行接口，子类按需实现
    def run(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        """执行 Agent 的主流程（需由具体子类实现）。"""
        raise NotImplementedError("Subclasses must implement run()")


