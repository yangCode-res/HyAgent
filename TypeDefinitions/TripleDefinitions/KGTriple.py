from dataclasses import asdict, dataclass


@dataclass
class KGTriple:
    """三元组定义信息。

    - head: 头实体名称
    - relation: 关系类型
    - tail: 尾实体名称
    - confidence: 置信度（0-1 之间的浮点数）
    - evidence: 证据（支持该关系的直接引用）
    - temporal_info: 时间信息（如有）
    - mechanism: 机制描述（50-100词）
    - source: 信息来源（如文章pid）
    """

    head: str
    relation: str
    tail: str
    confidence: float
    evidence: str="unknown"
    source:str="unknown"
    mechanism:str="unknown"
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return f"({self.head.lower()}, -[{self.relation.lower()}]->, {self.tail.lower()})"
    
    @classmethod
    def from_dict(cls, data: dict) -> "KGTriple":
        return cls(**data)

def export_triples_to_dicts(triples: list[KGTriple]) -> list[dict]:
    """将 KGTriple 列表导出为字典列表。"""
    triple_dict=[triple.to_dict() for triple in triples]
    return triple_dict