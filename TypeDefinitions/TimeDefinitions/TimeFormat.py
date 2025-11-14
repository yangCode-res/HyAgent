from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
@dataclass
class TimeFormat:
    """时间格式定义信息。
    - type: 时间格式的类型（instant、interval、relative其中一个)
    - value: 具体时间值（datetime 对象 instant类型）
    - start_time: 起始时间（字符串格式，如 "2023-01-01" interval类型）
    - end_time: 结束时间（字符串格式，如 "2023-12-31" interval类型）
    - offset: 相对时间偏移描述（relative类型）
    - source: 时间信息来源（如文章pid）
    - origin_text: 原始时间文本（从文本中提取的时间表达）
    """
    type:str
    value: Optional[datetime]=None
    start_time: Optional[str]=None
    end_time: Optional[str]=None
    granularity: Optional[str]=None
    offset: Optional[str]=None
    source: Optional[str]=None
    origin_text: Optional[str]=None
    
    def get_start_time(self) -> Optional[str]:
        return self.start_time
    
    def get_end_time(self) -> Optional[str]:
        return self.end_time
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __str__(self) -> str:
        if self.type == "instant":
            return f"{self.value} (Precision: {self.granularity})"
        elif self.type == "relative":
            return f"Offset: {self.offset} (Precision: {self.granularity})"
        elif self.type == "interval":
            return f"[{self.start_time} - {self.end_time}] (Precision: {self.granularity})"
        return "Unknown TimeFormat"
    
    @classmethod
    def from_dict(cls, data: dict) -> "TimeFormat":
        return cls(**data)
