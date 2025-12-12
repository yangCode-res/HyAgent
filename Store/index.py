# Memory/hub.py
from Memory.index import Memory

__all__ = ["get_memory"]

_memory = None

def get_memory() -> Memory:
    global _memory
    if _memory is None:
        _memory = Memory()      # 只在第一次创建
    return _memory