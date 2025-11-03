from __future__ import annotations

from typing import Any, Dict, List, Optional

from EntityTypeDefinitions.index import format_all_entity_definitions
from Core.Agent import Agent
class EntityExtractionAgent(Agent):
    """
    实体抽取 Agent 模板。

    继承自通用父类 Agent，预置：
    - template_id = "entity_extractor"
    - 合理的默认 name/responsibility

    你可以重写 extract_from_text() 来接入实际的 NER/LLM 抽取逻辑。
    """

    def __init__(
        self,
        *,
        name: str = "Entity Extraction Agent",
        responsibility: str = '''You are a specialized Entity Extraction Agent for biomedical literature. 
        Your task is to identify and classify all biomedical entities with high precision and appropriate ontological mapping''',
        entity_focus: Optional[List[Any]] = None,
        relation_focus: Optional[List[Any]] = None,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            template_id="entity_extractor",
            name=name,
            responsibility=responsibility,
            entity_focus=list(entity_focus or []),
            relation_focus=list(relation_focus or []),
            priority=priority,
            metadata=dict(metadata or {}),
        )
    def step1(self, text: str) -> str:
        """
        Step 1: Extract all entities from the text
        """
        return text
    def step2(self, text: str) -> str:
        """
        Step 2: Classify the entities into the appropriate ontology
        """
        return text
    def run(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        执行实体抽取主流程。

        入参格式示例：[{"id": "doc1", "text": "..."}, ...]
        返回格式示例：[{"doc_id": str, "entities": [{"name": str, "type": str, "span": [start, end]}]}]
        """
        results: List[Dict[str, Any]] = []
        for doc in documents:
            doc_id = doc.get("id") or ""
            text = doc.get("text") or ""
            entities = self.extract_from_text(text)
            results.append({"doc_id": doc_id, "entities": entities})
        return results

    def extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        文本级实体抽取的占位实现（请在子类或实例中重写）。

        默认返回空列表。你可以对接：
        - 传统 NER 模型（如 spaCy/BioBERT）
        - LLM 提示词抽取（调用你的 OpenAI/DeepSeek 客户端）
        - 规则 + 词典匹配
        返回的每个实体建议包含 name/type/span 等字段。
        """
        return []


