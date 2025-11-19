from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
from unittest.case import doModuleCleanups
from TypeDefinitions.EntityTypeDefinitions.index import format_all_entity_definitions,format_entity_definition
from Core.Agent import Agent
from TypeDefinitions.EntityTypeDefinitions.index import ENTITY_DEFINITIONS, EntityDefinition, EntityType
from ExampleText.index import ExampleText
from ChatLLM.index import ChatLLM
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from tqdm import tqdm
from Memory.index import Subgraph
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
        system: str = "You are a careful biomedical classifier. Return STRICT JSON only.",
        responsibility: str = '''You are a specialized Entity Extraction Agent for biomedical literature. 
        Your task is to identify and classify all biomedical entities with high precision and appropriate ontological mapping''',
        entity_focus: Optional[List[Any]] = None,
        relation_focus: Optional[List[Any]] = None,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        THRESH: float = 0.6
    ) -> None:
        # 按父类签名对齐：先构造 OpenAI 客户端与模型名，再调用父类 __init__
        llm = ChatLLM(system=system)
        super().__init__(
            client=llm.client,
            model_name=llm.model,
            system_prompt=system,
        )
        # 用父类提供的 configure 设置模板/名称/职责等元数据
        self.configure(
            template_id="entity_extractor",
            name=name,
            responsibility=responsibility,
            entity_focus=list(entity_focus or []),
            relation_focus=list(relation_focus or []),
            priority=priority,
            metadata=dict(metadata or {}),
        )
        self.THRESH = THRESH
        self.step1_sys_desc = (
            "You are a rigorous biomedical type detector. Return STRICT JSON only; "
            "no explanations, prefixes/suffixes, or Markdown."
        )
        self.step2_sys_desc = (
            "You are a rigorous biomedical entity extractor. Return STRICT JSON only; "
            "no explanations, prefixes/suffixes, or Markdown.\n\n"
        )
        self.allKGEntities: List[KGEntity] = []
    
    def build_type_detection_prompt(
        self,
        text: str,
        entity_definitions: Dict[EntityType, EntityDefinition] = ENTITY_DEFINITIONS,
        order: Optional[List[EntityType]] = None,
    ) -> str:
        """
        构造用于“实体类型判定（仅判类型，不抽实体）”的生产级提示词。
        - 闭集 = 由 entity_definitions / order 决定，类型键一律使用 Enum 的 value（你的枚举是小写：'disease','drug',...）
        - 输出必须严格 JSON：
        {
            "present": ["disease","drug", ...],        # 只允许出现在闭集里
            "scores": {"disease": 0.0, "drug": 0.0, ...}  # 闭集每个类型都给分，范围 [0,1]
        }
        - 若无命中：present=[] 且 scores 全 0.0
        """
        # 输出顺序：若未指定则按你的 EntityType 枚举顺序
        if order is None:
            order = list(EntityType)

        # 闭集类型：使用枚举的 value（小写），与你下游统一
        closed_set: List[str] = [et.value for et in order if et in entity_definitions]

        # 详细类型说明：直接用你已有的 formatter
        detailed_defs = format_all_entity_definitions(
            entity_definitions=entity_definitions,
            order=order,
            # labels 不传：保留你定义里的 name（如 "DRUG","DISEASE"...）
        )

        # 组装一个 scores 模板示意（仅作展示，模型仍需返回全量 scores）
        scores_template_items = ", ".join([f'"{t}": 0.0' for t in closed_set])
        scores_template = "{ " + scores_template_items + " }"

        
        task_desc = (
            "Task: Decide which ENTITY TYPES (closed set) appear in the text. "
            "Do NOT list mentions. Do NOT extrapolate or use world knowledge."
        )
        rules = [
            "Base your decision ONLY on the provided text; avoid hallucinations.",
            "If there is explicit, text-grounded evidence (incl. local synonym/abbreviation), mark that type as present.",
            "If no direct evidence, mark as absent.",
            "Output has two parts:",
            "  (1) present: list of type names (subset of the closed set; use lowercase keys, e.g., 'disease').",
            "  (2) scores: confidence in [0,1] for EACH type in the closed set; 0.0 when absent; ≥0.6 when present; ≥0.8 when strongly evident.",
            "If none present: present = [] and all scores = 0.0.",
            "Return STRICT JSON only.",
        ]
        boundary = (
            "Closed-set and detailed definitions (for boundary alignment; do NOT restate in output):\n\n"
            f"{detailed_defs}\n"
        )
        schema = (
            "Output (STRICT JSON):\n"
            "{\n"
            '  "present": ["disease","drug"],\n'
            f'  "scores": {scores_template}\n'
            "}\n"
            f"Closed set (allowed lowercase values only): {closed_set}"
        )
        prompt = (
            f"User:\n{task_desc}\n\n"
            + "\n".join(f"- {r}" for r in rules) + "\n\n"
            f"{boundary}\n"
            f"{schema}\n\n"
            "Text (decide ONLY from this text):\n<<<\n"
            f"{text}\n"
            ">>>\n"
        )

        return prompt
    def validate_and_fix_type_result(self,raw_json_text: str, closed_set: List[str]) -> Dict:
        """
        - 解析 JSON；present 强制为闭集子集；
        - 为闭集中每个类型补全 score；将分数裁剪到 [0,1]；
        - 如果 present 为空但部分分数>0，也不自动加入 present（保持“由模型判定”的语义）。
        """
        data = json.loads(raw_json_text)

        present = data.get("present", [])
        scores = data.get("scores", {})

        # present 只保留闭集合法项
        present = [t for t in present if t in closed_set]

        # 闭集每个类型都有分，且裁剪到 [0,1]
        fixed_scores = {}
        for t in closed_set:
            v = float(scores.get(t, 0.0))
            if v < 0.0: v = 0.0
            if v > 1.0: v = 1.0
            fixed_scores[t] = v

        return {"present": present, "scores": fixed_scores}

    def build_single_type_extraction_prompt(
        self,
        text: str,
        definition: EntityDefinition,
        max_entities: int = 50,
    ) -> str:
        """
         Step-2 prompt: extract ONLY the given entity type from `text`.
        Uses `format_entity_definition(definition, index=1)` to provide the boundary.
        """
        type_key = (definition.name or "").strip().lower()
        boundary = format_entity_definition(definition, index=1)  # e.g., "1. DRUG: ...\n\n   - Examples: ..."

        rules = [
            "Use ONLY the provided text; no world knowledge or extrapolation.",
            "Extract entities of this single type ONLY.",
            "Match local synonyms/abbreviations/case variants when evidenced in text.",
            "Deduplicate (case-insensitive): keep one entry per entity, prefer first occurrence; sort by first offset.",
            "Character spans use 0-based, half-open [start, end) over the raw text (including spaces/newlines).",
            f"Return at most {max_entities} entities.",
            "If none found, return an empty array.",
            "For EACH entity, provide a ~20-word single-sentence description grounded ONLY in the document; if insufficient evidence, use \"N/A\".",
            "The description must not rely on external/world knowledge; it must be inferable from the document."
            "Return STRICT JSON only; no explanations or extra text.",
        ]
        ontoloty_mapping=(
            "ONTOLOGY MAPPING:\n"
            "- Use standard identifiers when known (MESH:D001241, NCBI:5743)\n"
            "- Set \"N/A\" when no standard identifier available\n"
            "- Prioritize well-established ontologies (MeSH, NCBI, UniProt)\n"
        )
        example=(
            "EXAMPLES:\n"
            "Text: \"Aspirin inhibits COX-2 enzyme activity\"\n"
            "Output: [{\"mention\": \"Aspirin\", \"type\": \"DRUG\", \"normalized_id\": \"MESH:D001241\", \"aliases\": [\"acetylsalicylic acid\"], \"description\": \"Drug described as inhibiting COX-2 enzyme activity.\"}, {\"mention\": \"COX-2\", \"type\": \"PROTEIN\", \"normalized_id\": \"NCBI:5743\", \"aliases\": [\"cyclooxygenase-2\", \"PTGS2\"], \"description\": \"Enzyme described as the inhibition target of aspirin.\"}]\n"
        )
        schema = (
            "Output (STRICT JSON):\n"
            "{\n"
            f'  "type": "{boundary}",\n'
            '  "entities": [\n'
            "    {\n"
            '      "mention": "verbatim mention from text",\n'
            '      "span": [start, end],\n'
            '      "confidence": 0.0,\n'
            '      "normalized_id": "ontology:identifier or N/A",\n'
            '      "aliases": ["synonym1", "synonym2"]\n'
            '      "description": "20-word single-sentence description grounded ONLY in the document; if insufficient evidence, use \"N/A\""\n'
            "    }\n"
            "  ]\n"
            "}"
        )


        return (
            "User:\n"
            f"Task: Extract entities of type [{boundary}] only (closed set = this single type). Do not output other types.\n"
            + "\n".join(f"- {r}" for r in rules)
            + "\n\n"
            "Type boundary for disambiguation (DO NOT restate in output):\n"
            f"{ontoloty_mapping}\n"
            f"{example}\n"
            f"{schema}\n\n"
            "Text (extract ONLY from this text):\n<<<\n"
            f"{text}\n"
            ">>>\n"
        )
    def _deduplicate_entities(self, entities: List[KGEntity]) -> List[KGEntity]:
        """
        Remove duplicate entities based on name similarity.

        Args:
            entities: List of entities to deduplicate

        Returns:
            List of unique entities
        """
        if not entities:
            return []

        unique_entities = {}

        for entity in entities:
            # Create a normalized key for comparison
            key = entity.name.lower().strip()

            # If we've seen this entity before, merge information
            if key in unique_entities:
                existing = unique_entities[key]

                # Prefer more specific entity types
                if entity.entity_type != "Unknown" and existing.entity_type == "Unknown":
                    existing.entity_type = entity.entity_type

                # Merge aliases
                if entity.aliases:
                    existing.aliases.extend(entity.aliases)
                    existing.aliases = list(set(existing.aliases))  # Remove duplicates

                # Prefer non-N/A normalized IDs
                if entity.normalized_id != "N/A" and existing.normalized_id == "N/A":
                    existing.normalized_id = entity.normalized_id

            else:
                unique_entities[key] = entity

        return list(unique_entities.values())
    def step1(self, text: str) -> str:
        """
        Step 1: 在候选实体类型中检查存在的实体类型
        """
        
        llm = ChatLLM(system=self.step1_sys_desc)
        step1_prompt = self.build_type_detection_prompt(text=text,entity_definitions=ENTITY_DEFINITIONS,order=list(EntityType))
        response = llm.single(step1_prompt)
        closed_set = [et.value for et in EntityType]  # 小写键集合：['disease','drug',...]
        result = self.validate_and_fix_type_result(raw_json_text=response, closed_set=closed_set)
        selected = [t for t in result["present"] if result["scores"].get(t, 0.0) >= self.THRESH]
        print(selected)
        allowed = {e.value: e for e in EntityType}
        def defs_from_selected(selected, defs=ENTITY_DEFINITIONS):
            return [defs[allowed[t]] for t in selected if t in allowed]
        result=defs_from_selected(selected)
        return result
    
    def step2(self, text: str, type_list: List[EntityDefinition]) -> str:
        """
        Step 2: Classify the entities into the appropriate ontology
        """
        self.logger.info(f"Entity type list: {type_list}")
        llm = ChatLLM(system=self.step2_sys_desc)
        for i in tqdm(range(len(type_list))):
            prompt = self.build_single_type_extraction_prompt(text=text, definition=type_list[i])
            # print(prompt)
            response = llm.single(prompt)
            # self.logger.info(f"{type_list[i].name} Response: {response}")
            try:
                parsed = self.parse_json(response)
            except Exception:
                self.logger.error(f"Failed to parse JSON response {response}")
                parsed = {}

            items = []
            if isinstance(parsed, dict):
                items = parsed.get("entities") or []
            elif isinstance(parsed, list):
                items = parsed
            count=0
            for entity in items:
                if not isinstance(entity, dict):
                    continue
                self.allKGEntities.append(KGEntity(
                    entity_id=entity.get("mention", ""),
                    entity_type=type_list[i].name,
                    name=entity.get("mention", ""),
                    normalized_id=entity.get("normalized_id", "N/A"),
                    description=entity.get("description", "N/A"),  # ← 仅此一行
                    aliases=entity.get("aliases", []) or []
                ))
                count+=1
            self.logger.info(f"{type_list[i].name} Extracted {count} entities")
    def process(self, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        执行实体抽取主流程。

        入参格式示例：[{"id": "doc1", "text": "..."}, ...]
        返回格式示例：[{"doc_id": str, "entities": [{"name": str, "type": str, "span": [start, end]}]}]
        """

        for i,doc in enumerate(tqdm(documents)):
            doc_id = doc.get("id") or ""
            text = doc.get("text") or ""
            graph_id=doc_id+'_'+str(i)
            subgraph = Subgraph(subgraph_id=graph_id, name=doc_id, meta={"text": text})
            type_list = self.step1(text)
            self.step2(text, type_list)
            self.allKGEntities=self._deduplicate_entities(self.allKGEntities)
            subgraph.upsert_many_entities(self.allKGEntities)
            self.memory.register_subgraph(subgraph)
            self.allKGEntities = []        
        return self.allKGEntities

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

        # INSERT_YOUR_CODE
def main():
    # 示例文档列表
    documents = [
        {"id": "doc1", "text": "Aspirin is commonly used to treat fever and pain."},
        {"id": "doc2", "text": "BRCA1 mutations are associated with breast cancer."},
    ]

    # 实例化你的类（假设名为 EntityExtractor，实际请用相应类名替换）
    extractor = EntityExtractionAgent()
    results = extractor.run(documents)
    for result in results:
        print(result)

# 如果直接运行此脚本，则执行 main
if __name__ == "__main__":
    main()

