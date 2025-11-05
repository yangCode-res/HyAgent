import json
import os
from typing import List
from openai import OpenAI
from Core.Agent import Agent

class CausalExtractionAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str):
        self.system_prompt="""You are a specialized Causal relationship Extraction Agent for biomedical knowledge graphs. Your task is to identify precise causal relationships appearing in given paragraphs with high accuracy and appropriate confidence scoring.
        You are required to return existing relationship types from provided text and return them as a list. The relationship types are defined as follows:
        RELATIONSHIP TYPES:
1. TREATS: Drug/intervention treats disease/condition
   - Examples: "aspirin treats headache", "chemotherapy treats cancer"
   - Indicators: treats, cures, alleviates, therapeutic for

2. INHIBITS: Entity blocks or reduces activity/function
   - Examples: "aspirin inhibits COX-2", "statins inhibit cholesterol synthesis"
   - Indicators: inhibits, blocks, suppresses, reduces activity

3. ACTIVATES: Entity stimulates or increases activity/function
   - Examples: "insulin activates glucose uptake", "growth factors activate cell division"
   - Indicators: activates, stimulates, enhances, upregulates

4. CAUSES: Entity directly causes condition/effect
   - Examples: "smoking causes lung cancer", "mutations cause disease"
   - Indicators: causes, leads to, results in, triggers

5. ASSOCIATED_WITH: Statistical or observational association
   - Examples: "obesity associated with diabetes", "gene variants associated with risk"
   - Indicators: associated with, correlated with, linked to, related to

6. REGULATES: Entity controls expression/activity of another
   - Examples: "transcription factors regulate gene expression"
   - Indicators: regulates, controls, modulates, governs

7. INCREASES/DECREASES: Entity raises/lowers levels or activity
   - Examples: "exercise increases insulin sensitivity", "age decreases bone density"
   - Indicators: increases, raises, decreases, reduces, elevates

8. INTERACTS_WITH: Direct molecular interaction
   - Examples: "protein A interacts with protein B", "drug binds receptor"
   - Indicators: interacts with, binds to, complexes with

   EXTRACTION CRITERIA:
REQUIRED CONDITIONS:
- Both entities must be explicitly mentioned in text
- Relationship must be explicitly stated or clearly implied
- Evidence must be present within the analyzed text segment
- Relationship direction must be determinable from context

TEMPORAL ASPECTS:
- Note any temporal information indicating when the relationship occurs
- Capture if the relationship is transient or permanent
- Include timing phrases (e.g., "after treatment", "over time")
- Include the frequency if mentioned (e.g., "daily", "weekly")
CONFIDENCE SCORING:
HIGH CONFIDENCE (0.8-1.0):
- Direct experimental evidence stated
- Quantitative measurements provided
- Established scientific facts
- Clear causal language

MODERATE CONFIDENCE (0.5-0.7):
- Observational evidence
- Statistical associations
- Literature citations mentioned
- Qualified statements (may, appears to, suggests)

LOW CONFIDENCE (0.3-0.4):
- Preliminary findings
- Hypothetical relationships
- Weak associations
- Speculative statements

QUALITY CONTROLS:
- Ignore negated relationships ("does not treat", "no association")
- Avoid circular relationships (A-B, B-A unless distinct)
- Prioritize specific over general relationships
- Ensure entity names match exactly from entity extraction
- Include supporting evidence text

OUTPUT FORMAT:
Return only valid array:
[
  "treats", "regulates", "activates", "causes"
]

"""
        super().__init__(client, model_name, self.system_prompt)
    def extract_causal_relationships(self,texts:List[str])->List[str]:
        """
        提取文本中的因果关系三元组
        """
        prompt=f"""
        Identify and extract causal relationships from the following biomedical text. Use the defined relationship types and criteria to ensure accuracy.

        {self.system_prompt}

        Provide the output strictly as a JSON array of relationship types found in the text.

        Text to analyze:
        {texts}
        """
        try:
            response=self.call_llm(prompt)
            # 解析返回的 JSON 数组
            relationships=json.loads(response)
            relationships=self.remove_duplicates(relationships)
            return relationships
        except Exception as e:
            print("Error extracting causal relationships:", e)
            return []
    
    def remove_duplicates(self,relationships:List[str])->List[str]:
        """移除重复的关系"""
        return list(set(relationships)) 