import os
from calendar import c
from html import entities
from typing import Optional, Any, Dict, List

from openai import OpenAI
from Core.Agent import Agent
from HyAgent.Logger.index import get_global_logger
from HyAgent.TypeDefinitinons.EntityTypeDefinitions.index import KGEntity
from HyAgent.TypeDefinitinons.TripleDefinitions.KGTriple import KGTriple


class RelationshipExtractionAgent(Agent):
    def __init__(self,client:OpenAI,model_name:str) -> None:
        self.system_prompt="""You are a specialized Relationship Extraction Agent for biomedical knowledge graphs. Your task is to identify precise relationships between biomedical entities with high accuracy and appropriate confidence scoring.

OBJECTIVE: Extract explicit relationships between biomedical entities from text, focusing on scientifically validated interactions and associations.

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
Return only valid JSON array:
[
  {
    "head": "exact_entity_name",
    "relation": "RELATIONSHIP_TYPE",
    "tail": "exact_entity_name",
    "confidence": 0.85,
    "evidence": "direct quote supporting relationship",
    "temporal_info": "the temporal_info of the relationship(if any)",
    "mechanism": "To describe the mechanism of the cause(50-100 words)"
  }
]

EXAMPLE:
Text: "Aspirin significantly inhibited COX-2 activity (p<0.001), reducing PGE2 production by 60%."
Entities: ["aspirin", "COX-2", "PGE2"]
Output:
[
  {"head": "aspirin", "relation": "INHIBITS", "tail": "COX-2", "confidence": 0.95, "temporal_info":None,"evidence": "Aspirin significantly inhibited COX-2 activity (p<0.001),mechanism: Aspirin exerts its inhibitory effect on COX-2 by acetylating a serine residue in the enzyme's active site, which prevents the conversion of arachidonic acid to prostaglandins. This reduction in prostaglandin synthesis leads to decreased inflammation and pain. The significant p-value (p<0.001) indicates strong statistical support for this effect."}
]"""
        super().__init__(client,model_name,self.system_prompt)
        
    def process(self,texts,entities:List[KGEntity],causal_types:List[str])->List[KGTriple]:
        if not entities:
            return []
        triples=[]
        entities_name=[]
        for entity in entities:
            entities_name.append(entity.name)
        for text in texts:
            extracted_triples=self.extract_relationships(text,entities_name,causal_types)
            triples.append(extracted_triples) 
        triples=self.remove_duplicate_triples(triples)
        return triples

    def extract_relationships(self, text,entities:List[str],causal_types:List[str]) -> List[KGTriple]:
        """
        relationship extraction
        parameters:
        text:the paragraph to be extracted(the function could only settle with one paragraph each time so it might be called for times)
        entities:the string list of entities recognized from the entity_extraction agent
        causal_types:the causal relationships recognized from the causal_extraction agent
        output:
        the list filled with elements defined as data structure KGTriple(whose definition could be find in the file KGTriple) 
        """
        
        if len(entities)<2:
            return []
        entity_bullets = "\n".join(f"- {entity}" for entity in entities)
        causal_types_str="\n".join(f"-{causal_type} "for causal_type in causal_types)
        prompt = f"""
        Entities of interest:
        {entity_bullets}
        Causal types:
        {causal_types_str}

        From the text below, identify direct relationships between these entities.
        Only extract relationships that are explicitly stated or clearly implied in the text.

        Text to analyze:
        {text}

        Return only a JSON array of relationships:
        """
        try:
            response=self.call_llm(prompt)
            relations_data=self.parse_json(response)
            triples=[]
            for rel_data in relations_data:
                if isinstance(rel_data,dict) and all(key in rel_data for key in ["head","relation","tail"]):
                    head=rel_data.get("head","").strip()
                    tail=rel_data.get("tail","").strip()
                    if self.entities_exist(head,entities) and self.entities_exist(tail,entities):
                        confidence=float(rel_data.get("confidence",0.5))
                        evidence=rel_data.get("evidence","")
                        mechanism=rel_data.get("mechanism","")
                        temporal=rel_data.get("temporal","")
                        relation=rel_data.get("relation","").strip()
                        source="Relationship_extraction"
                        triple=KGTriple(
                            head=head,
                            relation=relation,
                            tail=tail,
                            confidence=confidence,
                            evidence=evidence,
                            mechanism=mechanism,
                            temporal_info=temporal,
                            source=source
                        )
                        triples.append(triple)
        except Exception as e:
            logger=get_global_logger()
            logger.info(f"Relationship extraction failed{str(e)}")
            return []
        return triples
    
    def entities_exist(self,entity_name:str,entities:List[str])->bool:
         """
         查看实体是否在实体识别的结果中
         """
         entity_lower=entity_name.lower()
         return any(entity.lower()==entity_lower for entity in entities)

    def remove_duplicate_triples(self,triples:List[KGTriple])->List[KGTriple]:
        """
        remove duplicate triples(might with different information like confidence)
        we remove the triples with lower confidence and preserve the higher ones
        """
        unique_triple={}
        for triple in triples:
            triple_key=triple.__str__
            if triple_key not in unique_triple or triple.confidence>unique_triple[triple_key].confidence:
                unique_triple[triple_key]=triple
        return list(unique_triple.values())

if __name__=="__main__":
   open_ai_api=os.environ.get("OPEN_AI_KEY")
   open_ai_url=os.environ.get("OPEN_AI_URL")
   client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)