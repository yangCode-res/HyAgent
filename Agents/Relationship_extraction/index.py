import json
from tqdm import tqdm
from Memory.index import Subgraph
from Store.index import get_memory
from typing import Optional, Any, Dict, List
from openai import OpenAI
from Core.Agent import Agent
from Logger.index import get_global_logger
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


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
    "tail": "exact_entity_name"
  }
]

EXAMPLE:
Text: "Aspirin significantly inhibited COX-2 activity (p<0.001), reducing PGE2 production by 60%."
Output:
[
  {"head": "aspirin", "relation": "INHIBITS", "tail": "COX-2", "confidence": 0.95, "evidence": "Aspirin significantly inhibited COX-2 activity (p<0.001)"}
]"""
        self.memory=get_memory()
        super().__init__(client,model_name,self.system_prompt)

    def process(self,texts:List[Dict[str,str]])->Dict[str,List[KGTriple]]:
        """
        process the relationship extraction for multiple paragraphs
        parameters:
        texts:the paragraphs with their ids to be extracted
        causal_types:the causal relationships recognized from the causal_extraction agent
        output:
        the list filled with elements defined as data structure KGTriple(whose definition could be find in the file KGTriple) 
        """
        results={}
        for i,text in enumerate(tqdm(texts)):
            text_id=text.get("id","")
            paragraph=text.get("text","")
            # print(paragraph)
            graph_id=text_id+'_'+str(i)
            causal_types=self.extract_existing_relation(paragraph)
            # print(causal_types)
            extracted_triples=self.extract_relationships(paragraph, text_id, causal_types)
            extracted_triples=self.remove_duplicate_triples(extracted_triples)
            subgraph=self.memory.get_subgraph(graph_id)
            if not subgraph:
                subgraph=Subgraph(graph_id,graph_id,{"text":text})
            subgraph.add_relations(extracted_triples)
            self.memory.register_subgraph(subgraph)
    ###step 1: extract existing relationship types from the text
    def extract_existing_relation(self,text:str)->List[str]:
        prompt=f"""return existing relationship types from provided text and return them as a list. The relationship types are defined as follows:
        1. TREATS
2. INHIBITS
3. ACTIVATES
4. CAUSES
5. ASSOCIATED_WITH
6. REGULATES
7. INCREASES/DECREASES
8. INTERACTS_WITH
OUTPUT FORMAT:
Return only valid array:
[
  "treats", "regulates", "activates", "causes"
]
        Text to analyze:
        {text}
"""
        try:
            response=self.call_llm(prompt)
            # 解析返回的 JSON 数组
            relationships=json.loads(response)
            relationships=list(set(relationships))
            return relationships
        except Exception as e:
            print("Error extracting causal relationships:", e)
            return []
        return []
    ###step 2: extract relationships from the text with provided existing relationship types
    def extract_relationships(self,text:str,text_id:str,causal_types:List[str]) -> List[KGTriple]:
        """
        relationship extraction
        parameters:
        text:the paragraph to be extracted(the function could only settle with one paragraph each time so it might be called for times)
        causal_types:the causal relationships recognized from the causal_extraction agent
        output:
        the list filled with elements defined as data structure KGTriple(whose definition could be find in the file KGTriple) 
        """
        relations='\n'.join(causal_type for causal_type in causal_types)
        prompt = f"""

        From the text below, identify direct relationships between entities.
        Only extract relationships that are explicitly stated or clearly implied in the text.
        Please make sure that the relationships you extract are not in conflict with the provided causal types.
        Text to analyze:
        {text}
        Existing relationship types:
        {relations}
        Return only a JSON array of relationships
        """
        try:
            response=self.call_llm(prompt)
            relations_data=self.parse_json(response)
            triples=[]
            for rel_data in tqdm(relations_data):
                if isinstance(rel_data,dict) and all(key in rel_data for key in ["head","relation","tail"]):
                    head=rel_data.get("head","").strip()
                    tail=rel_data.get("tail","").strip()
                    relation=rel_data.get("relation","").strip()
                    source=text_id
                    triple=KGTriple(
                        head=head,
                        relation=relation,
                        tail=tail,
                        confidence=None,
                        evidence=["unknown"],
                        mechanism="unknown",
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

    
