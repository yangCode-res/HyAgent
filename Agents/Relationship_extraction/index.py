import concurrent.futures
import json
from typing import Dict, List

from openai import OpenAI
from tqdm import tqdm

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Subgraph
from Store.index import get_memory
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple

"""
关系抽取 Agent。
基于已有的文本，抽取实体之间的关系并形成三元组。
输入:无 (文本从子图工作区的meta属性中获取)
输出:无（将抽取的三元组存储到内存中的子图）
调用入口：agent.process()
"""
class RelationshipExtractionAgent(Agent):
    def __init__(self,client:OpenAI,model_name:str) -> None:
        self.system_prompt="""You are a specialized Relationship Extraction Agent for biomedical knowledge graphs. Your task is to identify precise relationships between biomedical entities.

OBJECTIVE: 
1. Extract the specific **verb predicate** (e.g., "phosphorylates", "binds to").
2. Map this predicate to one of the following **BioLink Categories**:
   - POSITIVE_REGULATE (activates, increases, stimulates, upregulates)
   - NEGATIVE_REGULATE (inhibits, decreases, blocks, downregulates)
   - CAUSES (leads to, results in, triggers)
   - TREATS (cures, alleviates, prevents)
   - INTERACTS (binds, complexes with)
   - ASSOCIATED (correlated with, linked to)


INPUT: Biomedical text containing entity mentions.

EXTRACTION STRATEGY:
1. Identify biological entities.
2. Identify the specific **verb or verb phrase** connecting them.
3. Normalize the verb to ensure consistency (see NORMALIZATION RULES).

NORMALIZATION RULES (Crucial for Graph Alignment):
1. **Lemmatization:** Convert verbs to their base/infinite form.
   - "inhibited" -> "inhibit"
   - "reduces" -> "reduce"
   - "treating" -> "treat"
2. **Active Voice:** Whenever possible, formulate the relationship in active voice.
   - If text says "A is activated by B", extract: Head=B, Predicate="activate", Tail=A.
3. **Particle Inclusion:** Include necessary prepositions that define the interaction.
   - "binds to" (keep "to")
   - "associated with" (keep "with")
4. **Remove Adverbs/Modifiers:** Strip away words that describe intensity or certainty to facilitate matching.
   - "significantly inhibits" -> "inhibit"
   - "potentially causes" -> "cause"
   - "strongly downregulates" -> "downregulate"
5. **Atomic Predicates:** If multiple verbs are used, split them into separate records.
   - "binds and inhibits" -> Create two entries: one for "bind", one for "inhibit".

QUALITY CONTROLS:
- **Explicit Only:** Do not infer relationships not stated in the text.
- **No Negation:** Ignore relationships that are explicitly negated (e.g., "does not cause").
- **Precision:** The predicate must scientifically describe the interaction mechanism (e.g., prefer "phosphorylate" over "affect").

OUTPUT FORMAT (JSON):
[
  {
    "head": "Entity A",
    "relation": "verb_from_text (normalized)",
    "relation_type": "BIOLINK_CATEGORY",
    "tail": "Entity B",
    "evidence": "..."
  }
]

EXAMPLE:
Text: "Metformin significantly phosphorylates AMPK, thereby activating the pathway."
Output:
[
  {
    "head": "Metformin", 
    "relation": "phosphorylate", 
    "relation_type": "POSITIVE_REGULATE", 
    "tail": "AMPK"
  }
]
"""
        self.memory=get_memory()
        super().__init__(client,model_name,self.system_prompt)

    def process(self)->None:
        """
        process the relationship extraction for multiple paragraphs
        parameters:
        texts:the paragraphs with their ids to be extracted
        causal_types:the causal relationships recognized from the causal_extraction agent
        output:
        Store extracted triples back to memory subgraphs. 
        """
        subgraphs=self.memory.subgraphs
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures=[]
            for subgraph_id,subgraph in tqdm(subgraphs.items()):
                if subgraph:
                    futures.append(executor.submit(self.process_subgraph,subgraph))
            
            for future in tqdm(futures,desc="Processing relationship extraction"):
                try:
                    future.result()
                except Exception as e:
                    logger=get_global_logger()
                    logger.info(f"Relationship extraction failed in concurrent processing: {str(e)}")
    
    def process_subgraph(self,subgraph:Subgraph):
        subgraph_id=subgraph.id # type: ignore
        paragraph=subgraph.meta.get("text","")
        causal_types=self.extract_existing_relation(paragraph)
        # print(causal_types)
        extracted_triples=[]
        for causal_type in causal_types:
            triples=self.extract_relationships(paragraph,subgraph_id,causal_type)
            for triple in triples:
                extracted_triples.append(triple)
        extracted_triples=self.remove_duplicate_triples(extracted_triples)
        subgraph=self.memory.get_subgraph(subgraph_id) # type: ignore
        if not subgraph:
            subgraph=Subgraph(subgraph_id,subgraph_id,{"text":paragraph})
        subgraph.add_relations(extracted_triples)
        self.memory.register_subgraph(subgraph)

    ###step 1: extract existing relationship types from the text
    def extract_existing_relation(self,text:str)->List[str]:
        prompt=f"""return relationship types from provided text and return them as a list (strings only).
        Text to analyze:
        {text}
   - POSITIVE_REGULATE (activates, increases, stimulates, upregulates)
   - NEGATIVE_REGULATE (inhibits, decreases, blocks, downregulates)
   - CAUSES (leads to, results in, triggers)
   - TREATS (cures, alleviates, prevents)
   - INTERACTS (binds, complexes with)
   - ASSOCIATED (correlated with, linked to)
Example output:
[
  "positive_regulate", "negative_regulate", "causes", "treats", "interacts", "associated"
]
"""
        try:
            response=self.call_llm(prompt)
            relationships=self.parse_json(response)
            if not isinstance(relationships,list):
                return []
            # normalize strings and keep order while deduping
            seen=set()
            deduped: List[str]=[]
            for rel in relationships:
                if isinstance(rel,str):
                    norm=self._normalize_relation_type(rel)
                    if norm and norm not in seen:
                        seen.add(norm)
                        deduped.append(norm)
            return deduped
        except Exception as e:
            print("Error extracting causal relationships:", e)
            return []

    ###step 2: extract relationships from the text with provided existing relationship types
    def extract_relationships(self,text:str,subgraph_id,causal_type:str) -> List[KGTriple]:
        """
        relationship extraction
        parameters:
        text:the paragraph to be extracted(the function could only settle with one paragraph each time so it might be called for times)
        causal_type:the causal relationship recognized from the causal_extraction agent
        output:
        the list filled with elements defined as data structure KGTriple(whose definition could be find in the file KGTriple) 
        """
        relations=causal_type
        prompt = f"""

        From the text below, identify direct relationships between entities.
        Only extract relationships that are explicitly stated or clearly implied in the text.
        Please make sure that the relationships you extract are not in conflict with the provided causal types.
        Text to analyze:
        {text}
        Existing relationship type:
        {relations}
        Return only a JSON array of relationships
        """
        try:
            response=self.call_llm(prompt)
            relations_data=self.parse_json(response)
            triples: List[KGTriple]=[]
            if not isinstance(relations_data,list):
                relations_data=[]
            for rel_data in tqdm(relations_data):
                if not isinstance(rel_data,dict):
                    continue
                head=(rel_data.get("head") or "").strip()
                tail=(rel_data.get("tail") or "").strip()
                relation=(rel_data.get("relation") or "").strip()
                if not head or not tail or not relation:
                    continue
                relation_type=(rel_data.get("relation_type") or "").strip()
                if relation_type:
                    relation_type=self._normalize_relation_type(relation_type)
                # fallback to provided causal_type if model omitted/unknown
                if not relation_type:
                    relation_type=self._normalize_relation_type(causal_type)
                triple=KGTriple(
                    head=head,
                    relation=relation,
                    relation_type=relation_type,
                    tail=tail,
                    confidence=None,
                    evidence=["unknown"],
                    mechanism="unknown",
                    source=subgraph_id
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
            triple_key=(triple.head, triple.relation, triple.tail, triple.relation_type)
            best=unique_triple.get(triple_key)
            if best is None:
                unique_triple[triple_key]=triple
            else:
                # Compare confidence lists by their max value when available
                def max_conf(t: KGTriple) -> float:
                    if t.confidence and isinstance(t.confidence, list) and len(t.confidence)>0:
                        try:
                            return float(max(t.confidence))
                        except Exception:
                            return -1.0
                    return -1.0
                if max_conf(triple) > max_conf(best):
                    unique_triple[triple_key]=triple
        return list(unique_triple.values())

    def _normalize_relation_type(self, rel: str) -> str:
        """Normalize relation type strings to canonical set.
        Supported: POSITIVE_REGULATE, NEGATIVE_REGULATE, CAUSES, TREATS, INTERACTS, ASSOCIATED
        Accepts common lowercase or synonym variants.
        """
        if not isinstance(rel, str):
            return ""
        r=rel.strip().lower()
        mapping = {
            "positive_regulate":"POSITIVE_REGULATE",
            "pos_regulate":"POSITIVE_REGULATE",
            "activate":"POSITIVE_REGULATE",
            "activates":"POSITIVE_REGULATE",
            "increase":"POSITIVE_REGULATE",
            "increases":"POSITIVE_REGULATE",
            "upregulate":"POSITIVE_REGULATE",
            "upregulates":"POSITIVE_REGULATE",

            "negative_regulate":"NEGATIVE_REGULATE",
            "neg_regulate":"NEGATIVE_REGULATE",
            "inhibit":"NEGATIVE_REGULATE",
            "inhibits":"NEGATIVE_REGULATE",
            "decrease":"NEGATIVE_REGULATE",
            "decreases":"NEGATIVE_REGULATE",
            "downregulate":"NEGATIVE_REGULATE",
            "downregulates":"NEGATIVE_REGULATE",

            "causes":"CAUSES",
            "cause":"CAUSES",
            "result in":"CAUSES",
            "results in":"CAUSES",
            "lead to":"CAUSES",
            "leads to":"CAUSES",
            "trigger":"CAUSES",
            "triggers":"CAUSES",

            "treats":"TREATS",
            "treat":"TREATS",
            "cures":"TREATS",
            "cure":"TREATS",
            "alleviates":"TREATS",
            "alleviate":"TREATS",
            "prevents":"TREATS",
            "prevent":"TREATS",

            "interacts":"INTERACTS",
            "interact":"INTERACTS",
            "bind":"INTERACTS",
            "binds":"INTERACTS",
            "complexes with":"INTERACTS",

            "associated":"ASSOCIATED",
            "associate":"ASSOCIATED",
            "associated with":"ASSOCIATED",
            "correlated":"ASSOCIATED",
            "correlated with":"ASSOCIATED",
            "linked":"ASSOCIATED",
            "linked to":"ASSOCIATED",
        }
        if r in mapping:
            return mapping[r]
        # also accept already canonical values
        if r in {"positive_regulate","negative_regulate","causes","treats","interacts","associated"}:
            return mapping.get(r, r.upper())
        if r.upper() in {"POSITIVE_REGULATE","NEGATIVE_REGULATE","CAUSES","TREATS","INTERACTS","ASSOCIATED"}:
            return r.upper()
        return r.upper()

    
