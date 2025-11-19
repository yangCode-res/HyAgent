import json
import os
from typing import Dict, List, Optional

from networkx import graph_atlas
from openai import OpenAI
from tqdm import tqdm

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory
from Store.index import get_memory
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple

"""
因果关系评估 Agent。
基于已有的文本和三元组，评估三元组中的关系是否为因果关系，并给出置信度评分和支持证据。
输入:无（从内存中的子图获取文本和三元组）
输出:无（将评估结果存储到内存中的子图）
调用入口：agent.process()
"""

class CausalExtractionAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str,memory:Optional[Memory]=None):
        self.system_prompt="""You are a causal relationship evaluation agent. In last term, the Relationship extraction expert has extracted various relationships from text paragraphs. 
        So now your task is to identify whether these causal relationships are supported by evidence and rate them with confidence from given text paragraphs.
        INSTRUCTIONS:
        Based on the provided text and triples, evaluate each relationship for causal validity.
        For each relationship, assign a confidence score between 0 and 1, where the more the score is close to 1, the more likely it is a true causal relationship.

        Additionally, provide the evidence that the relationship is causal from the text with specific quotes or references.
        The evidence could be multiple sentences or clauses from the text that directly support the relationship. So return them as a list.

        Finally, return the results in a JSON array format, where each entry contains the head entity, relation, tail entity, confidence score, and evidence.
        The text will be given in the user prompt along with the relationships to be evaluated with triples in the form of (head, -[relation]->, tail)
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

        NO CONFIDENCE (0.0-0.2):
        - No evidence provided
        - Contradictory information
        - Irrelevant data

        NOTICE:The relationship might not be symmetric, so please evaluate each direction separately, and you should return the confidence of the triple in two directions if necessary.
        And if there exists reverse relationship in the text, please also evaluate it and return it as a separate entry in the JSON array.

        OUTPUT FORMAT:
        Return only valid JSON array:
        [
        {
            "head": "exact_entity_name",
            "relation": "RELATIONSHIP_TYPE",
            "tail": "exact_entity_name",
            "confidence": [forward,reverse],
            "evidence": ["supporting_evidence_texts"]
        }
        ]

        EXAMPLE:
        Text: "Aspirin significantly inhibited COX-2 activity (p<0.001), reducing PGE2 production by 60%."
        Output:
        [
        {"head": "aspirin", "relation": "INHIBITS", "tail": "COX-2", "confidence": [0.95,0.15], "evidence": ["Aspirin significantly inhibited COX-2 activity (p<0.001)"]}
        ]
        """
        super().__init__(client, model_name, self.system_prompt)
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
        
    def process(self,texts:List[Dict[str,str]]): 
        """
        process the causal evaluation for multiple paragraphs
        parameters:
        texts:the paragraphs with their ids to be evaluated
        And the result will be written in the memory store directly.
        """
        for i,text in enumerate(tqdm(texts)):
            self.logger.info(f"CausalExtractionAgent: Processing text {i+1}/{len(texts)}")
            graph_id=text.get("id","")+'_'+str(i)
            max_retries=5
            retries=0
            if not self.memory.get_subgraph(graph_id):
                while not self.memory.get_subgraph(graph_id) and retries < max_retries:
                    self.logger.warning(f"CausalExtractionAgent: Subgraph {graph_id} not found in memory. Retrying...")
                    retries+=1
            if not self.memory.get_subgraph(graph_id):
                self.logger.error(f"CausalExtractionAgent: Subgraph {graph_id} still not found after {max_retries} retries. Skipping...")
                continue
            self.run(text, graph_id)

    def run(self,text:Dict[str,str],graph_id:str):
        """
        run the causal evaluation for a single paragraph
        parameters:
        text:the paragraph with its id to be evaluated
        graph_id:the graph id in the memory store
        output:
        the list filled with elements defined as data structure KGTriple(whose definition could be find in the file KGTriple) 
        """
        plain_text=text.get("text","")
        subgraph=self.memory.get_subgraph(graph_id)
        self.logger.info(f"CausalExtractionAgent: Processing Subgraph {graph_id} with {len(subgraph.relations.all())} relations.")
        extracted_triples=subgraph.get_relations()
        if not extracted_triples:
            self.logger.info(f"CausalExtractionAgent: No relations found in Subgraph {graph_id}. Skipping...")
            return
        triple_str='\n'.join(triple.__str__() for triple in extracted_triples)
        prompt=f"""Evaluate the following relationships for causal validity based on the provided text.
        the text is: '''{plain_text}'''
        the relationships are:'''{triple_str}'''
        Please follow the instructions in the system prompt to assign confidence scores and provide supporting evidence.
        """
        response=self.call_llm(prompt=prompt)
        try:
            causal_evaluations=self.parse_json(response)
            triples=[]
            for eval in causal_evaluations:
                head=eval.get("head","unknown")
                relation=eval.get("relation","unknown")
                tail=eval.get("tail","unknown")
                confidence=eval.get("confidence",[0.0,0.0])
                evidence=eval.get("evidence",[])
                triple=subgraph.relations.find_Triple_by_head_and_tail(head,tail)
                object=triple.object if triple else None
                subject=triple.subject if triple else None
                triples.append(KGTriple(head,relation,tail,confidence,evidence=evidence,mechanism="unknown",source=text.get("id","unknown"),subject=subject,object=object))
            subgraph.relations.reset()
            subgraph.relations.add_many(triples)
            self.memory.register_subgraph(subgraph)
        except Exception as e:
            self.logger.error(f"CausalExtractionAgent: Failed to parse response JSON. Error: {e}")
            self.logger.error(f"Response was: {response}")
            return
