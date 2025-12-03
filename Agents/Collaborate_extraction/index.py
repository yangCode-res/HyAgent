import concurrent
import concurrent.futures
from os import link
from re import sub
from sys import executable
from typing import List, Optional

from fuzzywuzzy import fuzz
from numpy import tri
from openai import OpenAI
from sympy import false

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, Subgraph
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple

"""
协同抽取 Agent。
结合实体抽取和关系抽取两个 Agent 的能力，协同优化实体和关系的抽取结果,并将关系的实体链接到对应实体。
输入: 无（从内存中获取子图的文本）
输出: 无（将优化后的实体和关系存储回内存的子图）
调用入口：agent.process()
"""

class CollaborationExtractionAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str,memory:Optional[Memory]=None):
        self.system_prompt=""""""""
        super().__init__(client,model_name,self.system_prompt)
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
    
    def process(self):
        subgraphs=self.memory.subgraphs
        futures=[]
        with concurrent.futures.ThreadPoolExecutor() as executor: 
            for subgraph_id,subgraph in subgraphs.items():
                if not subgraph:
                    continue
                futures.append(executor.submit(self.process_subgraph,subgraph))
        concurrent.futures.wait(futures)
        
    
    def process_subgraph(self,subgraph:Subgraph):
        if subgraph.entities.all()==[]:
            self.logger.info(f"CollaborationExtractionAgent: Subgraph {subgraph.id} has no entities, skipping.")
            return
        if subgraph.get_relations()==[]:
            self.logger.info(f"CollaborationExtractionAgent: Subgraph {subgraph.id} has no relationships, skipping.")
            return
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_entity=executor.submit(self.entity_extraction,subgraph)
            future_relationship=executor.submit(self.relationship_extraction,subgraph)
            concurrent.futures.wait([future_entity])
            concurrent.futures.wait([future_relationship])
        extracted_entities=future_entity.result()
        extracted_relationships=future_relationship.result()
        subgraph.entities.update(extracted_entities)
        subgraph.relations.reset()
        subgraph.relations.add_many(extracted_relationships)
        self.entity_relation_linking(subgraph)
        self.memory.register_subgraph(subgraph)

    def entity_extraction(self,subgraph)->List[KGEntity]:
        entities=subgraph.entities.all()
        relations=subgraph.get_relations()
        entities_str="\n".join("id:"+entity.entity_id+",name:"+entity.name+",type:"+entity.entity_type for entity in entities)#需要根据id确定更改哪个实体
        relations_str="\n".join(relation.__str__() for relation in relations)
        text=subgraph.meta.get("text","")
        prompt="""
        You are an entity extraction expert.The followings are the entities and relationships extracted from another entity extraction and relationship extraction agents.
        Based on their results,please adjust and refine the entities to ensure accuracy and coherence.
        INSTRUCTIONS:
        1. Review the provided entities carefully according to given relationships.
        2. Ensure that each entity correctly reflects the information in the relationships especially the situation that less words but not alias(such as "abdominal obesity" vs "obesity").
        3. Based on the relationships, modify any entities that seem extremely inconsistent or incorrect, if there exists the case that the name of the entity from entities is the alias of the one of relationships, please NOT adjust it and keep the origin entity name.
        4. If the entity has no relationships with others, just keep it unchanged.
        5. You are allowed to change the name and type of the entity, but DO NOT change the id of the entity.
        OUTPUT FORMAT:
        Return only valid JSON array:
        [
        {
            "id":"entity_id",
            "name": "exact_entity_name",
            "type":"ENTITY_TYPE",
        }
        ]
        EXAMPLE:
        Entities:
        id:1,name:obesity,type:Disease

        Relationships:
        abdominal obesity(-ASSOCIATED_WITH>)T2DM

        You could find out that the head entity name is not exactly matched with entity extraction results,so you need to modify it as follows:
        [
          {"id":"1","name": "abdominal obesity", "type":"Disease"}
        ]"""
        prompt+=f"""
        Now please adjust the entities based on the relationships and resource paragraph below:
        Entities:
        {entities_str}
        Relationships:
        {relations_str}
        Paragraph:
        {text}
        NOTE:
        if there is no conflict between entities and relationships, just return the entities in given output format directly.
        DO NOT CHANGE THE ID OF ENTITIES.
        """
        response=self.call_llm(prompt)
        results=self.parse_json(response)
        entities=[]
        for item in results:
            entity_id=item.get("id","unknown")
            if subgraph.entities.by_id[entity_id]:
                entity=subgraph.entities.by_id[entity_id]
                entity.name=item.get("name","unknown")
                entity.type=item.get("type","unknown")
                entities.append(entity)
        return entities
        # except Exception as e:
        #     logger=f"CollaborationExtractionAgent: Entity extraction failed{str(e)}"
        #     print(logger)
        #     self.logger.error(logger)
        #     return []

    def relationship_extraction(self,subgraph)->List[KGTriple]:
        entities=subgraph.entities.all()
        relations=subgraph.get_relations()
        entities_str="\n".join(entity.__str__() for entity in entities).strip()
        relations_str="\n".join(relation.__str__() for relation in relations).strip()
        text=subgraph.meta.get("text","")
        prompt="""You are a relation extraction expert.The followings are the entities and relationships extracted from another entity extraction and relationship extraction agents.
        Based on their results,please adjust and refine the relationships between entities to ensure accuracy and coherence.
        INSTRUCTIONS:
        1. Review the provided relationships carefully according to given entities.
        2. Ensure that each relationship correctly reflects the interactions between the entities.
        3. Modify any relationships that seem inconsistent or incorrect based on the entity information.

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
        Entities:
        T2DM(Disease),cardiovascular disease(Disease)
        Relationships:
        Type 2 diabetes mellitus(-ASSOCIATED_WITH>)cardiovascular disease
        You could find out that the head entity name is not exactly matched with entity extraction results,so you need to modify it as follows:
        [
          {"head": "T2DM", "relation": "ASSOCIATED_WITH", "tail": "cardiovascular disease"}
        ]"""
        prompt+=f"""
        Now please adjust the relationships based on the entities and resource paragraph below:
        Entities:
        {entities_str}
        Relationships:
        {relations_str}
        Paragraph:
        {text}
        NOTE:
        if there is no conflict between entities and relationships, just return the relationships in given output format directly.
        """
        response=self.call_llm(prompt)
        try:
            results=self.parse_json(response)
            triples=[]
            for item in results:
                triple=KGTriple(
                    head=item.get("head",""),
                    relation=item.get("relation",""),
                    tail=item.get("tail",""),
                    confidence=None,
                    evidence=None,
                    mechanism=None,
                    source="unknown"
                )
                triples.append(triple)
            return triples
        except Exception as e:
            logger=f"CollaborationExtractionAgent: Relationship extraction failed{str(e)}"
            print(logger)
            self.logger.info(logger)
            return [] 
    
    def entity_relation_linking(self,subgraph)->Subgraph:
        entities=subgraph.entities.all()
        relations=subgraph.get_relations()
        for i,entity in enumerate(entities):
            entity_name=entity.name
            if subgraph.relations.by_head.get(entity_name) or subgraph.relations.by_tail.get(entity_name):
                for relation in relations:
                    if relation.head==entity_name or fuzz.partial_ratio(relation.head,entity_name)>90:
                        relation.subject=entity
                    if relation.tail==entity_name or fuzz.partial_ratio(relation.tail,entity_name)>90:
                        relation.object=entity
        unlinked_relations=[relation for relation in relations if not relation.subject or not relation.object]
        if unlinked_relations:
            self.logger.info(f"CollaborationExtractionAgent: Found {len(unlinked_relations)} unlinked relations in Subgraph {subgraph.id}.")
            head_unlinked=[relation for relation in unlinked_relations if not relation.subject and relation.object]
            tail_unlinked=[relation for relation in unlinked_relations if not relation.object and relation.subject]
            entities_name="\n".join(f"- {entity.name} (ID: {entity.entity_id}, Type: {entity.entity_type})" for entity in entities)
            head_unlinked_str="\n".join(relation.__str__() for relation in head_unlinked)
            tail_unlinked_str="\n".join(relation.__str__() for relation in tail_unlinked)
            prompt=f"""You are an expert in linking entities and relationships.
            Your task is to accurately link the entities to the relationships based on their names.
            Here are the entities and relationships:
            Entities:
            {entities_name}
            Head unlinked relationships:
            {head_unlinked_str}
            Tail unlinked relationships:
            {tail_unlinked_str}
            INSTRUCTIONS:
            1. For each unlinked relationship, find the best matching entity based on the name
            2. If a relationship's head or tail matches an entity name closely, link them together
            3. If no suitable entity is found for a relationship, leave it unlinked
            4. If the head is linked but the tail is not, only try to link the tail, and vice versa
            5. Please ensure all the entities to be linked, or find the most suitable one
            6. Use the entity IDs for linking
            OUTPUT FORMAT:
            Return only valid JSON array:
            [
            {{
                "head": "relationship_head_name",
                "relation": "RELATIONSHIP_TYPE",
                "tail": "relationship_tail_name",
                "head_id": "linked_entity_id_or_unknown",
                "tail_id": "linked_entity_id_or_unknown"
            }}
            ]
            NOTICE:PLEASE MAKE SURE ALL SUBJECTS AND OBJECTS TO BE LINKED
"""     
            response=self.call_llm(prompt)
            try:
                results=self.parse_json(response)
                for item in results:
                    head=item.get("head","")
                    tail=item.get("tail","")
                    subject_id=item.get("head_id","unknown")
                    object_id=item.get("tail_id","unknown")
                    for relation in unlinked_relations:
                        if relation.head==head and relation.tail==tail:
                            if subject_id!="unknown" and subgraph.entities.by_id.get(subject_id):
                                relation.subject=subgraph.entities.by_id[subject_id]
                            if object_id!="unknown" and subgraph.entities.by_id.get(object_id):
                                relation.object=subgraph.entities.by_id[object_id]
            except Exception as e:
                logger=f"CollaborationExtractionAgent: Entity-relation linking failed{str(e)}"
                print(logger)
                self.logger.info(logger)
        return subgraph
        

