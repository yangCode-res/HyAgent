import concurrent.futures
from dataclasses import dataclass
from time import sleep
from typing import Dict, List

from openai import OpenAI
from sympy import fu
from transformers import pipeline

from Agents.Alignment_triple.index import AlignmentTripleAgent
from Agents.Causal_extraction.index import CausalExtractionAgent
from Agents.Collaborate_extraction.index import CollaborationExtractionAgent
from Agents.Entity_extraction.index import EntityExtractionAgent
from Agents.Entity_normalize.index import EntityNormalizationAgent
from Agents.Mechanism_extraction.index import MechanismExtractionAgent
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Temporal_extraction.index import TemporalExtractionAgent
from Core.Agent import Agent
from Agents.Fusion_subgraph.index import SubgraphMerger
from Store.index import get_memory


@dataclass
class PipeLine:
    def __init__(self,graph_type:str,user_query:str,client:OpenAI,model_name:str):
        self.graph_type=graph_type
        self.user_query=user_query
        self.client=client
        self.model_name=model_name

    
    def get_pipeline(self)->List[Agent]:
        """
        Construct the pipeline of Agent modules based on the type of knowledge graph requested.
        Fundamental modules:
        1.Entity Extraction Agent
        2.Entity Normalization Agent
        3.Relationship Extraction Agent
        4.Collaboration Extraction Agent
        5.Alignment Triple Agent
        Optional modules (to be added based on graph type):
        - Causal Extraction Agent
        - Mechanism Extraction Agent
        - Temporal Extraction Agent
        Graph types:
        - Basic Knowledge Graph: Only fundamental modules.
        - Causal Knowledge Graph (without mechanism): Fundamental + Causal Extraction Agent.
        - Temporal Knowledge Graph: Fundamental + Temporal Extraction Agent.
        - Causal Knowledge Graph (with mechanism): Fundamental + Causal Extraction Agent + Mechanism
        - Comprehensive Knowledge Graph: All modules.
        The parallel agents will be represented as a list within the pipeline.
        Example:
        [[Entity Extraction Agent,Relationship Extraction Agent], Entity Normalization Agent, Collaboration Extraction Agent, [Causal Extraction Agent, Mechanism Extraction Agent], Alignment Triple Agent]
        Then the pipeline will be structed in Task Scheduler Agent.
        """
        pipeline=[]
        # Fundamental modules
        entity_extraction_agent=EntityExtractionAgent(self.client,self.model_name)
        relationship_extraction_agent=RelationshipExtractionAgent(self.client,self.model_name)
        entity_normalization_agent=EntityNormalizationAgent(self.client,self.model_name)
        collaboration_extraction_agent=CollaborationExtractionAgent(self.client,self.model_name)
        alignment_triple_agent=AlignmentTripleAgent(self.client,self.model_name)
        subgraph_merger=SubgraphMerger(self.client,self.model_name)
        pipeline.append([entity_extraction_agent,relationship_extraction_agent])
        pipeline.append(entity_normalization_agent)
        pipeline.append(collaboration_extraction_agent)
        # Optional modules based on graph type
        optional_pipelines=[]
        if self.graph_type=="Causal Knowledge Graph (without mechanism)" or self.graph_type=="Causal Knowledge Graph (with mechanism)" or self.graph_type=="Comprehensive Knowledge Graph":
            causal_extraction_agent=CausalExtractionAgent(self.client,self.model_name)
            optional_pipelines.append(causal_extraction_agent)
        if self.graph_type=="Temporal Knowledge Graph" or self.graph_type=="Comprehensive Knowledge Graph":
            temporal_extraction_agent=TemporalExtractionAgent(self.client,self.model_name)
            optional_pipelines.append(temporal_extraction_agent)
        if self.graph_type=="Causal Knowledge Graph (with mechanism)" or self.graph_type=="Comprehensive Knowledge Graph":
            mechanism_extraction_agent=MechanismExtractionAgent(self.client,self.model_name)
            optional_pipelines.append(mechanism_extraction_agent)
        if optional_pipelines:
            pipeline.append(optional_pipelines)
        pipeline.append(alignment_triple_agent)
        pipeline.append(subgraph_merger)
        return pipeline
    
    def print_pipeline(self):
        pipeline=self.get_pipeline()
        print("Constructed Agent Pipeline:")
        for step in pipeline:
            if isinstance(step, list):
                agents=[agent.__class__.__name__ for agent in step]
                print(f"Parallel Agents: {agents}")
            else:
                print(f"Agent: {step.__class__.__name__}")

    def run(self):
        """
        [[Entity Extraction Agent,Relationship Extraction Agent], Entity Normalization Agent, Collaboration Extraction Agent, [Causal Extraction Agent, Mechanism Extraction Agent], Alignment Triple Agent]
        """
        pipeline=self.get_pipeline()
        memory=get_memory()
        #让pipeline全部串行测试
        for step in pipeline:
            if isinstance(step, List):
                for agent in step:
                    if hasattr(agent,"process"):
                        try:
                            agent.process()
                        except Exception as e:
                            print("Error in parallel agent execution:", e)
            else:
                try:
                    step.process()
                except Exception as e:
                    print(f"Error executing agent {step.__class__.__name__}: {e}")
        # all_futures=[]
        # for step in pipeline:
        #     if isinstance(step, List):
        #         # Parallel execution
        #         for agent in step:
        #             if hasattr(agent,"process"):
        #                 future=concurrent.futures.ThreadPoolExecutor(max_workers=2).submit(agent.process)
        #             if future:
        #                 all_futures.append(future)
        #         for future in all_futures:
        #             try:
        #                 future.result()
        #             except Exception as e:
        #                 print("Error in parallel agent execution:", e)
        #     else:
        #         try:
        #             step.process()
        #         except Exception as e:
        #             print(f"Error executing agent {step.__class__.__name__}: {e}")
        memory.dump_json("./snapshots")
