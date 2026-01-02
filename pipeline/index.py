import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from sympy import evaluate

from Agents.Alignment_triple.index import AlignmentTripleAgent
from Agents.Causal_extraction.index import CausalExtractionAgent
from Agents.Collaborate_extraction.index import CollaborationExtractionAgent
from Agents.Entity_extraction.index import EntityExtractionAgent
from Agents.Entity_normalize.index import EntityNormalizationAgent
from Agents.Fusion_subgraph.index import SubgraphMerger
from Agents.Hypotheses_Edit.index import HypothesisEditAgent
from Agents.HypothesisGenerationAgent.index import HypothesisGenerationAgent
from Agents.KeywordEntitySearchAgent.index import KeywordEntitySearchAgent
from Agents.Path_extraction.penalty import PathExtractionAgent
from Agents.Query_clarify.index import QueryClarifyAgent
from Agents.ReflectionAgent.index import ReflectionAgent
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Review_fetcher.index import ReviewFetcherAgent
from Logger.index import get_global_logger
from Memory.index import Memory, load_memory_from_json
from Store.index import get_memory

load_dotenv()
class Pipeline:
    def __init__(self,user_query:str,client:OpenAI,model_name:str,memory:Optional[Memory]=None):
        self.user_query=user_query
        self.core_entities=[]
        self.client=client
        self.model_name=model_name
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
        self.reason_model="deepseek-reasoner"
        self.clarified_query=""
        self.scores={}
    def get_pipeline(self):
        pipeline=[]
        pipeline.append(EntityExtractionAgent(self.client,self.model_name))
        pipeline.append(EntityNormalizationAgent(self.client,self.model_name))
        pipeline.append(RelationshipExtractionAgent(self.client,self.model_name))
        pipeline.append(CollaborationExtractionAgent(self.client,self.model_name))
        pipeline.append(CausalExtractionAgent(self.client,self.model_name))
        pipeline.append(AlignmentTripleAgent(self.client,self.model_name))
        pipeline.append(SubgraphMerger(self.client,self.model_name))
        pipeline.append(KeywordEntitySearchAgent(self.client,self.model_name,keywords=self.core_entities))
        pipeline.append(PathExtractionAgent(self.client,self.model_name,query=self.clarified_query))
        pipeline.append(HypothesisGenerationAgent(self.client,self.model_name,query=self.clarified_query,max_paths=5,hypotheses_per_path=3))
        pipeline.append(ReflectionAgent(self.client,self.model_name))
        pipeline.append(HypothesisEditAgent(client=self.client,model_name=self.model_name,query=self.clarified_query))
        return pipeline
    def get_goOn(self,memory:Memory):
        pipeline=[]
        pipeline.append(KeywordEntitySearchAgent(self.client,self.model_name,keywords=self.core_entities,memory=memory))
        pipeline.append(PathExtractionAgent(client=self.client, model_name=self.model_name,k=20,memory=memory,query=self.clarified_query))
        pipeline.append(HypothesisGenerationAgent(self.client,self.model_name,query=self.clarified_query,max_paths=5,hypotheses_per_path=3,memory=memory))
        pipeline.append(ReflectionAgent(self.client,self.model_name,memory=memory))
        pipeline.append(HypothesisEditAgent(client=self.client,model_name=self.model_name,query=self.clarified_query,memory=memory))
        return pipeline
    def run(self):
        memory=load_memory_from_json('/home/nas2/path/yangmingjian/code/hygraph/snapshots/memory-20251221-184939.json')
        user_query=self.user_query
        queryclarifyagent = QueryClarifyAgent(self.client, self.model_name) # type: ignore
        response = queryclarifyagent.process(user_query)
        clarified_query = response.get("clarified_question", user_query) # type: ignore
        self.clarified_query=clarified_query
        core_entities= response.get("core_entities", []) # type: ignore
        print(f"Core Entities: {core_entities}")
        intention= response.get("main_intention", "") # type: ignore
        print("intention=>",intention)
        print("clarified_query=>",clarified_query)
        reviewfetcheragent = ReviewFetcherAgent(self.client, self.model_name) # type: ignore
        reviewfetcheragent.process(clarified_query)

        self.core_entities=core_entities
        self.intention=intention
        self.pipeline=self.get_pipeline()
        for agent in self.pipeline:
            print(f"Running agent: {agent.__class__.__name__}")
            agent.process()
            self.memory.dump_json("./snapshots")
        evaluateAgent=ReflectionAgent(client=self.client,model_name=self.reason_model)
        scores=evaluateAgent.get_scores_only()
        self.scores=scores
        open("scores.json","w").write(json.dumps(self.scores,indent=4)) 