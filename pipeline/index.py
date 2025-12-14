from dotenv import load_dotenv
from openai import OpenAI
import os
import json
from sympy import evaluate
from Agents.Entity_extraction.index import EntityExtractionAgent
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Entity_normalize.index import EntityNormalizationAgent
from Agents.Collaborate_extraction.index import CollaborationExtractionAgent
from Agents.Alignment_triple.index import AlignmentTripleAgent
from Agents.Fusion_subgraph.index import SubgraphMerger
from Agents.ReflectionAgent.index import ReflectionAgent
from Agents.HypothesisGenerationAgent.index import HypothesisGenerationAgent
from Agents.Causal_extraction.index import CausalExtractionAgent
from Agents.KeywordEntitySearchAgent.index import KeywordEntitySearchAgent
from Agents.Path_extraction.index import PathExtractionAgent
from Agents.Hypotheses_Edit.index import HypothesisEditAgent
from Store.index import get_memory
from Logger.index import get_global_logger
from Memory.index import Memory
from typing import Optional
load_dotenv()
class Pipeline:
    def __init__(self,user_query:str,client:OpenAI,model_name:str,memory:Optional[Memory]=None):
        super().__init__(client=client,model_name=model_name)
        self.user_query=user_query
        self.client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),base_url=os.environ.get("OPENAI_API_BASE_URL"))
        self.model_name=os.environ.get("OPENAI_MODEL")
        self.core_entities=[]
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
        self.reason_model="deepseek-reasoner"
        self.clarified_query=""
    def get_pipeline(self):
        pipeline=[]
        pipeline.append(EntityExtractionAgent(self.client,self.model_name))
        pipeline.append(EntityNormalizationAgent(self.client,self.model_name))
        pipeline.append(RelationshipExtractionAgent(self.client,self.model_name))
        pipeline.append(CollaborationExtractionAgent(self.client,self.model_name))
        pipeline.append(CausalExtractionAgent(self.client,self.model_name))
        pipeline.append(AlignmentTripleAgent(self.client,self.model_name))
        pipeline.append(SubgraphMerger(self.client,self.model_name))
        pipeline.append(KeywordEntitySearchAgent(self.client,self.reason_model,keywords=self.core_entities))
        pipeline.append(PathExtractionAgent(self.client,self.reason_model,query=self.clarified_query))
        pipeline.append(HypothesisGenerationAgent(self.client,self.reason_model,query=self.clarified_query,max_paths=5,hypotheses_per_path=3))
        pipeline.append(ReflectionAgent(self.client,self.reason_model))
        pipeline.append(HypothesisEditAgent(client=self.client,model_name=self.reason_model,query=self.clarified_query))
        return pipeline
    def run(self):
        user_query=self.user_query
        queryclarifyagent = QueryClarifyAgent(self.client, self.model_name) # type: ignore
        response = queryclarifyagent.process(user_query)
        clarified_query = response.get("clarified_question", user_query) # type: ignore
        self.clarified_query=clarified_query
        core_entities= response.get("core_entities", []) # type: ignore
        intention= response.get("main_intention", "") # type: ignore
        reviewfetcheragent = ReviewFetcherAgent(client, model_name=model_name) # type: ignore
        reviewfetcheragent.process(clarified_query)
        self.core_entities=core_entities
        self.intention=intention
        self.pipeline=self.get_pipeline()
        for agent in self.pipeline:
            agent.process()
            self.memory.dump_json("./snapshots")
        evaluateAgent=ReflectionAgent(client=self.client,model_name=self.reason_model)
        scores=evaluateAgent.get_scores_only()
        open("scores.json","w").write(json.dumps(scores,indent=4))