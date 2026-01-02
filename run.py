import json
import os
import sys
import warnings
from asyncio import Task
from copy import copy

from Agents.Task_scheduler.index import TaskSchedulerAgent
from Agents.Temporal_extraction.index import TemporalExtractionAgent
from dotenv import find_dotenv, load_dotenv
from matplotlib.pyplot import cla
from networkx import core_number
from openai import OpenAI

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
from ExampleText.index import ExampleText
from Logger.index import get_global_logger
from Memory.index import load_memory_from_json
from Store.index import get_memory
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from utils.visualize import visualize_global_kg

load_dotenv()
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
if __name__ == "__main__":
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    model_name=os.environ.get("OPENAI_MODEL")
    memory=get_memory()
    logger=get_global_logger()
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    memory=load_memory_from_json('/home/nas2/path/yangmingjian/code/hygraph/snapshots/memory-20251224-174934.json')
    visualize_global_kg(memory)
    user_query = "Tuberculous meningitis is often lethal, and many survivors have disabilities despite antimicrobial treatment and adjunctive glucocorticoid therapy. Standard-dose rifampin has limited central nervous system penetration."
    # queryclarifyagent = QueryClarifyAgent(client, model_name=model_name) # type: ignore
    # response = queryclarifyagent.process(user_query)
    # clarified_query = response.get("clarified_question", user_query) # type: ignore
    # core_entities= response.get("core_entities", []) # type: ignore
    # intention= response.get("main_intention", "") # type: ignore
    # keywordAgent=KeywordEntitySearchAgent(client=client, model_name=model_name,memory=memory,keywords=core_entities)
    # print("core_entities=>",core_entities)
    # keywordAgent.process()
    # path_extraction_agent = PathExtractionAgent(client=client, model_name=model_name,k=20,memory=memory,query=clarified_query)
    # path_extraction_agent.process()
    # hypothesis_agent = HypothesisGenerationAgent(
    #     client=client,
    #     model_name=model_name,
    #     query=user_query,
    #     memory=memory,
    #     max_paths=5,
    #     hypotheses_per_path=3,
    # )
    # results = hypothesis_agent.process()
    # memory.dump_json("./snapshots")
    # reflection_agent = ReflectionAgent(client=client,model_name=model_name,memory=memory)
    # reflection_agent.process()
    # hypothesis_edit_agent = HypothesisEditAgent(client=client,model_name=model_name,query=user_query,memory=memory)
    # hypothesis_edit_agent.process()

    # path_extraction_agent = PathExtractionAgent(client=client, model_name=model_name,k=20,memory=memory,query=user_query)
    # path_extraction_agent.process()
    # memory.dump_json("./snapshots")
    # hypothesis_agent = HypothesisGenerationAgent(
    #     client=client,
    #     model_name=model_name,
    #     query=user_query,
    #     memory=memory,
    #     max_paths=5,
    #     hypotheses_per_path=3,
    # )
    # results = hypothesis_agent.process()
    # for result in results:
    #     print("Generated Hypotheses:", result.get("hypotheses"))
    #     print("Modified Hypotheses:", result.get("modified_hypotheses"))
    # with open('output.json', 'w', encoding='utf-8') as f:
    #     json.dump(
    #         results, 
    #         f, 
    #         ensure_ascii=False, 
    #         indent=4, 
    #         # 只需要这一行 lambda
    #         default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o)
    #     )
    # queryclarifyagent = QueryClarifyAgent(client, model_name=model_name) # type: ignore
    # response = queryclarifyagent.process(user_query)
    # clarified_query = response.get("clarified_question", user_query) # type: ignore
    # core_entities= response.get("core_entities", []) # type: ignore
    # intention= response.get("main_intention", "") # type: ignore
    # print("Clarified Query:", clarified_query)
    # print("Core Entities:", core_entities)
    # reviewfetcheragent = ReviewFetcherAgent(client, model_name=model_name) # type: ignore
    # reviewfetcheragent.process(clarified_query)

    # entityAgent=EntityExtractionAgent(client=client, model=model_name)
    # entityAgent.process()
    # entityAgent=EntityExtractionAgent(client=client, model=model_name)
    # entityAgent.process()

    # normalizeAgent=EntityNormalizationAgent(client=client, model_name=model_name)
    # normalizeAgent.process()
    # logger.info("Relationship extraction started...")
    # memory.dump_json("./snapshots")
    # relationAgent=RelationshipExtractionAgent(client=client, model_name=model_name)
    # relationAgent.process()    
    # logger.info("Relationship extraction finished.")

    # logger.info("Collaboration extraction started...")
    # collaborationAgent=CollaborationExtractionAgent(client=client, model_name=model_name,memory=memory)
    # collaborationAgent.process()
    # logger.info("Collaboration extraction finished.")
    # memory.dump_json("./snapshots")
    # casualAgent=CausalExtractionAgent(client=client, model_name=model_name,memory=memory)
    # casualAgent.process()
    # alignmentAgent=AlignmentTripleAgent(client=client, model_name=model_name,memory=memory)
    # alignmentAgent.process()
    # fusionAgent=SubgraphMerger(client=client, model_name=model_name,memory=memory)
    # fusionAgent.process()
    # memory.dump_json("./snapshots")
    # keywordAgent=KeywordEntitySearchAgent(client=client, model_name=model_name,memory=memory,keywords=core_entities)
    # keywordAgent.process()
    # PathExtractionAgent=PathExtractionAgent(client=client, model_name=model_name,k=5,memory=memory,query=clarified_query)
    # PathExtractionAgent.process()
    # memory.dump_json("./snapshots")
