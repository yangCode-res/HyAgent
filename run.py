from asyncio import Task
from copy import copy
import json
import os
import warnings

from Agents.HypothesisGenerationAgent.index import HypothesisGenerationAgent
from dotenv import find_dotenv, load_dotenv
from matplotlib.pyplot import cla
from networkx import core_number
from openai import OpenAI
import sys
from Agents.Causal_extraction.index import CausalExtractionAgent
from Agents.Collaborate_extraction.index import CollaborationExtractionAgent
from Agents.Entity_extraction.index import EntityExtractionAgent
from Agents.Entity_normalize.index import EntityNormalizationAgent
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Review_fetcher.index import ReviewFetcherAgent
from Agents.Temporal_extraction.index import TemporalExtractionAgent
from ExampleText.index import ExampleText
from Agents.Task_scheduler.index import TaskSchedulerAgent
from Agents.Query_clarify.index import QueryClarifyAgent
from Agents.Alignment_triple.index import AlignmentTripleAgent
from Agents.Fusion_subgraph.index import SubgraphMerger
from Agents.KeywordEntitySearchAgent.index import KeywordEntitySearchAgent
from Agents.Path_extraction.index import PathExtractionAgent
from Logger.index import get_global_logger
from Memory.index import load_memory_from_json
from Store.index import get_memory
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
if __name__ == "__main__":
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    model_name=os.environ.get("OPENAI_MODEL")
    memory=get_memory()
    logger=get_global_logger()
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    memory=load_memory_from_json('/home/nas3/biod/dongkun/snapshots/memory-20251212-095930.json')
    user_query = "Cardiovascular diseases and endothelial dysfunction may be related to what factors?"
    hypothesis_agent = HypothesisGenerationAgent(
        client=client,
        model_name=model_name,
        query=user_query,
        memory=memory,
        max_paths=5,
        hypotheses_per_path=3,
    )
    results = hypothesis_agent.process()
    for result in results:
        print("Generated Hypotheses:", result.get("hypotheses"))
        print("Modified Hypotheses:", result.get("modified_hypotheses"))
    timestamp=datetime.now().strftime("%Y%m%d%H%M")
    with open(f'output/output_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(
            results, 
            f, 
            ensure_ascii=False, 
            indent=4, 
            # 只需要这一行 lambda
            default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o)
        )
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
    # # memory.dump_json("./snapshots")
    # keywordAgent=KeywordEntitySearchAgent(client=client, model_name=model_name,memory=memory,keywords=core_entities)
    # keywordAgent.process()
    # PathExtractionAgent=PathExtractionAgent(client=client, model_name=model_name,k=5,memory=memory,query=clarified_query)
    # PathExtractionAgent.process()
    # memory.dump_json("./snapshots")
