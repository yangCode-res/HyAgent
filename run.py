from asyncio import Task
from copy import copy
import os
import warnings

from dotenv import find_dotenv, load_dotenv
from matplotlib.pyplot import cla
from networkx import core_number
from openai import OpenAI

# from Agents.Causal_extraction.index import CausalExtractionAgent
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
    test=ExampleText()
    json_texts=test.get_text()
    logger=get_global_logger()
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    user_query = "What are the latest advancements in CRISPR-Cas9 gene editing technology for treating genetic disorders?"
    queryclarifyagent = QueryClarifyAgent(client, model_name=model_name) # type: ignore
    response = queryclarifyagent.process(user_query)
    clarified_query = response.get("clarified_question", user_query) # type: ignore
    core_entities= response.get("core_entities", []) # type: ignore
    intention= response.get("main_intention", "") # type: ignore
    print("Clarified Query:", clarified_query)
    reviewfetcheragent = ReviewFetcherAgent(client, model_name=model_name) # type: ignore
    reviewfetcheragent.process(user_query)
    # task_scheduler=TaskSchedulerAgent(client=client, model_name=model_name) # type: ignore
    # pipeline=task_scheduler.process(user_query)
    # pipeline.run()
   # agent.memory.dump_json("./snapshots")
    # logger.info("Entity extraction started...")
    entityAgent=EntityExtractionAgent(client=client, model=model_name)
    entityAgent.process()

    normalizeAgent=EntityNormalizationAgent(client=client, model_name=model_name)
    normalizeAgent.process()
    logger.info("Relationship extraction started...")

    relationAgent=RelationshipExtractionAgent(client=client, model_name=model_name)
    relationAgent.process()    
    logger.info("Relationship extraction finished.")

    logger.info("Collaboration extraction started...")
    memory.dump_json("./snapshots")
    collaborationAgent=CollaborationExtractionAgent(client=client, model_name=model_name,memory=memory)
    collaborationAgent.process()
    logger.info("Collaboration extraction finished.")
    # logger.info("Causal extraction started...")
    # causalAgent=CausalExtractionAgent(client=client, model_name=model_name)
    # causalAgent.process()
    # logger.info("Causal extraction finished.")
    # # memory.dump_json("./snapshots")
    # # memory=load_memory_from_json('/home/nas3/biod/dongkun/snapshots/memory-20251203-144947.json')
    # logger.info("Alignment extraction started...")
    alignmentAgent=AlignmentTripleAgent(client=client, model_name=model_name,memory=memory)
    alignmentAgent.process()
    # logger.info("Alignment extraction finished.")
    # subgraphs_ids=[]
    # for subgraph in memory.subgraphs.values():
    #     if subgraph.entities.all()==[]:
    #         subgraphs_ids.append(subgraph.id)
    # for subgraph_id in subgraphs_ids:
    #     memory.remove_subgraph(subgraph_id)
    #     logger.info(f"Removed empty subgraph {subgraph_id}")
    # logger.info("Fusion Subgraphs started...")
    fusionAgent=SubgraphMerger(client=client, model_name=model_name,memory=memory)
    fusionAgent.process()
    # logger.info("Fusion Subgraphs finished...")
    memory.dump_json("./snapshots")