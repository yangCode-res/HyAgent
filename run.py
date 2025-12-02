from asyncio import Task
import os
import warnings

from dotenv import find_dotenv, load_dotenv
from matplotlib.pyplot import cla
from networkx import core_number
from openai import OpenAI

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
    user_query = "(1)Cancer classification accuracy and stability vary between different network constructions. (2) Ordering network displays better overall performance compared to correlation network across multiple cancer datasets. (3) Optimal classification performance does not necessarily correlate with the number of genes used in the model.  "
    queryclarifyagent = QueryClarifyAgent(client, model_name=model_name) # type: ignore
    response = queryclarifyagent.process(user_query)
    clarified_query = response.get("clarified_question", user_query) # type: ignore
    core_entities= response.get("core_entities", []) # type: ignore
    intention= response.get("main_intention", "") # type: ignore
    print("Clarified Query:", clarified_query)
    reviewfetcheragent = ReviewFetcherAgent(client, model_name=model_name) # type: ignore
    reviewfetcheragent.process(clarified_query)
    task_scheduler=TaskSchedulerAgent(client=client, model_name=model_name) # type: ignore
    pipeline=task_scheduler.process(clarified_query)
    pipeline.run()
# agent.memory.dump_json("./snapshots")
#     logger.info("Entity extraction started...")
#     entityAgent=EntityExtractionAgent(client=client, model=model_name)
#     entityAgent.process(documents=json_texts)
#     logger.info("Entity extraction finished.")
#     logger.info("Entity normalization started...")
#     normalizeAgent=EntityNormalizationAgent(client=client, model_name=model_name)
#     normalizeAgent.process(memory)
#     logger.info("Relationship extraction started...")
#     relationAgent=RelationshipExtractionAgent(client=client, model_name=model_name)
#     relationAgent.process(json_texts)    
#     logger.info("Relationship extraction finished.")
#     # memory=load_memory_from_json('/home/nas3/biod/dongkun/snapshots/memory-20251110-165915.json')
#     logger.info("Collaboration extraction started...")
#     collaborationAgent=CollaborationExtractionAgent(client=client, model_name=model_name,memory=memory)
#     collaborationAgent.process()
#     memory.dump_json("./snapshots")
#     logger.info("Collaboration extraction finished.")
#     logger.info("Causal extraction started...")
#     causalAgent=CausalExtractionAgent(client=client, model_name=model_name,memory=memory)
#     causalAgent.process(json_texts)
#     logger.info("Causal extraction finished.")
#     logger.info("Temporal extraction started...")
#     agent=TemporalExtractionAgent(client=client,model_name=model_name)
#     agent.process()
#     logger.info("Temporal extraction finished.")
#     logger.info("HyGraph finished.")
#     logger.info("="*100)
#     memory.dump_json("./snapshots")