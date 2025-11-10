import json
import os
from httpx import get
from openai import OpenAI
from Agents.Entity_extraction.index import EntityExtractionAgent, main
from ExampleText.index import ExampleText
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Causal_extraction.index import CausalExtractionAgent
from Agents.Collaborate_extraction.index import CollaborationExtractionAgent
from Memory.index import load_memory_from_json
from TypeDefinitions.TripleDefinitions.KGTriple import export_triples_to_dicts
from Logger.index import get_global_logger
from Store.index import get_memory
if __name__ == "__main__":
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    test=ExampleText()
    json_texts=test.get_text()
    logger=get_global_logger()
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    # logger.info("Entity extraction started...")
    # entityAgent=EntityExtractionAgent()
    # entityAgent.run(documents=json_texts)
    # memory.dump_json("./snapshots")
    # logger.info("Entity extraction finished.")
    # logger.info("Relationship extraction started...")
    # relationAgent=RelationshipExtractionAgent(client, model_name="deepseek-chat")
    # relationAgent.process(json_texts)    
    # memory.dump_json("./snapshots")
    # logger.info("Relationship extraction finished.")
    memory=get_memory()
    memory=load_memory_from_json('/home/nas3/biod/dongkun/snapshots/memory-20251110-165915.json')
    logger.info("Collaboration extraction started...")
    collaborationAgent=CollaborationExtractionAgent(client, model_name="deepseek-chat",memory=memory)
    collaborationAgent.process()
    memory.dump_json("./snapshots")
    logger.info("Collaboration extraction finished.")
    logger.info("Causal extraction started...")
    causalAgent=CausalExtractionAgent(client, model_name="deepseek-chat",memory=memory)
    causalAgent.process(json_texts)
    logger.info("Causal extraction finished.")
    logger.info("HyGraph finished.")
    logger.info("="*100)
    memory.dump_json("./snapshots")
