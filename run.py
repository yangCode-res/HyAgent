import json
import os
from httpx import get
from openai import OpenAI
from Agents.Entity_extraction.index import EntityExtractionAgent, main
from ExampleText.index import ExampleText
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Causal_extraction.index import CausalExtractionAgent
from TypeDefinitions.TripleDefinitions.KGTriple import export_triples_to_dicts
from Logger.index import get_global_logger
from Store.index import get_memory
if __name__ == "__main__":
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    test=ExampleText()
    json_texts=test.get_text()
    logger=get_global_logger()
    memory=get_memory()
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    logger.info("Starting HyGraph...")
    relationAgent=RelationshipExtractionAgent(client, model_name="deepseek-chat")
    relationAgent.process(json_texts)
    logger.info("Relationship extraction finished.")
    logger.info("Starting Causal Extraction...")
    causalAgent=CausalExtractionAgent(client, model_name="deepseek-chat")
    causalAgent.process(json_texts)
    logger.info("Causal Extraction finished.")
    logger.info("HyGraph finished.")
    logger.info("="*100)
    memory.dump_json("./snapshots")
