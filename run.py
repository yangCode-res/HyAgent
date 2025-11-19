import os
from openai import OpenAI
from ExampleText.index import ExampleText
from Agents.Causal_extraction.index import CausalExtractionAgent
from Agents.Collaborate_extraction.index import CollaborationExtractionAgent
from Agents.Entity_extraction.index import EntityExtractionAgent
from Agents.Entity_normalize.index import EntityNormalizationAgent
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Temporal_extraction.index import TemporalExtractionAgent
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from Memory.index import load_memory_from_json
from Logger.index import get_global_logger
from Store.index import get_memory
from dotenv import load_dotenv, find_dotenv
if __name__ == "__main__":
    try:
            env_path = find_dotenv(usecwd=True)
            if env_path:
                load_dotenv(env_path, override=False)
    except Exception:
            pass
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    memory=get_memory()
    test=ExampleText()
    json_texts=test.get_text()
    logger=get_global_logger()
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    logger.info("Entity extraction started...")
    entityAgent=EntityExtractionAgent()
    entityAgent.process(documents=json_texts)
    logger.info("Entity extraction finished.")
    logger.info("Entity normalization started...")
    normalizeAgent=EntityNormalizationAgent(client, model_name="deepseek-chat")
    normalizeAgent.process(memory)
    logger.info("Relationship extraction started...")
    relationAgent=RelationshipExtractionAgent(client, model_name="deepseek-chat")
    relationAgent.run(json_texts)    
    logger.info("Relationship extraction finished.")
    # memory=load_memory_from_json('/home/nas3/biod/dongkun/snapshots/memory-20251110-165915.json')
    logger.info("Collaboration extraction started...")
    collaborationAgent=CollaborationExtractionAgent(client, model_name="deepseek-chat",memory=memory)
    collaborationAgent.process()
    memory.dump_json("./snapshots")
    logger.info("Collaboration extraction finished.")
    logger.info("Causal extraction started...")
    causalAgent=CausalExtractionAgent(client, model_name="deepseek-chat",memory=memory)
    causalAgent.process(json_texts)
    logger.info("Causal extraction finished.")
    logger.info("Temporal extraction started...")
    agent=TemporalExtractionAgent(client=client,model_name="deepseek-chat")
    agent.process()
    logger.info("Temporal extraction finished.")
    logger.info("HyGraph finished.")
    logger.info("="*100)
    memory.dump_json("./snapshots")