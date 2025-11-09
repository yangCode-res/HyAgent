from Agents.Entity_extraction.index import main
from ChatLLM.index import ChatLLM
from Logger.index import get_global_logger
from Store.index import get_memory
from ExampleText.index import ExampleText
import os
from openai import OpenAI
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Mechanism_extraction.index import MechanismExtractionAgent
from dotenv import load_dotenv, find_dotenv
if __name__ == "__main__":

    try:
            env_path = find_dotenv(usecwd=True)
            if env_path:
                load_dotenv(env_path, override=False)
    except Exception:
            pass
    test=ExampleText()
    text=test.get_text()
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    # print(open_ai_api,open_ai_url)
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    memory = get_memory()
    relationAgent=RelationshipExtractionAgent(client, model_name="deepseek-chat")
    logger = get_global_logger()
    logger.info("Starting HyGraph...")
    extract_relationships=relationAgent.process(text)
    subgraph=memory.get_subgraph("0")
    MechanismAgent=MechanismExtractionAgent(client, model_name="deepseek-chat")
    mechanism=MechanismAgent.process(memory, max_workers=8)
    print('mechanism',mechanism)
    # print(subgraph.to_dict())
    logger.info("HyGraph finished.")
    logger.info("="*100)
    memory.dump_json("./snapshots")