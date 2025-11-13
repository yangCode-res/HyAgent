from Agents.Entity_extraction.index import main
from ChatLLM.index import ChatLLM
from Logger.index import get_global_logger
from Store.index import get_memory
from ExampleText.index import ExampleText
import os
from openai import OpenAI
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Mechanism_extraction.index import MechanismExtractionAgent
from Agents.Entity_extraction.index import EntityExtractionAgent
from dotenv import load_dotenv, find_dotenv
from Agents.Entity_normalize.index import EntityNormalizationAgent
from Agents.Alignment_triple.index import AlignmentTripleAgent
from Memory.index import load_memory_from_json
if __name__ == "__main__":

    try:
            env_path = find_dotenv(usecwd=True)
            if env_path:
                load_dotenv(env_path, override=False)
    except Exception:
            pass
    test=ExampleText()
    text=test.get_text()
    logger = get_global_logger()
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    # print(open_ai_api,open_ai_url)
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    memory = get_memory()
    memory = load_memory_from_json("./snapshots/memory-20251113-094606.json")
   
#     extract_agent=EntityExtractionAgent()
    # normalize_agent=EntityNormalizationAgent(client, model_name="deepseek-chat")
    logger.info("Starting HyGraph...")
    alignment_agent=AlignmentTripleAgent(client, model_name="deepseek-chat",memory=memory)
    alignment_agent.process()
    # print(subgraph.to_dict())
#     extract_agent.run(text)
    # normalize_agent.process(memory)
    logger.info("HyGraph finished.")
    logger.info("="*100)
    memory.dump_json("./snapshots")