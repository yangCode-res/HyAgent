
import os
import warnings
from copy import copy

from Agents.Hypotheses_Edit.index import HypothesisEditAgent
from Agents.ReflectionAgent.index import ReflectionAgent
from Agents.HypothesisGenerationAgent.index import HypothesisGenerationAgent
from Agents.TruthHypo.index import TruthHypoAgent
from Memory.index import load_memory_from_json
from dotenv import load_dotenv
from openai import OpenAI

from pipeline.index import Pipeline
from Logger.index import get_global_logger
from Store.index import get_memory

load_dotenv()
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
if __name__ == "__main__":
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    model_name=os.environ.get("OPENAI_MODEL")
    # memory=load_memory_from_json("/data/dongkun/snapshots/memory-20260103-195134.json")
    
    memory=get_memory()
    logger=get_global_logger()
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    user_query="Can we hypothesize the potential relation between Gene MAO-B (4129) and Gene STC2 (8614)? The final hypothesis can be one of ['positive_correlate', 'negative_correlate', 'no_relation']."
    truthHypoAgent=TruthHypoAgent(client,model_name,user_query,memory=memory)
    result=truthHypoAgent.process()
    logger.info(f"TruthHypoAgent result: {result}")
    # pipeline=Pipeline(user_query,client,model_name,memory)
    # # # pipeline.run()
    # pipeline.run_goOn(memory=memory)
    # hypothesis_agent=HypothesisGenerationAgent(client,model_name,user_query,memory=memory)
    # hypothesis_agent.process()
    # reflection_agent=ReflectionAgent(client,model_name,memory=memory)
    # reflection_agent.process()
    # hypothesis_edit_agent=HypothesisEditAgent(client,model_name,user_query,memory=memory)
    # hypothesis_edit_agent.process()