
import os
import warnings
from copy import copy

from Agents.Hypotheses_Edit.index import HypothesisEditAgent
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
    memory=load_memory_from_json("/data/dongkun/snapshots/memory-20260103-145943.json")
    logger=get_global_logger()
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    user_query="Can we hypothesize the potential relation between Gene CLP (12950) and Gene TNF-alpha (21926)? The final hypothesis can be one of ['positive_correlate', 'negative_correlate', 'no_relation']."
    # pipeline=Pipeline(user_query,client,model_name,memory)
    # pipeline.run()
    hypothesis_edit_agent=HypothesisEditAgent(client,model_name,user_query,memory)
    hypothesis_edit_agent.process()