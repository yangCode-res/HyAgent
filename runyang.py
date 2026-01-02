import os
import warnings

from dotenv import load_dotenv
from openai import OpenAI

from benchmark.index import Benchmark

load_dotenv()
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
if __name__ == "__main__":
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    model_name=os.environ.get("OPENAI_MODEL")
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    # memory=get_memory()
    benchmark=Benchmark(limit=1,client=client,model_name=model_name)
    benchmark.run()