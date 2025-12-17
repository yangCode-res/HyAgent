import os
import warnings

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from Agents.Causal_extraction.index import CausalExtractionAgent
from Agents.Collaborate_extraction.index import CollaborationExtractionAgent
from Agents.Entity_extraction.index import EntityExtractionAgent
from Agents.Entity_normalize.index import EntityNormalizationAgent
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Review_fetcher.index import ReviewFetcherAgent
from Agents.Temporal_extraction.index import TemporalExtractionAgent
from Agents.Fusion_subgraph.index import SubgraphMerger
from Agents.KeywordEntitySearchAgent.index import KeywordEntitySearchAgent
from Agents.Path_extraction.index import PathExtractionAgent
from ExampleText.index import ExampleText
from Logger.index import get_global_logger
from Memory.index import load_memory_from_json
from Store.index import get_memory
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from dotenv import load_dotenv
from Agents.Fusion_subgraph.index import SubgraphMerger
from utils.visualize import visualize_global_kg,export_memory_to_neo4j
from Agents.Alignment_triple.index import AlignmentTripleAgent
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