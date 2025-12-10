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
load_dotenv()
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
if __name__ == "__main__":
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    model_name=os.environ.get("OPENAI_MODEL")
    # memory=get_memory()
    test=ExampleText()
    json_texts=test.get_text()
    logger=get_global_logger()
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    agent = ReviewFetcherAgent(client, model_name=model_name)
    
    user_query = "Cardiovascular diseases and endothelial dysfunction may be related to what factors?"
    # agent.process(user_query)
    memory=load_memory_from_json('/home/nas2/path/yangmingjian/code/hygraph/snapshots/memory-20251210-170857.json')
    # alignmentAgent=AlignmentTripleAgent(client=client, model_name=model_name,memory=memory)
    # alignmentAgent.process()
    # fusionAgent=SubgraphMerger(client=client, model_name=model_name,memory=memory)
    # fusionAgent.process()
    EntityNormalizationAgent=EntityNormalizationAgent(client=client, model_name=model_name,memory=memory)
    EntityNormalizationAgent.process()
    # keywordAgent=KeywordEntitySearchAgent(client=client, model_name=model_name,memory=memory,keyword="endothelial dysfunction")
    # keywordAgent.process()
    # PathExtractionAgent=PathExtractionAgent(client=client, model_name=model_name,k=5,memory=memory)
    # PathExtractionAgent.process()
    memory.dump_json("./snapshots")

    # memory.dump_json("./snapshots")
    # fusionAgent=SubgraphMerger(client=client, model_name=model_name,memory=memory)
    # fusionAgent.process(memory=memory)
#     visualize_global_kg(memory)
#     export_memory_to_neo4j(
#         mem=memory,
#         uri="bolt://localhost:7687",
#         user="neo4j",
#         password="mingming0.+",
#         clear_db=True,      # 如果希望每次都清空图再导入
#         max_edges=5000,
# )
    # memory.dump_json("./snapshots")
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