import json
import os
from openai import OpenAI
from Agents.Entity_extraction.index import EntityExtractionAgent, main
from Agents.Causal_extraction.index import CausalExtractionAgent
from ExampleText.index import ExampleText
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from HyAgent.TypeDefinitions.TripleDefinitions.KGTriple import export_triples_to_dicts
from Logger.index import get_global_logger
from Store.index import get_memory
if __name__ == "__main__":
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    test=ExampleText()
    json_texts=test.get_text()
    texts=[]
    for text in json_texts:
        texts.append(text["text"])
    client=OpenAI(api_key=open_ai_api,base_url=open_ai_url)
    entityAgent=EntityExtractionAgent()
    entities=entityAgent.run(json_texts)
    causalAgent=CausalExtractionAgent(client, model_name="deepseek-chat")
    causal_relationships=causalAgent.extract_causal_relationships(texts)
    entities_names=[]
    for entity in entities:
        entities_names.append(entity.name)
        for alias in entity.aliases:
            entities_names.append(alias)
    entities_names=list(set(entities_names))
    relationAgent=RelationshipExtractionAgent(client, model_name="deepseek-chat")
    relationships=relationAgent.extract_relationships(texts,entities_names,causal_relationships)
    relationships=export_triples_to_dicts(relationships)
    with open("/home/nas3/biod/dongkun/HyAgent/snapshots/relationships-20251104-205056.json","w",encoding="utf-8") as f:
        json.dump(relationships,f,ensure_ascii=False,indent=4)
