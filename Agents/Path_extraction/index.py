from typing import List, Optional

from numpy import tri
from openai import OpenAI
from sympy import false

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, Subgraph
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from TypeDefinitions.KnowledgeGraphDefinitions.index import KnowledgeGraph

class PathExtractionAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str,memory:Optional[Memory]=None):
        self.system_prompt=""""""""
        super().__init__(client,model_name,self.system_prompt)
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
        self.query='What are the latest advancements in CRISPR-Cas9 gene editing technology for treating genetic disorders?'
        self.keyEntitys:List[KGEntity]=self.memory.get_key_entities()
        self.knowledgeGraph:KnowledgeGraph=KnowledgeGraph(self.memory.get_allRealationShip())
        # self.knowledgeGraph.init()
        
    def process(self):
        # print(self.knowledgeGraph.Graph)
        subgraph=self.knowledgeGraph.get_subgraph(self.keyEntitys[0],depth=5)
        print(subgraph)
 
    


