from typing import List, Optional
from numpy import tri
from sympy import false
from openai import OpenAI
from Core.Agent import Agent
from Memory.index import Memory, Subgraph
from Logger.index import get_global_logger
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from Store.index import get_memory


class ConflictResolveAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str,memory:Optional[Memory]=None):
        self.system_prompt=""""""""
        super().__init__(client,model_name,self.system_prompt)
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
    def process(self):
        subgraphs=self.memory.subgraphs

 
    


