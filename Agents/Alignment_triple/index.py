
from openai import OpenAI
from Core.Agent import Agent
from Memory.index import Memory, Subgraph
from Logger.index import get_global_logger
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from transformers import AutoTokenizer, AutoModel
from Store.index import get_memory
import numpy as np
import torch
from Config.index import BioBertPath
from typing import Dict, List, Tuple, Any, Optional
Embedding = List[float]
class AlignmentTripleAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str,memory:Optional[Memory]=None):
        self.system_prompt=""""""""
        super().__init__(client,model_name,self.system_prompt)
        self.memory=memory or get_memory()
        self.logger=get_global_logger()
        self.subgraph_entity_embeddings: Dict[str, Dict[str, Embedding]] = {}
        self.biobert_dir = BioBertPath
        self.biobert_model=None
        self.biobert_tokenizer=None
        self._load_biobert()
        self.subgraph_adj: Dict[str, Tuple[np.ndarray, Dict[str, int]]] = {}
    def process(self) -> None:
        for sg_id, sg in self.memory.subgraphs.items():
            ent_embeds: Dict[str, Embedding] = {}
            for ent in sg.entities.all():
                text = ent.description or ent.name or ent.normalized_id
                embedding = self._encode_text(text)  # 返回 List[float] 或 np.ndarray
                ent_embeds[ent.entity_id] = embedding
            # 这里的 sg 在类型系统里就是 Subgraph
            self.subgraph_entity_embeddings[sg_id] = ent_embeds
            adj, eid2idx = self.build_adj_for_subgraph(sg)
            self.subgraph_adj[sg_id] = (adj, eid2idx)
            # 简单打印检查一下
            self.logger.info(
                f"[Adjacency] subgraph={sg_id}, |V|={adj.shape[0]}, |E|={int(adj.sum())}"
            )
    def build_adj_for_subgraph(self, subgraph: Subgraph, directed: bool = False):
        """
        给一个子图（就是 memory['subgraphs'][sg_id] 那种 dict），
        返回：
        - entity_id -> 行列索引的映射
        - 邻接矩阵 (numpy.ndarray, shape = [n_entities, n_entities])
        """
        entities = subgraph.get_entities()
        # print(entities)
        relations = subgraph.get_relations()
        id2idx = {}
        for idx, ent in enumerate(entities):
            eid = ent.get_id()
            if not eid:
                continue
            id2idx[eid] = idx
        n = len(id2idx)
        adj = np.zeros((n, n), dtype=int)
        # 2. 遍历关系，用 subject/object 里的 entity_id 建边
        for rel in relations:
            subj = rel.get_subject()
            obj = rel.get_object()
            subj=KGEntity.from_dict(subj)
            obj=KGEntity.from_dict(obj)

            # 只用 subject / object 的 entity_id
            print(subj,obj)
            head_id = subj.get_id() 
            tail_id = obj.get_id() 
            # 如果缺少任一端的 entity_id，就跳过这条边
            if head_id is None or tail_id is None:
                continue
            i = id2idx[head_id]
            j = id2idx[tail_id]
            adj[i, j] += 1
            adj[j, i] += 1
        return id2idx, adj
            
    def _encode_text(self, text: str):
        if not self.biobert_model or not self.biobert_tokenizer:
            self.logger.info(f"[EntityNormalize][BioBERT] model or tokenizer not loaded")
            return None
        with torch.no_grad():
            inputs = self.biobert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            outputs = self.biobert_model(**inputs)
            vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return vec
    def _load_biobert(self) -> None:
        try:
            self.biobert_tokenizer = AutoTokenizer.from_pretrained(
                    self.biobert_dir,
                    local_files_only=True,
                )
            self.biobert_model = AutoModel.from_pretrained(
                    self.biobert_dir,
                    local_files_only=True,
                )
            self.biobert_model.eval()
        except Exception as e:
            self.logger.info(f"[EntityNormalize][BioBERT] load failed ({e}), skip similarity-based suggestions.")
    