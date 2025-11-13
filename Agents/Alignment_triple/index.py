
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
            id2idx, adj = self.build_adj_for_subgraph(sg)
            self.subgraph_adj[sg_id] = (adj, id2idx)
            # 简单打印检查一下
            self.logger.info(
                f"[Adjacency] subgraph={sg_id}, |V|={adj.shape[0]}, |E|={int(adj.sum())}"
            )
    def build_adj_for_subgraph(self, subgraph: Subgraph):
        """
        给一个子图（Memory 里的 Subgraph 对象），返回：
        - id2idx: entity_id -> 行列索引
        - adj: 邻接矩阵 (numpy.ndarray, shape = [n_entities, n_entities])
        """
        # 1. 所有实体，构建 entity_id -> idx
        entities = subgraph.get_entities()   # 这里应该是 KGEntity 的列表
        relations = subgraph.get_relations() # 这里应该是 KGTriple 的列表

        id2idx: Dict[str, int] = {}
        for idx, ent in enumerate(entities):
            eid = ent.get_id()
            id2idx[eid] = idx

        n = len(id2idx)
        adj = np.zeros((n, n), dtype=int)
        # 2. 遍历关系，用 subject/object 里的 entity_id 建边
        for rel in relations:
            subj = rel.get_subject()   # 预期是 KGEntity 或 None
            obj  = rel.get_object()
            # 有些 triple 的 subject / object 可能是 None（比如你 JSON 里看到的 null），要先过滤掉
            if subj is None or obj is None:
                continue
            subj=KGEntity.from_dict(subj)
            obj=KGEntity.from_dict(obj)
            head_id = subj.get_id()
            tail_id = obj.get_id()

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
    