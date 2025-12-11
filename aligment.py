import json
import os
import faiss
import networkx as nx
import numpy as np
import torch
import Levenshtein  # 必须引入
from dataclasses import dataclass, field, replace
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Set
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI

# 假设这是你的项目结构引用
from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, load_memory_from_json
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
@dataclass
class TimeFormat: 
    pass



class AlignmentTripleAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str, memory: Optional[Memory] = None):
        self.system_prompt = "..." 
        super().__init__(client, model_name, self.system_prompt)
        
        self.memory = memory or get_memory()
        self.logger = get_global_logger()
        self.biobert_dir = "/home/nas2/path/models/SapBERT-from-PubMedBERT-fulltext"
        
        # 加载模型
        self.model = AutoModel.from_pretrained(self.biobert_dir, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.biobert_dir, local_files_only=True)
        self.model.eval()
        self.device = 'cpu' # 如果有 cuda 改为 'cuda'
        
        self.final_triples: List[KGTriple] = []
        self.final_entities: List[KGEntity] = [] 
        self.name_mapping: Dict[str, str] = {}

    def process(self):
        # 1. 收集原始数据
        raw_entities = []
        raw_triples = []
        for subgraph in self.memory.subgraphs.values():
            raw_entities.extend(subgraph.entities.all())
            raw_triples.extend(subgraph.relations.triples)

        print(f"Raw Input: {len(raw_entities)} entities, {len(raw_triples)} triples.")
        raw_entities,raw_triples=self.normalize(raw_entities, raw_triples)
        # 2. 调用整合后的对齐函数
        self.final_triples, self.final_entities, _ = self.align_and_merge(
            raw_entities, 
            raw_triples, 
            threshold=0.90 # 基础向量阈值
        )
        
        # 更新 Memory
        # 注意：这里假设 memory 有 clear 和 add/upsert 方法
        # self.memory.entities.clear() 
        for i in self.final_entities:
            self.memory.entities.upsert(KGEntity(**i.to_dict()))
        # self.memory.entities.upsert_many(self.final_entities) 
        self.memory.relations.triples = self.final_triples
        
        return self.final_triples, self.final_entities

    def get_embeddings(self, texts, batch_size=128):
        """批量获取 SAPBERT 向量"""
        if not texts: return np.array([])
        all_embs = []
        unique_texts = list(set(texts))
        text_to_idx = {t: i for i, t in enumerate(unique_texts)}
        
        for i in tqdm(range(0, len(unique_texts), batch_size), desc="Encoding entities"):
            batch = unique_texts[i : i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                  max_length=64, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                all_embs.append(cls_emb.cpu().numpy())
        
        unique_embs = np.concatenate(all_embs, axis=0)
        final_embs = np.array([unique_embs[text_to_idx[t]] for t in texts])
        return final_embs
    
    def align_and_merge(self, 
                        raw_entities: List[KGEntity], 
                        raw_triples: List[KGTriple], 
                        threshold=0.95, 
                        top_k=5) -> Tuple[List[KGTriple], List[KGEntity], Dict[str, str]]:
        
        # 定义受保护的类型（归一化为小写）
        # 这些类型的实体将跳过向量对齐，只允许基于显式别名的合并
        PROTECTED_TYPES = {'gene', 'biomarker'}

        # --- 内部辅助函数 ---
        def _calc_richness(ent: KGEntity) -> int:
            score = 0
            if ent.entity_type and ent.entity_type.lower() != 'unknown': score += 100
            if ent.description: score += len(ent.description)
            score += len(ent.aliases) * 10
            if ent.normalized_id and ent.normalized_id != "N/A": score += 200
            if len(ent.name) > 2: score += 10 
            return score

        def _merge_entity_list(ent_list: List[KGEntity], all_names: List[str]) -> KGEntity:
            if not ent_list: return None
            base = max(ent_list, key=_calc_richness)
            merged_aliases = set(all_names)
            merged_desc = base.description
            merged_type = base.entity_type
            merged_nid = base.normalized_id
            
            for e in ent_list:
                if e is base: continue
                if not merged_desc and e.description: merged_desc = e.description
                elif e.description and len(e.description) > len(merged_desc): merged_desc = e.description
                
                if merged_type.lower() == 'unknown' and e.entity_type.lower() != 'unknown':
                    merged_type = e.entity_type
                
                if (not merged_nid or merged_nid == "N/A") and (e.normalized_id and e.normalized_id != "N/A"):
                    merged_nid = e.normalized_id
            
            if base.name in merged_aliases: merged_aliases.remove(base.name)
            return replace(base, aliases=list(merged_aliases), description=merged_desc, entity_type=merged_type, normalized_id=merged_nid)

        # ==========================================
        # Step 1: 构建别名图 & 类型索引
        # ==========================================
        print("Step 1: Building Alias Graph & Type Index...")
        G = nx.Graph()
        str_to_raw_entities = defaultdict(list)
        
        # 【新增】: 记录每个字符串关联的实体类型集合
        # 格式: "BRCA1" -> {"gene"}, "IL-6" -> {"cytokine", "gene"}
        str_to_types: Dict[str, Set[str]] = defaultdict(set)

        for ent in raw_entities:
            symbols = {ent.name} | set(ent.aliases)
            symbols = {s for s in symbols if s and s.strip()}
            if not symbols: continue
            
            # 记录类型
            current_type = ent.entity_type.lower() if ent.entity_type else "unknown"
            
            for s in symbols:
                str_to_raw_entities[s].append(ent)
                G.add_node(s)
                # 记录该字符串属于什么类型
                str_to_types[s].add(current_type)
            
            # 建立硬连接
            symbol_list = list(symbols)
            for i in range(len(symbol_list)):
                for j in range(i + 1, len(symbol_list)):
                    G.add_edge(symbol_list[i], symbol_list[j], type='hard')

        for t in raw_triples:
            if t.head not in G: G.add_node(t.head)
            if t.tail not in G: G.add_node(t.tail)

        # ==========================================
        # Step 2: 向量化 (Vectorization)
        # ==========================================
        print("Step 2: Vectorizing...")
        all_node_strings = list(G.nodes())
        if not all_node_strings: return [], [], {}
        
        id2name = {i: name for i, name in enumerate(all_node_strings)}
        
        embeddings = self.get_embeddings(all_node_strings)
        embeddings = embeddings.astype(np.float32)
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)
        
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        D, I = index.search(embeddings, int(top_k))
        
        # ==========================================
        # Step 3: 混合校验与受保护类型过滤
        # ==========================================
        print("Step 3: Hybrid Verification with Type Guard...")
        edges_added = 0
        
        for i in range(len(all_node_strings)):
            src_name = id2name[i]
            
            # 【新增】检查源实体是否受保护
            src_types = str_to_types.get(src_name, set())
            # 如果该名字关联的类型里包含 gene 或 biomarker，则 src_is_protected = True
            src_is_protected = bool(src_types & PROTECTED_TYPES)

            for j, score in zip(I[i], D[i]):
                if i == j or j == -1: continue 
                tgt_name = id2name[j]
                
                if G.has_edge(src_name, tgt_name): continue

                # 【新增】类型守卫逻辑 (Type Guard)
                # 检查目标实体是否受保护
                tgt_types = str_to_types.get(tgt_name, set())
                tgt_is_protected = bool(tgt_types & PROTECTED_TYPES)
                
                # 核心逻辑：如果任一方是 Gene/Biomarker，直接禁止向量合并
                # 除非它们原本就有硬别名连接(Step 1已处理)，否则不让 SapBERT 拉近它们
                if src_is_protected or tgt_is_protected:
                    continue

                # --- 下面是常规混合校验 ---
                should_merge = False
                
                # 规则 A: 极高相似度 (非保护类型才允许)
                if score >= 0.985: 
                    should_merge = True
                # 规则 B: 较高相似度 + 字面相似
                elif score >= threshold:
                    lex_sim = Levenshtein.ratio(src_name.lower(), tgt_name.lower())
                    if lex_sim > 0.5: 
                        should_merge = True
                
                if should_merge:
                    G.add_edge(src_name, tgt_name, type='soft', weight=score)
                    edges_added += 1

        print(f"Added {edges_added} semantic edges (Protected types skipped).")

        # ==========================================
        # Step 4: 聚类与合并 (Clustering)
        # ==========================================
        print("Step 4: Clustering & Merging...")
        clusters = list(nx.connected_components(G))
        
        final_entities = []
        name_mapping = {} 
        merged_entity_map = {}

        for cluster in clusters:
            cluster_list = list(cluster)
            
            # 确定标准名
            valid_names = [n for n in cluster_list if len(n) >= 3]
            canonical_name = sorted(valid_names, key=len)[0] if valid_names else sorted(cluster_list, key=len)[0]

            cluster_raw_entities = []
            for name_str in cluster_list:
                name_mapping[name_str] = canonical_name
                if name_str in str_to_raw_entities:
                    cluster_raw_entities.extend(str_to_raw_entities[name_str])
            
            unique_raw_entities = list({id(e): e for e in cluster_raw_entities}.values())
            
            # ================= [新增打印逻辑 开始] =================
            # 只有当发生了"合并"行为时才打印，避免刷屏
            # 条件：涉及超过1个实体对象 OR 涉及超过1个不同的名字
            if len(unique_raw_entities) > 1 or len(cluster_list) > 1:
                print(f"\n🔹 [Merge Event] Canonical Name: '{canonical_name}'")
                
                # 1. 打印同义词簇的所有名字
                print(f"   └── Synonyms/Aliases ({len(cluster_list)}): {cluster_list}")
                
                # 2. 打印涉及的原始实体对象
                if len(unique_raw_entities) > 1:
                    print(f"   └── ⚠️ Merging {len(unique_raw_entities)} Distinct Entities:")
                    for idx, e in enumerate(unique_raw_entities):
                        print(f"       {idx+1}. ID: {e.entity_id:<10} | Name: {e.name:<20} | Type: {e.entity_type}")
                elif len(unique_raw_entities) == 1:
                    print(f"   └── Single Entity Updated: ID {unique_raw_entities[0].entity_id} ({unique_raw_entities[0].name})")
                else:
                    print(f"   └── No Entity Objects (Pure string merge from Triples)")
            # ================= [新增打印逻辑 结束] =================

            if unique_raw_entities:
                final_ent = _merge_entity_list(unique_raw_entities, cluster_list)
                if final_ent.name != canonical_name:
                    if final_ent.name not in final_ent.aliases:
                        final_ent.aliases.append(final_ent.name)
                    final_ent = replace(final_ent, name=canonical_name)
            else:
                final_ent = KGEntity(
                    entity_id=f"auto-{abs(hash(canonical_name))}",
                    name=canonical_name,
                    entity_type="Unknown",
                    aliases=[n for n in cluster_list if n != canonical_name]
                )
            
            final_entities.append(final_ent)
            merged_entity_map[canonical_name] = final_ent

        # ==========================================
        # Step 5: 重写三元组
        # ==========================================
        print("Step 5: Rewriting Triples...")
        final_triples = []
        seen_triples = set()
        
        for t in raw_triples:
            new_h = name_mapping.get(t.head, t.head)
            new_t = name_mapping.get(t.tail, t.tail)
            
            if new_h == new_t: continue
            
            subj_obj = merged_entity_map.get(new_h)
            obj_obj = merged_entity_map.get(new_t)
            
            triple_key = (new_h, t.relation, new_t)
            
            if triple_key not in seen_triples:
                seen_triples.add(triple_key)
                new_triple = replace(t, head=new_h, tail=new_t, subject=subj_obj, object=obj_obj)
                final_triples.append(new_triple)

        return final_triples, final_entities, name_mapping

    def normalize(self, entities: List[KGEntity], triples: List[KGTriple]):
        if not entities: # type: ignore
            return entities, triples

        # --- 1. 构建同义词连通图 ---
        # 节点：所有的 name 和 alias 字符串
        # 边：同一个实体内的 name 和 alias 之间互连
        g = nx.Graph()
        
        # 记录每个名字对应的原始实体（用于后续更新属性）
        name_to_entities = defaultdict(list)

        for ent in entities:
            # 收集该实体携带的所有名称符号
            # 过滤空字符串
            symbols = {ent.name} | set(ent.aliases)
            symbols = {s for s in symbols if s and s.strip()}
            
            if not symbols:
                continue
                
            symbol_list = list(symbols)
            base_node = symbol_list[0]
            
            # 将所有名字加入图并连线
            g.add_node(base_node)
            for s in symbol_list:
                name_to_entities[s].append(ent)
                if s != base_node:
                    g.add_edge(base_node, s)

        # --- 2. 生成映射字典 (Variant -> Canonical) ---
        # 找出连通分量
        clusters = list(nx.connected_components(g))
        print(f"Found {len(clusters)} explicit synonym groups.")
        
        normalization_map: Dict[str, str] = {}
        
        for cluster in clusters:
            cluster_list = list(cluster)
            
            # 策略：选择最长的名字作为标准名
            # 例如: ["NO", "Nitric Oxide"] -> "Nitric Oxide"
            canonical_name = max(cluster_list, key=len)
            
            for name in cluster_list:
                normalization_map[name] = canonical_name

        # --- 3. 重写实体 (Rewrite Entities) ---
        normalized_entities = []
        # 使用 seen_ids 防止重复处理同一个对象（虽然 replace 会生成新对象，但输入列表可能有重复引用）
        seen_ids = set()
        
        for ent in entities:
            canon_name = normalization_map.get(ent.name, ent.name)
            
            old_names = {ent.name} | set(ent.aliases)
            new_aliases = set()
            
            for n in old_names:
                if n != canon_name:
                    new_aliases.add(n)
            
            # 创建新实体对象
            new_ent = replace(ent, 
                            name=canon_name, 
                            aliases=list(new_aliases))
            
            normalized_entities.append(new_ent)

        # --- 4. 重写三元组 (Rewrite Triples) ---
        normalized_triples = []
        for t in triples:
            # 查找映射，如果没在 map 里（说明没别名信息），保持原样
            new_head = normalization_map.get(t.head, t.head)
            new_tail = normalization_map.get(t.tail, t.tail)
            
            # 如果名字没变，直接复用；变了则 replace
            if new_head != t.head or new_tail != t.tail:
                # 注意：这里只改了字符串。
                # 如果 triples 里包含 subject/object 对象引用，最好置空或指向新的实体，
                # 但由于实体列表也重建了，这里先只处理文本，后续流程会重新链接对象。
                new_t = replace(t, head=new_head, tail=new_tail)
                normalized_triples.append(new_t)
            else:
                normalized_triples.append(t)
                
        print(f"Normalized {len(normalized_entities)} entities and {len(normalized_triples)} triples.")
        return normalized_entities, normalized_triples
# --- 使用示例 ---
if __name__ == "__main__":
    # 假设环境配置
    memory_path = '/home/nas2/path/yangmingjian/code/hygraph/snapshots/memory-20251209-231406.json'
    if os.path.exists(memory_path):
        memory = load_memory_from_json(memory_path)
        
        open_ai_api = os.environ.get("OPENAI_API_KEY")
        open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
        client = OpenAI(api_key=open_ai_api, base_url=open_ai_url)
        model_name=os.environ.get("OPENAI_MODEL")

        aligner = AlignmentTripleAgent(client, model_name, memory=memory)
        
        # 运行处理
        triples, entities = aligner.process()

        aligner.memory.dump_json("./snapshots")
        # print("\n--- Processing Complete ---")
        # print(f"Total Merged Entities: {len(entities)}")
        # print(f"Total Merged Triples: {len(triples)}")
        
        # if len(triples) > 0:
        #     print("\nExample Triple:")
        #     t = triples[0]
        #     print(f"{t.head} --[{t.relation}]--> {t.tail}")
        #     print(f"Linked Subject Name: {t.subject.name if t.subject else 'None'}")
        #     print(f"Linked Subject Aliases: {t.subject.aliases if t.subject else 'None'}")
        