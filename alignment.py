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

import pandas as pd  # 引入 pandas
# 假设这是你的项目结构引用
from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, load_memory_from_json
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple

class AlignmentTripleAgent(Agent):
    def __init__(self, client: OpenAI, model_name: str, memory: Optional[Memory] = None):
        self.system_prompt = """You are an expert in biomedical knowledge graph entity alignment.

You will receive a single JSON string as user input, with fields such as:
- "candidates": a pair of objects { "src_name": ..., "tgt_name": ... }

Your task:
1. Parse the JSON input.
2. Decide whether candidates refer to the SAME real-world biomedical entity as the source entity.

Output format (VERY IMPORTANT):
- You MUST respond with STRICT JSON only.
- The JSON must have exactly one top-level key "align" which is a boolean key.
- "align" must be a list of candidate ids (strings) that should be kept.
- Example: {"align": true} means the candidate matches the source entity.

Rules:
- If no candidate should be aligned with the source entity, return {"align": false}.
- Do NOT add any other keys, text, comments, or explanations.
- Do NOT change, rename, or invent candidate ids.
- The response must be valid JSON and parseable by a standard JSON parser.""" 
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
        print("--- Step 0: Data Collection ---")
        # 1. 收集原始数据
        # 使用字典去重，防止同一个实体在不同子图中重复出现导致数量爆炸
        raw_entities_map = {}
        raw_triples = []
        
        # for subgraph in self.memory.subgraphs.values():
        #     # 收集实体 (按 entity_id 去重，或者按 name 去重，视你的数据情况而定)
        #     # 这里假设 entity_id 是唯一的标识符
        for subgraph in self.memory.subgraphs.values():
            for ent in subgraph.entities.all():
                if ent.entity_id not in raw_entities_map:
                    raw_entities_map[ent.entity_id] = ent
        
        # 收集三元组
        raw_triples.extend(triple for subgraph in self.memory.subgraphs.values() for triple in subgraph.relations.triples)

        raw_entities = list(raw_entities_map.values())
        print(f"Raw Input: {len(raw_entities)} unique entities, {len(raw_triples)} triples.")

        # 2. 预处理：基于显式别名的归一化 (Normalize)
        print("\n--- Step 1: Explicit Alias Normalization ---")
        normalized_entities, normalized_triples = self.normalize(raw_entities, raw_triples)
        
        # 3. 核心对齐：基于 SapBERT 的语义合并 (Align & Merge)
        print("\n--- Step 2: Semantic Alignment (SapBERT) ---")
        # 注意：这里传入的是 normalize 后的数据
        self.final_triples, self.final_entities, _ = self.align_and_merge(
            normalized_entities, 
            normalized_triples
        )
        
        # 4. 更新 Memory
        print("\n--- Step 3: Updating Memory ---")
        # 建议先清空旧实体，避免残留
        # self.memory.entities.clear() 
        for i in self.final_entities:
            # 假设 memory.entities.upsert 接受 KGEntity 对象
            self.memory.entities.upsert(KGEntity(**i.to_dict()))
            
        self.memory.relations.triples = self.final_triples
        
        return self.final_triples, self.final_entities

    def get_embeddings(self, texts, batch_size=128):
        """
        [修复版] 批量获取 SAPBERT 向量
        修复了索引错位导致不同实体获得相同向量(Score=1.0)的严重Bug。
        """
        if not texts: return np.array([])
        
        # 1. 确定性去重：使用 sorted 确保每次运行顺序一致，防止 set 的随机性
        unique_texts = sorted(list(set(texts)))
        
        # 2. 建立映射表
        text_to_idx = {t: i for i, t in enumerate(unique_texts)}
        
        all_embs = []
        
        # 3. 批量推理
        # 使用 tqdm 显示进度
        for i in tqdm(range(0, len(unique_texts), batch_size), desc="Encoding unique entities"):
            batch = unique_texts[i : i + batch_size]
            
            # Tokenizer
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                  max_length=64, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 取 [CLS] token (batch_size, hidden_size)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                all_embs.append(cls_emb.cpu().numpy())
        
        if not all_embs:
            return np.array([])
            
        unique_embs = np.concatenate(all_embs, axis=0)
        
        # 4. 安全检查 (至关重要)
        if len(unique_embs) != len(unique_texts):
            raise RuntimeError(f"向量生成数量不匹配! 文本数: {len(unique_texts)}, 向量数: {len(unique_embs)}")

        # 5. 映射回原始列表顺序
        try:
            final_embs = np.array([unique_embs[text_to_idx[t]] for t in texts])
        except KeyError as e:
            raise RuntimeError(f"索引映射失败，找不到键: {e}")
            
        return final_embs
    
    def align_and_merge(self, 
                        raw_entities: List[KGEntity], 
                        raw_triples: List[KGTriple], 
                        top_k=3) -> Tuple[List[KGTriple], List[KGEntity], Dict[str, str]]:
        
        # 定义受保护的类型（归一化为小写）
        # 这些类型的实体将跳过向量对齐，只允许基于显式别名的合并
        PROTECTED_TYPES = {'gene', 'biomarker'}
        MIN_ENSURE_SCORE=0.99
        MIN_MIX_ENSURE_SCORE=0.96
        MIN_MIX_LEX_SCORE=0.65
        MIN_LLM_CHECKIN_SCORE=0.92
        MIN_LLM_LEX_SCORE=0.40
        # --- 内部辅助函数 ---
        def _calc_richness(ent: KGEntity) -> int:
            score = 0
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
        
        for i in tqdm(range(len(all_node_strings))):
            src_name = id2name[i]
            
            # 【新增】检查源实体是否受保护
            src_types = str_to_types.get(src_name, set())
            # 如果该名字关联的类型里包含 gene 或 biomarker，则 src_is_protected = True

            for j, score in zip(I[i], D[i]):
                if i == j or j == -1: continue 
                tgt_name = id2name[j]
                
                if G.has_edge(src_name, tgt_name): continue
                tgt_types = str_to_types.get(tgt_name, set())
                is_sensitive_pair = bool((src_types | tgt_types) & PROTECTED_TYPES)
                # 【新增】类型守卫逻辑 (Type Guard)
                # 检查目标实体是否受保护
                
                # 核心逻辑：如果任一方是 Gene/Biomarker，直接禁止向量合并
                # 除非它们原本就有硬别名连接(Step 1已处理)，否则不让 SapBERT 拉近它们

                # --- 下面是常规混合校验 ---
                should_merge = False
                
                lex_sim = Levenshtein.ratio(src_name.lower(), tgt_name.lower())
                if is_sensitive_pair:
                    if score>=0.985:
                        should_merge = True
                # 规则 A: 极高相似度 (非保护类型才允许)
                if score >= MIN_ENSURE_SCORE: 
                    should_merge = True
                # 规则 B: 较高相似度 + 字面相似
                elif score >= MIN_MIX_ENSURE_SCORE:
                    if lex_sim > MIN_MIX_LEX_SCORE: 
                        should_merge = True
                # 规则 C: 中高分 + LLM 复核
                elif score >= MIN_LLM_CHECKIN_SCORE and lex_sim > MIN_LLM_LEX_SCORE:
                    prompt=f"""Now, please verify if the following two entity names refer to the SAME real-world biomedical entity.
                    entity 1: "{src_name}"
                    entity 2: "{tgt_name}"
                    Answer with a JSON object: {{"align": true}} if they are the same, or {{"align": false}} if they are different.
                    """
                    res=self.call_llm(prompt)
                    try:
                        data=json.loads(res)
                        if data.get("align", False):
                            should_merge = True
                    except Exception as e:
                        print(f"LLM 解析失败: {e}. 原始响应: {res}")

                if should_merge:
                    G.add_edge(src_name, tgt_name, type='soft', weight=score)
                    edges_added += 1

        print(f"Added {edges_added} semantic edges (Protected types skipped).")

     # ==========================================
        # Step 4: 聚类与合并 (Enhanced Greedy Star Strategy)
        # 引入"反包含"和"关键词互斥"逻辑，彻底消除冗余
        # ==========================================
        print("Step 4: Clustering (Strict Anti-Drift Strategy)...")
        
        # 辅助函数：判断是否构成包含关系（泛指 vs 特指）
        def _is_substring_relation(s1: str, s2: str) -> bool:
            s1_lower, s2_lower = s1.lower(), s2.lower()
            # 如果一个是另一个的子串，且长度差异超过一定比例，视为泛指关系，不可合并
            if s1_lower in s2_lower and len(s1) < len(s2) * 0.8: return True
            if s2_lower in s1_lower and len(s2) < len(s1) * 0.8: return True
            return False

        # 辅助函数：计算 Token Jaccard 相似度
        def _token_jaccard(s1: str, s2: str) -> float:
            # 移除停用词（简单版）
            stopwords = {'of', 'and', 'in', 'the', 'with', 'to', 'for', 'a', 'an'}
            set1 = set(w for w in s1.lower().split() if w not in stopwords and len(w) > 1)
            set2 = set(w for w in s2.lower().split() if w not in stopwords and len(w) > 1)
            if not set1 or not set2: return 0.0
            return len(set1 & set2) / len(set1 | set2)


        G_work = G.copy()
        processed_nodes = set()
        
        # 排序策略优化：优先处理“长”的名字。
        # 为什么？因为长名字通常是“特指”（如 Acute Myocardial Infarction）。
        # 如果先处理短名字（如 Infarction），它容易把长的吸附进来。
        # 让长名字先占山为王，短名字（泛指）就无法吞并它们。
        all_nodes_sorted = sorted(G_work.nodes(), key=lambda n: len(n), reverse=True)
        
        clusters = []
        
        for center_node in all_nodes_sorted:
            if center_node in processed_nodes:
                continue
                
            current_cluster = {center_node}
            processed_nodes.add(center_node)
            
            # 获取直接邻居
            neighbors = list(G_work.neighbors(center_node))
            
            for neighbor in neighbors:
                if neighbor in processed_nodes:
                    continue
                
                edge_data = G_work.get_edge_data(center_node, neighbor)
                edge_type = edge_data.get('type', 'soft')
                score = edge_data.get('weight', 0)
                
                is_safe_merge = False
                
                # --- 1. 硬连接 (Explicit Alias) ---
                if edge_type == 'hard':
                    is_safe_merge = True
                    
                # --- 2. 软连接 (Vector Sim) - 极严苛校验 ---
                else:
                    # [规则 A]: 包含关系阻断 (防泛化)
                    # 例如 "Plaque" in "Atherosclerotic Plaque" -> 拒绝
                    if _is_substring_relation(center_node, neighbor):
                        is_safe_merge = False
                    
                        
                    # [规则 C]: 极高分通过
                    # 只有向量分极高，且没有上述冲突时，才允许
                    elif score > 0.95: 
                        is_safe_merge = True
                        
                    # [规则 D]: 中高分 + 严格 Token 重叠
                    elif score > 0.95: # threshold 建议 0.98
                        jaccard = _token_jaccard(center_node, neighbor)
                        # 要求：向量相似 + 共享至少 60% 的特异性 Token
                        if jaccard > 0.6: 
                            is_safe_merge = True

                if is_safe_merge:
                    current_cluster.add(neighbor)
                    processed_nodes.add(neighbor)
            
            clusters.append(list(current_cluster))

        print(f"Total clusters formed: {len(clusters)}")
        
        final_entities = []
        name_mapping = {} 
        merged_entity_map = {}

        for cluster in clusters:
            cluster_list = list(cluster)
            
            # 确定标准名
            valid_names = [n for n in cluster_list if len(n) >= 5]
            canonical_name = sorted(valid_names, key=len,reverse=True)[0] if valid_names else sorted(cluster_list, key=len)[0]

            cluster_raw_entities = []
            for name_str in cluster_list:
                name_mapping[name_str] = canonical_name
                if name_str in str_to_raw_entities:
                    cluster_raw_entities.extend(str_to_raw_entities[name_str])
            
            unique_raw_entities = list({id(e): e for e in cluster_raw_entities}.values())
            
            # ================= [新增打印逻辑 开始] =================
            # 只有当发生了"合并"行为时才打印，避免刷屏
            # 条件：涉及超过1个实体对象 OR 涉及超过1个不同的名字
            # if len(unique_raw_entities) > 1 or len(cluster_list) > 1:
            #     print(f"\n🔹 [Merge Event] Canonical Name: '{canonical_name}'")
                
            #     # 1. 打印同义词簇的所有名字
            #     print(f"   └── Synonyms/Aliases ({len(cluster_list)}): {cluster_list}")
                
            #     # 2. 打印涉及的原始实体对象
            #     if len(unique_raw_entities) > 1:
            #         print(f"   └── ⚠️ Merging {len(unique_raw_entities)} Distinct Entities:")
            #         for idx, e in enumerate(unique_raw_entities):
            #             print(f"       {idx+1}. ID: {e.entity_id:<10} | Name: {e.name:<20} | Type: {e.entity_type}")
            #     elif len(unique_raw_entities) == 1:
            #         print(f"   └── Single Entity Updated: ID {unique_raw_entities[0].entity_id} ({unique_raw_entities[0].name})")
            #     else:
            #         print(f"   └── No Entity Objects (Pure string merge from Triples)")
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
        if not triples:
            return normalized_entities, normalized_triples
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
    memory_path = '/home/nas3/biod/dongkun/snapshots/memory-20251210-171318.json'
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
        print("\n--- Processing Complete ---")
        print(f"Total Merged Entities: {len(memory.entities.all())}")
        print(f"Total Merged Triples: {len(memory.relations.triples)}")
        
        # if len(triples) > 0:
        #     print("\nExample Triple:")
        #     t = triples[0]
        #     print(f"{t.head} --[{t.relation}]--> {t.tail}")
        #     print(f"Linked Subject Name: {t.subject.name if t.subject else 'None'}")
        #     print(f"Linked Subject Aliases: {t.subject.aliases if t.subject else 'None'}")
        