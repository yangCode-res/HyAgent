from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from Core.Agent import Agent
from Memory.index import Memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from Store.index import get_memory
"""
子图合并 Agent。
将内存中的多个子图合并为一个全局知识图，处理实体对齐和关系映射。
输入: 无（从内存中获取子图和对齐信息）
输出: 无（将合并后的实体和关系存储到内存的全局知识图中:EntityStore和RelationStore）
调用入口：agent.process()
"""

class SubgraphMerger(Agent):
    def __init__(self,client:OpenAI,model_name:str,memory:Memory):
        # (subgraph_id, local_entity_id) -> global_entity_id
        self.local2global: Dict[Tuple[str, str], str] = {}
        # self.memory=get_memory()
        self.memory=memory or get_memory()
        self.client=client
        self.model_name=model_name
        super().__init__(client,model_name,"")
    # ------- 工具方法 -------

    def _get_entity(self, mem: Memory, sg_id: str, ent_id: str) -> Optional[KGEntity]:
        sg = mem.subgraphs.get(sg_id)
        if sg is None:
            return None
        return sg.entities.by_id.get(ent_id)

    def _ensure_entity(self, x: Any) -> Optional[KGEntity]:
        """把 subject/object 统一转成 KGEntity；否则返回 None。"""
        if x is None:
            return None
        if isinstance(x, KGEntity):
            return x
        if isinstance(x, dict):
            from_dict = getattr(KGEntity, "from_dict", None)
            if callable(from_dict):
                return from_dict(x)
            return KGEntity(**x)
        return None

    # ------- 步骤 1：合并对齐实体 -------

    def _merge_alignments(self, mem: Memory):
        for (src_sg, src_eid), aligns in mem.alignments.by_source.items():
            src_ent = self._get_entity(mem, src_sg, src_eid)
            if src_ent is None:
                continue

            if (src_sg, src_eid) in self.local2global:
                gid = self.local2global[(src_sg, src_eid)]
                base = mem.entities.by_id[gid]
            else:
                base = mem.entities.upsert(KGEntity(**src_ent.to_dict()))
                gid = base.entity_id
                self.local2global[(src_sg, src_eid)] = gid

            for al in aligns:
                key = (al.tgt_subgraph, al.tgt_entity)
                if key in self.local2global:
                    continue
                tgt_ent = self._get_entity(mem, al.tgt_subgraph, al.tgt_entity)
                if tgt_ent is None:
                    continue
                mem.entities._merge(base, KGEntity(**tgt_ent.to_dict()))
                self.local2global[key] = gid

    # ------- 步骤 2：未对齐实体走正常 upsert -------

    def _merge_unaligned_entities(self, mem: Memory):
        for sg_id, sg in mem.subgraphs.items():
            for e in sg.entities.all():
                key = (sg_id, e.entity_id)
                if key in self.local2global:
                    continue
                g = mem.entities.upsert(KGEntity(**e.to_dict()))
                self.local2global[key] = g.entity_id

    # ------- 步骤 3：合并关系，并映射到全局实体 -------

    def _merge_relations(self, mem: Memory):
        for sg_id, sg in mem.subgraphs.items():
            for r in sg.relations.all():
                # 统一把 subject/object 转成 KGEntity
                subj = self._ensure_entity(r.subject)
                obj = self._ensure_entity(r.object)

                sid = subj.entity_id if subj else None
                oid = obj.entity_id if obj else None

                g_subj = subj
                g_obj = obj

                if sid is not None:
                    gid = self.local2global.get((sg_id, sid))
                    if gid:
                        g_subj = mem.entities.by_id[gid]

                if oid is not None:
                    gid = self.local2global.get((sg_id, oid))
                    if gid:
                        g_obj = mem.entities.by_id[gid]

                new_triple = KGTriple(
                    head=g_subj.name if g_subj else r.head,
                    relation=r.relation,
                    tail=g_obj.name if g_obj else r.tail,
                    confidence=r.confidence,
                    evidence=r.evidence,
                    mechanism=r.mechanism,
                    source=r.source,
                    subject=g_subj,
                    object=g_obj,
                    time_info=r.time_info,
                )
                mem.relations.add(new_triple)

    # ------- 对外入口 -------

    def process(self,memory:Optional[Memory]):
        mem=memory or self.memory
        print("this is memory",mem)
        print("this is memory.subgraphs",mem.subgraphs.items())
        self.local2global = {}
        self._merge_alignments(mem)
        self._merge_unaligned_entities(mem)
        self._merge_relations(mem)