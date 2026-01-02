import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from Core.Agent import Agent
from Memory.index import Memory
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


class SubgraphMerger(Agent):
    """
    子图合并 Agent。

    功能：
    - 将 Memory.subgraphs 里的所有子图，合并到 Memory.entities / Memory.relations
    - 利用 Memory.alignments 里的实体对齐信息，先合并“对齐实体”，再合并未对齐实体
    - 在合并过程中维护：
        * local2global: (subgraph_id, local_entity_id) -> global_entity_id
        * global2local: global_entity_id -> Set[(subgraph_id, local_entity_id)]
      但这两个映射**不会直接挂在 memory 上**，
      而是写入一个 .pkl 文件，
      然后把文件路径记录在 memory.entity_id_mapping_path 里。
    """

    def __init__(self, client: OpenAI, model_name: str, memory: Optional[Memory] = None):
        super().__init__(client, model_name, system_prompt="")

        # 只在本次合并流程中使用的映射
        self.local2global: Dict[Tuple[str, str], str] = {}
        self.global2local: Dict[str, Set[Tuple[str, str]]] = {}

        self.memory: Memory = memory or get_memory()
        self.client = client
        self.model_name = model_name

    # ------------ 工具：映射登记 ------------

    def _register_mapping(self, sg_id: str, local_eid: str, global_eid: str) -> None:
        """
        把一条 (子图, 本地实体id) <-> 全局实体id 的映射记录到
        - self.local2global
        - self.global2local
        """
        if not sg_id or not local_eid or not global_eid:
            return

        key = (sg_id, local_eid)
        self.local2global[key] = global_eid

        if global_eid not in self.global2local:
            self.global2local[global_eid] = set()
        self.global2local[global_eid].add(key)

    # ------------ 工具：从子图拿实体 ------------

    def _get_entity(self, sg_id: str, ent_id: str) -> Optional[KGEntity]:
        sg = self.memory.subgraphs.get(sg_id)
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

    # ------------ 步骤 1：合并“对齐实体” ------------

    def _merge_alignments(self):
        """
        使用 memory.alignments.by_source 里的对齐结果，
        把“同一簇”的实体合并到同一个全局实体上，
        并记录 local <-> global 映射。
        """
        for (src_sg, src_eid), aligns in self.memory.alignments.by_source.items():
            # 源实体
            src_ent = self._get_entity(src_sg, src_eid)
            if src_ent is None:
                continue

            # 源实体是否已经有全局实体
            if (src_sg, src_eid) in self.local2global:
                gid = self.local2global[(src_sg, src_eid)]
                base = self.memory.entities.by_id[gid]
            else:
                # 否则 upsert 到全局实体库（会得到一个新的 global entity_id）
                base = self.memory.entities.upsert(KGEntity(**src_ent.to_dict()))
                gid = base.entity_id
                self._register_mapping(src_sg, src_eid, gid)

            # 处理与之对齐的目标实体
            for al in aligns:
                tgt_key = (al.tgt_subgraph, al.tgt_entity)

                if tgt_key in self.local2global:
                    # 已经归到某个全局实体里了
                    continue

                tgt_ent = self._get_entity(al.tgt_subgraph, al.tgt_entity)
                if tgt_ent is None:
                    continue

                # 把目标实体的属性 merge 到 base（不改变 base.entity_id）
                self.memory.entities._merge(base, KGEntity(**tgt_ent.to_dict()))
                # 记录：这个目标实体也对应 gid
                self._register_mapping(al.tgt_subgraph, al.tgt_entity, gid)

    # ------------ 步骤 2：未对齐实体 upsert ------------

    def _merge_unaligned_entities(self):
        """
        遍历所有子图实体：
        - 对于没有出现在 local2global 里的实体，直接 upsert 到全局实体库
        - 并记录 (sg_id, local_id) -> global_id
        """
        for sg_id, sg in self.memory.subgraphs.items():
            for e in sg.entities.all():
                key = (sg_id, e.entity_id)
                if key in self.local2global:
                    # 对齐阶段已经处理过
                    continue

                g = self.memory.entities.upsert(KGEntity(**e.to_dict()))
                self._register_mapping(sg_id, e.entity_id, g.entity_id)

    # ------------ 步骤 3：合并关系 ------------

    def _merge_relations(self):
        """
        遍历所有子图关系，将其中的 subject/object 映射到全局实体，
        然后写入 memory.relations。
        """
        for sg_id, sg in self.memory.subgraphs.items():
            for r in sg.relations.all():
                # subject / object 标准化成 KGEntity
                subj = self._ensure_entity(r.subject)
                obj = self._ensure_entity(r.object)

                sid = subj.entity_id if subj else None
                oid = obj.entity_id if obj else None

                g_subj = subj
                g_obj = obj

                # 通过 local2global 去找对应的全局实体
                if sid is not None:
                    gid = self.local2global.get((sg_id, sid))
                    if gid:
                        g_subj = self.memory.entities.by_id.get(gid, g_subj)

                if oid is not None:
                    gid = self.local2global.get((sg_id, oid))
                    if gid:
                        g_obj = self.memory.entities.by_id.get(gid, g_obj)

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
                self.memory.relations.add(new_triple)

    # ------------ 步骤 4：把映射写到 pkl 文件 ------------

    def _dump_mappings_to_pkl(self) -> str:
        """
        把 local2global / global2local 写入一个 .pkl 文件，并返回文件路径。

        结构大致是：
        {
          "local2global": {
            (subgraph_id, local_eid): global_eid,
            ...
          },
          "global2local": {
            global_eid: [
              (subgraph_id, local_eid),
              ...
            ],
            ...
          }
        }
        """
        # 你可以把这个路径改成你项目习惯的 cache 目录
        base_dir = Path("cache")
        base_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"entity_id_mapping_{ts}.pkl"
        fpath = base_dir / filename

        # global2local 里的 value 是 set，需要先转成 list，方便序列化和后续使用
        data = {
            "local2global": dict(self.local2global),
            "global2local": {
                gid: list(pairs) for gid, pairs in self.global2local.items()
            },
        }

        with open(fpath, "wb") as f:
            pickle.dump(data, f)

        return str(fpath)

    # ------------ 对外入口 ------------

    def process(self):
        """
        1. 初始化本次合并用的映射
        2. 合并对齐实体
        3. 合并未对齐实体
        4. 合并关系
        5. 把映射写入 pkl，并把路径挂到 memory 上
        """
        self.local2global = {}
        self.global2local = {}

        self._merge_alignments()
        self._merge_unaligned_entities()
        self._merge_relations()

        # 写 pkl，并把路径挂到 memory 上
        mapping_path = self._dump_mappings_to_pkl()

        # Memory 是普通 Python 类，可以动态挂属性
        # 之后你就可以通过 memory.entity_id_mapping_path 拿到这个路径
        self.memory.entity_id_mapping_path = mapping_path