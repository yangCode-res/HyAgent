import networkx as nx
import matplotlib.pyplot as plt
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from Memory.index import Memory


def visualize_global_kg(mem: Memory, max_edges: int = 500):
    """
    可视化 global KG（mem.entities + mem.relations）
    - 节点：global 实体
    - 边：global 三元组（最多 max_edges 条）
    """

    G = nx.MultiDiGraph()

    # 1) 加所有全局实体作为节点
    for ent in mem.entities.all():
        G.add_node(
            ent.entity_id,
            label=ent.name or ent.entity_id,
            etype=getattr(ent, "entity_type", "Unknown"),
        )

    # 小工具：从 triple 里拿到 head/tail 的全局 entity_id
    def get_head_tail_ids(tri: KGTriple):
        # 优先 subject / object 里的 KGEntity
        h_ent = tri.subject if isinstance(tri.subject, KGEntity) else None
        t_ent = tri.object if isinstance(tri.object, KGEntity) else None

        if h_ent is None and tri.head:
            h_ent = mem.entities.find_by_norm(tri.head)
        if t_ent is None and tri.tail:
            t_ent = mem.entities.find_by_norm(tri.tail)

        h_id = h_ent.entity_id if h_ent else None
        t_id = t_ent.entity_id if t_ent else None
        return h_id, t_id

    # 2) 加边（关系）
    edge_count = 0
    for tri in mem.relations.all():
        if edge_count >= max_edges:
            break
        h_id, t_id = get_head_tail_ids(tri)
        if not h_id or not t_id:
            continue
        if h_id not in G or t_id not in G:
            continue
        G.add_edge(h_id, t_id, label=tri.relation)
        edge_count += 1

    # 3) 画图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.4, iterations=50)

    # 节点
    nx.draw_networkx_nodes(G, pos, node_size=300)
    # 节点标签用实体名
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # 边
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", width=0.5)
    # 边标签用 relation
    edge_labels = {(u, v, k): d["label"] for u, v, k, d in G.edges(keys=True, data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig("global_kg.png", dpi=300)