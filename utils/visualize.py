import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase

from Memory.index import Memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


def visualize_global_kg(mem: Memory, max_edges: int = 5000):
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
    plt.savefig("global_kg.png", dpi=600)
def export_memory_to_neo4j(
    mem: Memory,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
    clear_db: bool = False,
    max_edges: int = 5000,
):
    """
    将 global KG（mem.entities + mem.relations）导入 Neo4j。
    节点标签: :Entity，带属性：
        - entity_id
        - name
        - entity_type
        - normalized_id

    关系类型: :REL，带属性：
        - rel_type  (原来的 relation / rel_type)
        - source    (可选：来源信息)

    导入后在 Neo4j Browser 中可以用：
        MATCH (h:Entity)-[r:REL]->(t:Entity)
        RETURN h,r,t LIMIT 200;
    来进行可视化。
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def get_head_tail_ids(tri: KGTriple):
        # 优先 subject/object 上挂的 KGEntity
        h_ent = tri.subject if isinstance(tri.subject, KGEntity) else None
        t_ent = tri.object if isinstance(tri.object, KGEntity) else None

        # 如果没有 subject/object，就退回到 head/tail 文本，用你的 entity store 里找
        if h_ent is None and getattr(tri, "head", None):
            h_ent = mem.entities.find_by_norm(tri.head)
        if t_ent is None and getattr(tri, "tail", None):
            t_ent = mem.entities.find_by_norm(tri.tail)

        # 某些 pipeline 里你可能已经有 head_id / tail_id
        h_id = getattr(tri, "head_id", None) or (h_ent.entity_id if h_ent else None)
        t_id = getattr(tri, "tail_id", None) or (t_ent.entity_id if t_ent else None)
        return h_id, t_id

    with driver.session() as session:
        # 可选：清空库
        if clear_db:
            session.run("MATCH (n) DETACH DELETE n")

        # 1) 写入节点
        for ent in mem.entities.all():
            session.run(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                SET  e.name          = $name,
                     e.entity_type   = $entity_type,
                     e.normalized_id = $normalized_id
                """,
                {
                    "entity_id": ent.entity_id,
                    "name": ent.name or ent.entity_id,
                    "entity_type": getattr(ent, "entity_type", "Unknown"),
                    "normalized_id": getattr(ent, "normalized_id", "N/A"),
                },
            )

        # 2) 写入关系
        edge_count = 0
        for tri in mem.relations.all():
            if edge_count >= max_edges:
                break

            h_id, t_id = get_head_tail_ids(tri)
            if not h_id or not t_id:
                continue

            rel_type = getattr(tri, "rel_type", None) or getattr(tri, "relation", None)
            source = getattr(tri, "source", "unknown")

            session.run(
                """
                MATCH (h:Entity {entity_id: $h_id})
                MATCH (t:Entity {entity_id: $t_id})
                MERGE (h)-[r:REL {rel_type: $rel_type, source: $source}]->(t)
                """,
                {"h_id": h_id, "t_id": t_id, "rel_type": rel_type, "source": source},
            )
            edge_count += 1

    driver.close()
    print(f"Exported {len(mem.entities.all())} entities and {edge_count} relations to Neo4j.")