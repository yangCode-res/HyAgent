import os
import csv
import hashlib
from pathlib import Path
from typing import List, Optional, Dict

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from Agents.Query_clarify.index import QueryClarifyAgent
from Agents.Review_fetcher.index import ReviewFetcherAgent
from Agents.Entity_extraction.index import EntityExtractionAgent
from Agents.Entity_normalize.index import EntityNormalizationAgent
from Agents.Relationship_extraction.index import RelationshipExtractionAgent
from Agents.Collaborate_extraction.index import CollaborationExtractionAgent
from Agents.Causal_extraction.index import CausalExtractionAgent
from Agents.Alignment_triple.index import AlignmentTripleAgent
from Agents.Fusion_subgraph.index import SubgraphMerger

from Memory.index import Subgraph
from Store.index import reset_memory, get_memory
from utils.process_markdown import split_md_by_mixed_count


# --------- Helpers ---------

def _slug(s: str, length: int = 12) -> str:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:length]


def ensure_exp_dirs(base: Path, exp_id: str) -> Dict[str, Path]:
    exp_dir = base / exp_id
    reviews_dir = exp_dir / "reviews"
    graphs_dir = exp_dir / "graphs"
    meta_dir = exp_dir / "meta"
    for d in [exp_dir, reviews_dir, graphs_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return {
        "exp_dir": exp_dir,
        "reviews_dir": reviews_dir,
        "graphs_dir": graphs_dir,
        "meta_dir": meta_dir,
    }


def export_kg_json(output_path: Path) -> None:
    mem = get_memory()
    entities = [e.to_dict() for e in mem.entities.all()]
    relations = [r.to_dict() for r in mem.relations.all()]
    data = {"entities": entities, "relations": relations}
    output_path.write_text(__import__("json").dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# --------- Phase 1: Literature Review ---------

def phase1_literature(client: OpenAI, model_name: str, query: str, exp_id: Optional[str] = None,
                      base_dir: str = "Experiment") -> Dict[str, str]:
    """
    Phase 1: Clarify query, fetch reviews, save markdowns under Experiment/<exp_id>/reviews,
    and persist a memory snapshot under Experiment/<exp_id>/meta.
    Returns dict with paths for downstream use.
    """
    base = Path(base_dir)
    exp_id = exp_id or _slug(query)
    dirs = ensure_exp_dirs(base, exp_id)

    # Clarify query
    qc = QueryClarifyAgent(client, model_name)
    qc_resp = qc.process(query)
    clarified = qc_resp.get("clarified_question", query)

    # Save clarified query for traceability
    (dirs["meta_dir"] / "clarified_query.txt").write_text(clarified, encoding="utf-8")

    # Fetch and OCR → Markdown into the experiment folder
    rf = ReviewFetcherAgent(client, model_name)
    md_paths: List[str] = rf.process(clarified, save_dir=str(dirs["reviews_dir"]))  # type: ignore

    # Snapshot memory that already contains registered subgraphs from RF (if any)
    mem_path = get_memory().dump_json(str(dirs["meta_dir"]))

    return {
        "exp_id": exp_id,
        "clarified_query_path": str(dirs["meta_dir"] / "clarified_query.txt"),
        "memory_snapshot": mem_path,
        "reviews_dir": str(dirs["reviews_dir"]),
        "graphs_dir": str(dirs["graphs_dir"]),
    }


# --------- Phase 2: Graph Construction ---------

def _register_md_as_subgraphs(reviews_dir: Path) -> int:
    """Read markdown files and register paragraph chunks as Subgraphs in memory."""
    mem = get_memory()
    count = 0
    for md in sorted(reviews_dir.glob("*.md")):
        parts = split_md_by_mixed_count(str(md))
        for section_id, chunks in parts.items():
            for i, content in enumerate(chunks):
                sg_id = f"{md.stem}_{section_id}_{i}"
                s = Subgraph(subgraph_id=sg_id, meta={"text": content, "source": md.name})
                mem.register_subgraph(s)
                count += 1
    return count


def phase2_build_graph(client: OpenAI, model_name: str, exp_paths: Dict[str, str],
                       include_causal: bool = True, do_alignment: bool = True,
                       base_dir: str = "Experiment") -> Dict[str, str]:
    """
    Phase 2: Build KG from saved markdowns and persist artifacts under Experiment/<exp_id>/graphs.
    """
    # Fresh memory to keep experiments isolated
    reset_memory()

    reviews_dir = Path(exp_paths["reviews_dir"]) if "reviews_dir" in exp_paths else None
    if reviews_dir is None or not reviews_dir.exists():
        raise FileNotFoundError("reviews_dir not found in exp_paths")

    # Register MD → Subgraphs
    _ = _register_md_as_subgraphs(reviews_dir)

    # Run graph-construction agents
    pipeline: List[object] = [
        EntityExtractionAgent(client, model_name),
        EntityNormalizationAgent(client, model_name),
        RelationshipExtractionAgent(client, model_name),
        CollaborationExtractionAgent(client, model_name),
    ]
    if include_causal:
        pipeline.append(CausalExtractionAgent(client, model_name))
    if do_alignment:
        pipeline.append(AlignmentTripleAgent(client, model_name))
    pipeline.append(SubgraphMerger(client, model_name))

    for agent in pipeline:
        print(f"[Phase2] Running agent: {agent.__class__.__name__}")
        agent.process()

    # Export artifacts
    graphs_dir = Path(exp_paths["graphs_dir"]) if "graphs_dir" in exp_paths else None
    graphs_dir.mkdir(parents=True, exist_ok=True)

    kg_path = graphs_dir / "knowledge_graph.json"
    export_kg_json(kg_path)

    mem_snapshot = get_memory().dump_json(str(graphs_dir))

    return {
        "kg_path": str(kg_path),
        "memory_snapshot": mem_snapshot,
    }


# --------- Batch Driver (optional) ---------

def run_batch_from_tsv(tsv_path: str, client: OpenAI, model_name: str,
                       base_dir: str = "Experiment", limit: Optional[int] = None,
                       question_col: str = "question") -> List[Dict[str, str]]:
    """Batch run: Phase1 then Phase2 for each row in a TSV with a 'question' column."""
    results: List[Dict[str, str]] = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            query = row.get(question_col) or ""
            if not query:
                continue
            exp_id = f"exp_{idx+1}_{_slug(query)}"
            print(f"\n=== Running experiment {exp_id} ===")
            p1 = phase1_literature(client, model_name, query, exp_id=exp_id, base_dir=base_dir)
            p2 = phase2_build_graph(client, model_name, p1, include_causal=True, do_alignment=True, base_dir=base_dir)
            results.append({"exp_id": exp_id, **p1, **p2})
    return results


if __name__ == "__main__":
    # Minimal CLI entry for quick try
    try:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        pass

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE_URL")
    model_name = os.environ.get("OPENAI_MODEL")
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Example: run first 2 questions from dataset TSV
    tsv = os.environ.get("EXPERIMENT_TSV", "Experiment/dataset/edges_test_100.tsv")
    run_batch_from_tsv(tsv, client, model_name, limit=2)
