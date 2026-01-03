import argparse
import csv
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from openai import OpenAI


SYSTEM_PROMPT = """
You are a biomedical relation classifier.
Given a user question that names two entities, output only the relation label.
Choose labels strictly from the allowed set for the provided relation type:
- Gene & Gene: positive_correlate | negative_correlate | no_relation
- Chemical & Gene: positive_correlate | negative_correlate | no_relation
- Disease & Gene: stimulate | inhibit | no_relation
Respond with JSON: {"label": "..."}
Do not add any extra text or code fences.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-4o relation labels on TSV questions")
    parser.add_argument(
        "--tsv",
        default="Experiment/dataset/edges_test_100.tsv",
        help="Path to the TSV with columns: x_id y_id x_type y_type rel_type relation question",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    return parser.parse_args()


def load_rows(tsv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def build_prompt(rel_type: str, question: str) -> str:
    return json.dumps({"rel_type": rel_type, "question": question}, ensure_ascii=False)


def normalize_label(label: str, rel_type: str) -> str:
    key = (label or "").strip().lower()
    mapping = {
        "positive": "positive_correlate",
        "positive_correlate": "positive_correlate",
        "negative": "negative_correlate",
        "negative_correlate": "negative_correlate",
        "stimulate": "stimulate",
        "inhibit": "inhibit",
        "no_relation": "no_relation",
        "no relation": "no_relation",
        "none": "no_relation",
    }
    norm = mapping.get(key, "")
    if rel_type == "Disease & Gene" and norm in {"positive_correlate", "negative_correlate"}:
        # map to disease labels
        norm = "stimulate" if norm == "positive_correlate" else "inhibit"
    if rel_type != "Disease & Gene" and norm in {"stimulate", "inhibit"}:
        norm = "positive_correlate" if norm == "stimulate" else "negative_correlate"
    return norm


def call_model(client: OpenAI, model: str, rel_type: str, question: str) -> str:
    prompt = build_prompt(rel_type, question)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    try:
        obj = json.loads(text.strip("`"))
        return str(obj.get("label", ""))
    except Exception:
        return text.splitlines()[0].strip()


def f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def safe_div(n, d):
    return n / d if d > 0 else 0.0

def compute_metrics(records: List[Tuple[str, str, str]]) -> Dict[str, float]:
    # 结构: stats[rel_type][class_label] = {tp, fp, fn}
    stats = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))
    
    # 1. 统计 TP, FP, FN
    for rel_type, gold, pred in records:
        # 定义合法标签集合
        if rel_type == "Disease & Gene":
            allowed = {"stimulate", "inhibit", "no_relation"}
        else:
            allowed = {"positive_correlate", "negative_correlate", "no_relation"}
            
        # 过滤非法金标准（可选，视数据质量而定）
        if gold not in allowed:
            continue
            
        # 统计
        if pred == gold:
            stats[rel_type][gold]["tp"] += 1
        else:
            stats[rel_type][gold]["fn"] += 1  # 漏报了 gold
            if pred in allowed:
                stats[rel_type][pred]["fp"] += 1  # 误报了 pred

    metrics = {}
    
    # 2. 计算指标
    for rel_type, class_stats in stats.items():
        macro_f1_sum = 0.0
        valid_classes = 0
        
        # 计算每个类别的 P/R/F1
        for cls, counts in class_stats.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            
            p = safe_div(tp, tp + fp)
            r = safe_div(tp, tp + fn)
            f1 = safe_div(2 * p * r, p + r)
            
            metrics[f"{rel_type}::{cls}_f1"] = f1
            metrics[f"{rel_type}::macro_f1"] = metrics.get(f"{rel_type}::macro_f1", 0.0) + f1
            valid_classes += 1
            
        # 计算该任务的 Macro F1
        if valid_classes > 0:
            metrics[f"{rel_type}::macro_f1"] /= valid_classes

    return metrics

    def pack(counter_map: Dict[Any, Counter]) -> Dict[str, float]:
        metrics = {}
        for (rtype, cls), c in counter_map.items():
            tp = c.get("tp", 0)
            fp = c.get("fp", 0)
            fn = c.get("fn", 0)
            metrics[f"{rtype}::{cls}_f1"] = f1(tp, fp, fn)
        return metrics

    metrics = pack(per_type)
    metrics.update(pack(overall))
    return metrics


def main() -> None:
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    rows = load_rows(args.tsv)

    records: List[Tuple[str, str, str]] = []
    for row in rows:
        rel_type = row["rel_type"]
        gold = row["relation"]
        question = row["question"]
        raw_pred = call_model(client, args.model, rel_type, question)
        pred = normalize_label(raw_pred, rel_type)
        records.append((rel_type, gold, pred))
        print(f"[{rel_type}] gold={gold} pred={pred} raw={raw_pred}")

    metrics = compute_metrics(records)
    print("\nMetrics:")
    for k in sorted(metrics):
        print(f"{k}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
