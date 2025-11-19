import os
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from metapub import FindIt, PubMedFetcher

from utils.download import save_pdfs_from_url_list
from utils.pdf2md import deepseek_pdf_to_md_batch, DEFAULT_DEEPSEEK_MODEL_DIR
from utils.process_markdown import clean_markdown

fetch = PubMedFetcher()


def _strip_page_markers(md: str) -> str:
    md = re.sub(r"<!--\s*Page\s*\d+\s*-->\s*", "\n\n", md, flags=re.IGNORECASE)
    md = re.sub(r"<!--\s*FILE:.*?-->\s*", "\n\n", md, flags=re.IGNORECASE)
    return md.strip()


def _md_to_paragraph_dicts_for_pmid(md_text: str, pmid: str) -> List[Dict[str, str]]:
    parts = [p.strip() for p in re.split(r"\n\s*\n+", md_text) if p.strip()]
    return [{"id": f"{pmid}_p{i+1}", "text": p} for i, p in enumerate(parts)]


def search_review_pmids(user_query: str, retmax: int = 50) -> List[str]:
    """
    使用 NCBI eutils 搜索 PubMed 中的 review 类型文章，返回 pmid 列表（字符串）。
    """
    term = f"({user_query}) AND review[ptyp]"
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": str(retmax),
        "retmode": "json",
    }
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        idlist = data.get("esearchresult", {}).get("idlist", []) or []
        return idlist
    except Exception:
        return []


def _map_downloads_to_pmids(download_results: List[Dict], pmids: List[str]) -> List[Tuple[str, str]]:
    """
    根据 download_results 的顺序，把成功下载的本地 pdf 路径与对应 pmid 绑定并返回列表 (pdf_path, pmid)
    """
    pairs: List[Tuple[str, str]] = []
    for idx, item in enumerate(download_results):
        pmid = pmids[idx] if idx < len(pmids) else None
        if not pmid:
            continue
        if item.get("status") in {"OK", "EXISTS"} and item.get("path_or_msg"):
            p = Path(item["path_or_msg"])
            if p.is_file() and p.suffix.lower() == ".pdf":
                pairs.append((str(p), pmid))
    return pairs


def pipeline_search_download_ocr_to_paragraphs(
    user_query: str,
    topk: int = 10,
    download_outdir: str = "downloaded_pdfs",
    markdown_outdir: str = "markdown",
    dpi: int = 220,
    cpu: bool = False,
    keep_refs: bool = False,
    model_dir: Optional[str] = None,
) -> dict:
    """
    从用户 query 搜索 review -> 下载 PDF -> OCR 转 Markdown -> 清洗并按段落返回 List[Dict[id,text]]。
    id 格式为 "<pmid>_p<段落号>"。
    返回字典包含中间与最终结果：
      {
        "selected_pmids": [...],
        "selected_urls": [...],
        "download_results": [...],
        "pdf_pmid_pairs": [ (pdf_path, pmid), ... ],
        "md_files": [...],
        "paragraphs_by_pmid": { pmid: [ {id,text}, ... ] },
        "merged": [ {id,text}, ... ]
      }
    """
    # 1) 搜索 review pmids
    pmids = search_review_pmids(user_query, retmax=max(50, topk * 5))
    if not pmids:
        return {
            "selected_pmids": [],
            "selected_urls": [],
            "download_results": [],
            "pdf_pmid_pairs": [],
            "md_files": [],
            "paragraphs_by_pmid": {},
            "merged": [],
        }
    selected_pmids = pmids[:topk]

    # 2) 用 FindIt 获取全文 URL（顺序与 selected_pmids 对齐）
    selected_urls: List[Optional[str]] = []
    for pmid in selected_pmids:
        try:
            url = FindIt(pmid).url
            selected_urls.append(url)
        except Exception:
            selected_urls.append(None)

    # 3) 下载 PDFs（保持顺序）
    os.makedirs(download_outdir, exist_ok=True)
    download_results = save_pdfs_from_url_list(
        selected_urls,
        outdir=download_outdir,
        overwrite=False,
        timeout=20,
    )

    # 4) 将下载结果映射回 pmid 并收集成功的 pdf 路径
    pdf_pmid_pairs = _map_downloads_to_pmids(download_results, selected_pmids)
    if not pdf_pmid_pairs:
        return {
            "selected_pmids": selected_pmids,
            "selected_urls": selected_urls,
            "download_results": download_results,
            "pdf_pmid_pairs": [],
            "md_files": [],
            "paragraphs_by_pmid": {},
            "merged": [],
        }

    pdf_paths = [p for p, _ in pdf_pmid_pairs]

    # 5) OCR -> Markdown（批量，模型只加载一次）
    markdown_dir = Path(markdown_outdir)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    md_outputs = deepseek_pdf_to_md_batch(
        pdf_paths=pdf_paths,
        out_dir=str(markdown_dir),
        first_page=1,
        last_page=None,
        dpi=dpi,
        keep_refs=keep_refs,
        cpu=cpu,
        model_dir=model_dir if model_dir else DEFAULT_DEEPSEEK_MODEL_DIR,
        combine_out=None,
    )

    paragraphs_by_pmid: Dict[str, List[Dict[str, str]]] = {}
    merged: List[Dict[str, str]] = []

    # md_outputs 顺序与 pdf_paths 保持一致，对应 pdf_pmid_pairs 顺序
    for idx, md_path in enumerate(md_outputs):
        pmid = pdf_pmid_pairs[idx][1] if idx < len(pdf_pmid_pairs) else None
        try:
            raw_md = Path(md_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raw_md = ""
        raw_md = _strip_page_markers(raw_md)
        cleaned_md, _ = clean_markdown(raw_md, tail_portion=0.30, min_ref_block=10, min_keep_ratio=0.60)
        if pmid:
            para_dicts = _md_to_paragraph_dicts_for_pmid(cleaned_md, pmid)
        else:
            # fallback 使用文件 stem 做 id 前缀（极不常见）
            stem = Path(md_path).stem
            para_dicts = _md_to_paragraph_dicts_for_pmid(cleaned_md, stem)
        paragraphs_by_pmid[pmid or Path(md_path).stem] = para_dicts
        merged.extend(para_dicts)

        # 写出每篇 json/txt
        out_json = markdown_dir / f"{Path(md_path).stem}.paragraphs.json"
        out_txt = markdown_dir / f"{Path(md_path).stem}.paragraphs.txt"
        out_json.write_text(json.dumps(para_dicts, ensure_ascii=False, indent=2), encoding="utf-8")
        out_txt.write_text("\n\n".join(p["text"] for p in para_dicts), encoding="utf-8")

    # 写合并文件
    merged_json = markdown_dir / "all_paragraphs.json"
    merged_txt = markdown_dir / "all_paragraphs.txt"
    merged_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    merged_txt.write_text("\n\n".join(p["text"] for p in merged), encoding="utf-8")

    return {
        "selected_pmids": selected_pmids,
        "selected_urls": selected_urls,
        "download_results": download_results,
        "pdf_pmid_pairs": pdf_pmid_pairs,
        "md_files": md_outputs,
        "paragraphs_by_pmid": paragraphs_by_pmid,
        "merged": merged,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="query -> search reviews -> download -> OCR -> paragraphs (id=pmid_p#)")
    ap.add_argument("query", help="检索查询文本")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out-pdf-dir", default="downloaded_pdfs")
    ap.add_argument("--out-md-dir", default="markdown")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--keep-refs", action="store_true")
    args = ap.parse_args()

    res = pipeline_search_download_ocr_to_paragraphs(
        user_query=args.query,
        topk=args.topk,
        download_outdir=args.out_pdf_dir,
        markdown_outdir=args.out_md_dir,
        dpi=args.dpi,
        cpu=args.cpu,
        keep_refs=args.keep_refs,
        model_dir=None,
    )
    print("Pipeline 完成，输出目录:", Path(args.out_md_dir).resolve())
    print("Selected PMIDs:", res["selected_pmids"])