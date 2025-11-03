#!/usr/bin/env python3
# download_pdfs_with_safe_names.py
"""
Minimal PDF downloader with safe filenames extracted from URLs.

Saves files to outdir (default "downloaded_pdfs") with names like "PMC8954705.pdf"
if the URL contains accid/pmcid/id query params, otherwise use path basename or a hashed fallback.

Usage as script:
    python download_pdfs_with_safe_names.py

Usage as module:
    from download_pdfs_with_safe_names import save_pdfs_from_url_list
    items = [
       "http://...accid=PMC8954705&blobtype=pdf",
       None,
       "http://...accid=PMC8026272&blobtype=pdf"
    ]
    results = save_pdfs_from_url_list(items, outdir="pdfs", overwrite=False)

Returns:
    results: list of dicts with keys:
      - name: chosen safe filename (or None if skipped)
      - url: original url
      - status: "OK"|"SKIP"|"EXISTS"|"FAIL"
      - path_or_msg: local path or message
"""

from typing import List, Optional, Iterable, Dict, Any
import os
import requests
from urllib.parse import urlparse, parse_qs, unquote
import hashlib
import re

# --------- Helpers for safe filename ----------
def _clean_name(s: str, maxlen: int = 200) -> str:
    # 保留字母数字和这些字符，去掉其它危险字符
    safe = "".join(c for c in s if c.isalnum() or c in " .-_()[]")
    safe = safe.strip()
    if len(safe) > maxlen:
        safe = safe[:maxlen]
    return safe or None

def make_safe_filename_from_url(url: Optional[str], fallback_prefix: str = "file", maxlen: int = 200) -> Optional[str]:
    """
    给定 URL，返回以 .pdf 结尾的安全文件名，或当 url 为 None/空时返回 None。
    策略：
      1. url 为 None/空 -> None
      2. 提取 query 中的 accid/id/pmcid/file/filename 等优先作为名字
      3. 否则使用 path basename（保持 .pdf 后缀如有）
      4. 否则使用 URL 的 sha1 哈希作为后备名
    """
    if not url:
        return None

    try:
        parsed = urlparse(url)
    except Exception:
        return None

    qs = parse_qs(parsed.query or "")
    candidates = []
    for key in ("accid", "id", "pmcid", "file", "filename"):
        v = qs.get(key)
        if v:
            candidates.append(v[0])

    for cand in candidates:
        if cand:
            name = unquote(str(cand))
            # remove weird chars
            name = re.sub(r"[^\w\-\.\(\)\[\] ]+", "", name)
            c = _clean_name(name, maxlen=maxlen-4)
            if c:
                if not c.lower().endswith(".pdf"):
                    c = c + ".pdf"
                return c

    # try basename of path
    path = unquote(parsed.path or "")
    base = os.path.basename(path)
    if base:
        base = base.split("?")[0].split("#")[0]
        base_clean = _clean_name(base, maxlen=maxlen)
        if base_clean:
            if base.lower().endswith(".pdf"):
                return base_clean if base_clean.lower().endswith(".pdf") else base_clean + ".pdf"
            else:
                return base_clean + ".pdf"

    # fallback: hash of url
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    name = f"{fallback_prefix}_{h}.pdf"
    return _clean_name(name, maxlen=maxlen)

# --------- Minimal PDF download ----------
def download_pdf(url: str, save_path: str, timeout: int = 20) -> bool:
    """
    下载 URL 如果它指向 PDF 则保存到 save_path（包含 .pdf 扩展名）。
    返回 True 成功，False 失败或不是 PDF。
    """
    if not url or not str(url).strip():
        return False
    try:
        # 先尝试 HEAD（部分站点不支持）
        try:
            head = requests.head(url, allow_redirects=True, timeout=8)
            ctype = head.headers.get("Content-Type", "").lower()
        except Exception:
            ctype = ""

        # 如果 URL 以 .pdf 结尾或 HEAD 表示 PDF，直接 GET 保存
        if url.lower().endswith(".pdf") or "application/pdf" in ctype:
            r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and ("application/pdf" in r.headers.get("Content-Type", "").lower() or url.lower().endswith(".pdf")):
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
                return True
            else:
                return False

        # 否则做 GET 再判断 content-type（有时候 GET 会返回 PDF）
        r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", "").lower():
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            return True

        # 不是 PDF（按你的要求不解析 HTML 获取 PDF 链接）
        return False

    except Exception:
        # 任意异常视为失败
        return False

# --------- Public API ----------
def save_pdfs_from_url_list(urls: Iterable[Optional[str]],
                            outdir: str = "downloaded_pdfs",
                            overwrite: bool = False,
                            timeout: int = 20) -> List[Dict[str, Any]]:
    """
    接受一个 URL 列表（可能包含 None），按顺序处理并尝试下载 PDF。
    返回每项的字典结果：
      { "name": safe_filename or None, "url": original_url, "status": "OK"/"SKIP"/"EXISTS"/"FAIL", "path_or_msg": path_or_message }
    """
    os.makedirs(outdir, exist_ok=True)
    results: List[Dict[str, Any]] = []
    session = requests.Session()

    for url in urls:
        res = {"name": None, "url": url, "status": None, "path_or_msg": None}
        if not url:
            res.update({"status": "SKIP", "path_or_msg": "empty URL or None"})
            results.append(res)
            continue

        safe_name = make_safe_filename_from_url(url)
        if not safe_name:
            # fallback name
            h = hashlib.sha1(str(url).encode("utf-8")).hexdigest()[:12]
            safe_name = f"file_{h}.pdf"

        res["name"] = safe_name
        save_path = os.path.join(outdir, safe_name)

        if os.path.exists(save_path) and not overwrite:
            res.update({"status": "EXISTS", "path_or_msg": save_path})
            results.append(res)
            continue

        # 尝试下载（直接使用 requests）
        ok = download_pdf(url, save_path, timeout=timeout)
        if ok:
            res.update({"status": "OK", "path_or_msg": save_path})
        else:
            # 清理可能存在的残缺文件
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception:
                    pass
            res.update({"status": "FAIL", "path_or_msg": "could not download PDF or not a PDF"})
        results.append(res)

    return results

# --------- CLI 示例 ----------
if __name__ == "__main__":
    # 示例输入：将替换为你真实的 URL 列表
    example_urls = [
        "http://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC8954705&blobtype=pdf",
        None,
        "http://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC8026272&blobtype=pdf",
        "http://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC10069006&blobtype=pdf",
        # 也可以加入更多 URL
    ]

    outdir = "downloaded_pdfs"
    print(f"Start downloading to '{outdir}' ...")
    results = save_pdfs_from_url_list(example_urls, outdir=outdir, overwrite=False, timeout=20)
    for r in results:
        print(f"{r['name']}\t{r['status']}\t{r['path_or_msg']}")
    print("Done.")