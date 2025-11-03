#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safer cleaner for OCR-ed Markdown:
- 移除版心残留(Author Manuscript/HHS Public Access等) —— 仅删单行
- 仅在文末 tail_portion 范围内裁剪 References / 致谢 等块
- 若无显式标题，只在文末检测“明显的参考文献编号块”再裁
- 设置 min_keep_ratio 保险，不在文件头部过早裁切
"""
import argparse, re
from pathlib import Path

# —— 单行垃圾（ anywhere 删除该行，不扩展为块 ）——
SINGLE_LINE_JUNK = [
    "author manuscript", "hhs public access",
    "this article has been accepted for publication",
    "for peer review", "not copyedited", "all rights reserved"
]

# —— 可能作为“文末标题”的关键词（仅在尾部触发）——
TAIL_HEAD_KEYWORDS = [
    # 参考文献
    "references", "bibliography", "参考文献",
    # 致谢/资助/贡献/利益冲突
    "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments", "致谢",
    "funding", "资助",
    "author contributions", "贡献声明", "贡献",
    "competing interests", "conflicts of interest", "利益冲突",
]

# —— 参考文献常见编号样式 —— 
REF_NUMBER_PATTERNS = [
    re.compile(r'^\s*\d{1,4}\.\s+'),     # "12. ..."
    re.compile(r'^\s*\[\d{1,4}\]\s+'),   # "[12] ..."
    re.compile(r'^\s*\d{1,4}\)\s+'),     # "12) ..."
]

# —— 参考文献常见“内容特征”词（放宽判断）——
BIB_TOKENS = [
    "doi", "https://", "http://", "et al", "vol.", "pp.", "pages",
    "journal", "conference", "proc.", "issue", "no.", "issn", "pmid",
    "(19", "(20",  # 年份括号开头，如 (2018)
]

HEADING_RE = re.compile(r'^\s*(#{1,6})\s*(.+?)\s*$')

def norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())

def is_ref_number_line(line: str) -> bool:
    return any(pat.match(line) for pat in REF_NUMBER_PATTERNS)

def looks_like_bib_line(line: str) -> bool:
    l = norm(line)
    if is_ref_number_line(line):
        return True
    # 有常见 bib token
    return any(tok in l for tok in BIB_TOKENS)

def remove_single_line_junk(lines):
    junk = set(SINGLE_LINE_JUNK)
    out = []
    for ln in lines:
        if norm(ln) in junk:
            continue
        out.append(ln)
    return out

def find_tail_heading_cut(lines, tail_portion=0.30):
    """
    在文末 tail_portion 内寻找“显式标题”触发的裁切点。
    仅当标题文本匹配 TAIL_HEAD_KEYWORDS 时生效。
    返回 index 或 None
    """
    n = len(lines)
    start = int(n * (1 - tail_portion))
    start = max(0, min(start, n-1))
    for i in range(start, n):
        m = HEADING_RE.match(lines[i])
        if m:
            title = norm(m.group(2))
            if any(kw in title for kw in TAIL_HEAD_KEYWORDS):
                return i
        else:
            # 纯文本标题（只在尾部触发，且下一行看起来像参考文献）
            t = norm(lines[i])
            if t in [ "references", "bibliography", "参考文献",
                      "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments", "致谢",
                      "funding", "资助",
                      "author contributions", "贡献", "贡献声明",
                      "competing interests", "conflicts of interest", "利益冲突" ]:
                # 需再确认下方若干行确实像参考文献/致谢而非正文
                window = "".join(lines[i+1:i+10])
                if any(looks_like_bib_line(x) for x in window.splitlines()):
                    return i
    return None

def find_tail_numeric_refs_cut(lines, tail_portion=0.30, min_ref_block=10):
    """
    在文末 tail_portion 中，若检测到“明显的编号式参考文献块”，返回起始 index。
    判定：从某行起，后续 ~20-40 行里，满足 looks_like_bib_line 的行数 >= min_ref_block
    """
    n = len(lines)
    start = int(n * (1 - tail_portion))
    start = max(0, min(start, n-1))
    for i in range(start, n):
        # 起点必须像 bib 行，避免把普通列表当成参考文献
        if not looks_like_bib_line(lines[i]): 
            continue
        window = lines[i:i+40]
        score = sum(1 for ln in window if looks_like_bib_line(ln))
        if score >= min_ref_block:
            return i
    return None

def clean_markdown(
    text: str,
    tail_portion: float = 0.30,
    min_ref_block: int = 10,
    min_keep_ratio: float = 0.60
) -> tuple[str, dict]:
    """
    返回 (clean_text, info)
    - 仅在尾部区域触发裁切；保留至少 min_keep_ratio 的正文
    """
    lines = text.splitlines(True)  # 保留换行
    n0 = len(lines)

    # 先清除版心垃圾（单行删除）
    lines = remove_single_line_junk(lines)

    # 尝试基于“显式尾部标题”的裁切
    cut_idx = find_tail_heading_cut(lines, tail_portion=tail_portion)

    # 若无标题，再尝试“尾部编号参考文献”裁切
    if cut_idx is None:
        cut_idx = find_tail_numeric_refs_cut(
            lines, tail_portion=tail_portion, min_ref_block=min_ref_block
        )

    # 安全阈：最早裁切不得早于全文 min_keep_ratio
    min_cut = int(len(lines) * min_keep_ratio)
    did_cut = False
    reason = None

    if cut_idx is not None and cut_idx >= min_cut:
        lines = lines[:cut_idx]
        did_cut = True
        reason = "tail-heading" if HEADING_RE.match(lines[cut_idx-1] if cut_idx-1 >=0 else "") else "tail-numeric"
    else:
        # 没裁，或裁切点太早，放弃裁切
        pass

    return "".join(lines), {
        "original_lines": n0,
        "final_lines": len(lines),
        "did_cut": did_cut,
        "reason": reason,
        "min_cut_guard": min_cut
    }

def main():
    ap = argparse.ArgumentParser(description="Safely remove references/acknowledgements from OCR-ed Markdown (conservative).")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="Output file (default: *_clean.md)")
    ap.add_argument("--tail-portion", type=float, default=0.30, help="Only scan last portion of file (default 0.30)")
    ap.add_argument("--min-ref-block", type=int, default=10, help="Min bib-like lines in window to trigger cut (default 10)")
    ap.add_argument("--min-keep-ratio", type=float, default=0.60, help="Never cut earlier than this ratio (default 0.60)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write; only report")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    raw = inp.read_text(encoding="utf-8", errors="ignore")
    cleaned, info = clean_markdown(
        raw,
        tail_portion=args.tail_portion,
        min_ref_block=args.min_ref_block,
        min_keep_ratio=args.min_keep_ratio
    )

    print(f"[INFO] original_lines={info['original_lines']} final_lines={info['final_lines']} did_cut={info['did_cut']} reason={info['reason']} guard_min_cut={info['min_cut_guard']}")
    if args.dry_run:
        print("[DRY-RUN] Not writing output. Use without --dry-run after you confirm.")
        return

    outp = Path(args.output) if args.output else inp.with_name(inp.stem + "_clean.md")
    outp.write_text(cleaned, encoding="utf-8")
    print(f"✅ Saved: {outp}")

if __name__ == "__main__":
    main()