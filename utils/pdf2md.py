#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-OCR: PDF -> Markdown（逐页 OCR 合并）
- 纯本地加载（local_files_only=True）
- GPU/CPU 自适应（3090 优先用 FP16，支持 BF16 则用 BF16）
- 可选去掉参考文献/致谢/附录等尾部内容
依赖：pip install pymupdf pillow transformers torch  （torch 安装 GPU 版）
"""

import os
import re
import argparse
import tempfile
import shutil
from typing import List, Tuple

import fitz                     # PyMuPDF
import torch
from transformers import AutoModel, AutoTokenizer
# 增量检测“尾部”标题（用于流式写入时提前停止）
BACK_MATTER_RE = re.compile(
    r"\n#{1,6}\s*(references|acknowledg(e)?ments?|funding|conflicts?\s+of\s+interest|author\s+contributions?|ethics|appendix|supplementary)\b",
    re.IGNORECASE
)

def choose_device_dtype(force_cpu: bool = False) -> Tuple[torch.device, torch.dtype]:
    """选择设备与精度：优先 BF16，其次 FP16，最后 FP32。"""
    if (not force_cpu) and torch.cuda.is_available():
        dev = torch.device("cuda")
        # A100/H100 等支持 BF16；3090 一般用 FP16 更稳
        if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return dev, torch.bfloat16
        return dev, torch.float16
    return torch.device("cpu"), torch.float32


def pdf_to_images(pdf_path: str, dpi: int = 220, first_page: int = 1, last_page: int = None) -> Tuple[str, List[str]]:
    """把 PDF 按页渲染为 PNG，返回临时目录与图片路径列表。"""
    tmpdir = tempfile.mkdtemp(prefix="deepseek_ocr_")
    doc = fitz.open(pdf_path)

    if last_page is None:
        last_page = len(doc)

    first_page = max(1, first_page)
    last_page = min(last_page, len(doc))
    if first_page > last_page:
        raise ValueError(f"Invalid page range: {first_page}..{last_page}")

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    out_paths: List[str] = []
    for i in range(first_page - 1, last_page):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out = os.path.join(tmpdir, f"page_{i+1:04d}.png")
        pix.save(out)
        out_paths.append(out)

    return tmpdir, out_paths


def strip_back_matter(md: str) -> str:
    """
    粗粒度删除文末：References / Acknowledgements / Funding / COI / Appendix / Supplementary 等。
    - 优先匹配 markdown 标题形式
    - 兜底匹配“references”尾部块
    """
    patterns = [
        r"\n#{1,6}\s*references\b.*",
        r"\n#{1,6}\s*acknowledg(e)?ments?\b.*",
        r"\n#{1,6}\s*funding\b.*",
        r"\n#{1,6}\s*conflicts?\s+of\s+interest\b.*",
        r"\n#{1,6}\s*author\s+contributions?\b.*",
        r"\n#{1,6}\s*ethics\b.*",
        r"\n#{1,6}\s*appendix\b.*",
        r"\n#{1,6}\s*supplementary\b.*",
    ]
    text = md
    for pat in patterns:
        text = re.sub(pat, "\n", text, flags=re.IGNORECASE | re.DOTALL)

    tail = re.search(r"(references\s*[:\-]?\s*\n.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if tail:
        text = text[:tail.start()]
    return text.strip()


def load_deepseek(model_path: str, device: torch.device, dtype: torch.dtype):
    """
    纯本地加载 DeepSeek-OCR。需要仓库包含自定义代码（trust_remote_code=True）。
    """
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
        local_files_only=True,
    ).eval()
    model.to(device=device, dtype=dtype)
    return tok, model


from PIL import Image
import tempfile

from PIL import Image
import json
import time

from PIL import Image
import json, io, time
from contextlib import redirect_stdout

def _read_text_like(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    if path.lower().endswith(".json"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            for k in ("markdown", "text", "md", "output"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        except Exception:
            pass
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def _newest_text_file(root: str) -> str:
    newest_path, newest_mtime = None, -1.0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.lower().endswith((".md", ".txt", ".json")):
                continue
            p = os.path.join(dirpath, fn)
            try:
                m = os.path.getmtime(p)
                if m > newest_mtime:
                    newest_mtime, newest_path = m, p
            except Exception:
                continue
    return _read_text_like(newest_path) if newest_path else ""

def _clean_stdout(s: str) -> str:
    # 去掉 <|ref|> 标签、det 坐标、分隔线和显存日志，只保留正文行
    lines = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t:
            lines.append("")  # 保留段落空行
            continue
        if t.startswith("<|") and t.endswith("|>"):
            continue
        if t.startswith("<|ref|>") or t.startswith("<|det|>"):
            continue
        if t.startswith("====") or t.startswith("BASE:") or t.startswith("PATCHES:"):
            continue
        if t.lower().startswith(("image size:", "valid image tokens", "output texts tokens", "compression ratio")):
            continue
        lines.append(ln)
    # 连续空行压成单空行
    out, prev_blank = [], False
    for ln in lines:
        if ln.strip() == "":
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(ln)
            prev_blank = False
    return "\n".join(out).strip()

def infer_page(model, tok, img_path: str, prompt: str, workdir: str) -> str:
    """
    统一把输出当“目录”传入；三重兜底顺序：
      1) infer 返回值（dict/text）
      2) 递归读取输出目录里最新的 .md/.txt/.json
      3) 捕获 stdout 中的文本并清洗
    """
    assert os.path.isfile(img_path), f"Image not found: {img_path}"

    common = dict(
        prompt=prompt,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,      # 让有文件落盘的分支写到目录里
        test_compress=True,
    )

    page_dir = os.path.join(workdir, os.path.splitext(os.path.basename(img_path))[0])
    os.makedirs(page_dir, exist_ok=True)

    out_arg_candidates = ["output_path", "out_dir", "save_dir"]
    img_arg_candidates = [
        ("image_path", img_path),
        ("image_file", img_path),
        ("img_path",   img_path),
        ("image",      Image.open(img_path)),
        ("img",        Image.open(img_path)),
    ]

    last_err = None
    for img_k, img_v in img_arg_candidates:
        try:
            for out_k in out_arg_candidates:
                kwargs = dict(common)
                kwargs[img_k] = img_v
                kwargs[out_k] = page_dir

                # 捕获 stdout
                buf = io.StringIO()
                with redirect_stdout(buf):
                    try:
                        res = model.infer(tok, **kwargs)
                    except TypeError as e:
                        last_err = e
                        continue

                # 1) 返回值优先
                if isinstance(res, dict):
                    for key in ("text", "markdown", "md", "output"):
                        v = res.get(key)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                elif isinstance(res, str) and res.strip():
                    return res.strip()

                # 2) 读取输出目录的结果文件
                time.sleep(0.15)
                txt = _newest_text_file(page_dir)
                if txt:
                    return txt

                # 3) 用 stdout 兜底
                captured = _clean_stdout(buf.getvalue())
                if captured:
                    return captured

        finally:
            if hasattr(img_v, "close"):
                try: img_v.close()
                except: pass

    raise RuntimeError(f"DeepSeek-OCR infer signature not matched. Last error: {last_err}")
def run_ocr(
    model_path: str,
    images: List[str],
    out_md: str,
    remove_back_matter: bool,
    device: torch.device,
    dtype: torch.dtype,
):
    tok, model = load_deepseek(model_path, device, dtype)

    # 文本提示：是否删除尾部参考文献/致谢
    base_prompt = "<image>\n<|grounding|>Convert the document page to markdown."
    if remove_back_matter:
        base_prompt += " Exclude references, acknowledgements, funding, conflicts of interest, appendices."

    os.makedirs(os.path.dirname(out_md), exist_ok=True)

    workdir = tempfile.mkdtemp(prefix="deepseek_infer_")
    try:
        # 先清空目标文件，再逐页追加写入
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("")
            f.flush()

        stop_writing = False
        for idx, img in enumerate(images, start=1):
            print(f"[DeepSeek-OCR] OCR page {idx}/{len(images)} -> {os.path.basename(img)}")
            if stop_writing:
                print("  [skip] back-matter detected; skipping remaining pages.")
                break

            try:
                text = infer_page(model, tok, img, base_prompt, workdir)
            except Exception as e:
                text = f"(OCR failed on page {idx}: {e})"

            # 增量去尾部：页内命中“References/Appendix …”标题，则截断本页并停止后续页
            if remove_back_matter and isinstance(text, str) and text:
                m = BACK_MATTER_RE.search("\n" + text)
                if m:
                    text = text[: m.start() - 1]  # 截到标题之前
                    page_block = f"\n\n<!-- Page {idx} -->\n\n{text}\n"
                    with open(out_md, "a", encoding="utf-8") as f:
                        f.write(page_block)
                        f.flush()
                    stop_writing = True
                    break  # 直接结束循环

            # 正常写入当前页
            page_block = f"\n\n<!-- Page {idx} -->\n\n{text}\n"
            with open(out_md, "a", encoding="utf-8") as f:
                f.write(page_block)
                f.flush()

        print(f"✅ Saved Markdown to: {out_md}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
# ===== 批量辅助：用已加载的 tokenizer/model 处理一份 PDF =====
def _run_ocr_with_loaded(
    tok,
    model,
    images: list[str],
    out_md: str,
    remove_back_matter: bool,
):
    # 下面这段与 run_ocr 的主体一致，但不重复加载模型
    base_prompt = "<image>\n<|grounding|>Convert the document page to markdown."
    if remove_back_matter:
        base_prompt += " Exclude references, acknowledgements, funding, conflicts of interest, appendices."

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="deepseek_infer_")
    try:
        # 先清空目标文件，再逐页追加写入
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("")
            f.flush()

        stop_writing = False
        for idx, img in enumerate(images, start=1):
            print(f"[DeepSeek-OCR] OCR page {idx}/{len(images)} -> {os.path.basename(img)}")
            if stop_writing:
                print("  [skip] back-matter detected; skipping remaining pages.")
                break

            try:
                text = infer_page(model, tok, img, base_prompt, workdir)
            except Exception as e:
                text = f"(OCR failed on page {idx}: {e})"

            # 增量去尾部：页内命中“References/Appendix …”标题，则截断本页并停止后续页
            if remove_back_matter and isinstance(text, str) and text:
                m = BACK_MATTER_RE.search("\n" + text)
                if m:
                    text = text[: m.start() - 1]
                    page_block = f"\n\n<!-- Page {idx} -->\n\n{text}\n"
                    with open(out_md, "a", encoding="utf-8") as f:
                        f.write(page_block)
                        f.flush()
                    stop_writing = True
                    break

            page_block = f"\n\n<!-- Page {idx} -->\n\n{text}\n"
            with open(out_md, "a", encoding="utf-8") as f:
                f.write(page_block)
                f.flush()
        print(f"✅ Saved Markdown to: {out_md}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


DEFAULT_DEEPSEEK_MODEL_DIR = "/home/nas2/path/yangmingjian/DeepSeek-OCR"

# ===== 可传入“单个路径或路径列表”的对外函数 =====
def deepseek_pdf_to_md_batch(
    pdf_paths: list[str],
    out_dir: str,
    *,
    first_page: int = 1,
    last_page: int | None = None,
    dpi: int = 220,
    keep_refs: bool = False,
    cpu: bool = False,
    model_dir: str = DEFAULT_DEEPSEEK_MODEL_DIR,
    combine_out: str | None = None,   # 若提供则把所有结果合并到该文件；否则逐文件写入 out_dir
) -> list[str]:
    """
    批量将多个 PDF 转为 Markdown。模型只加载一次，逐份 PDF 逐页写入。

    Args:
        pdf_paths: PDF 路径列表
        out_dir:   输出目录（当 combine_out=None 时，每个 PDF 输出一个 .md 到此目录）
        first_page, last_page, dpi, keep_refs, cpu, model_dir: 同单文件接口
        combine_out: 若给出路径，则把所有 PDF 的内容合并写到这个文件里（并在各 PDF 之间插入分隔），
                     此时 out_dir 仍需存在，但主要用于临时/中间文件组织。

    Returns:
        list[str]: 生成的 Markdown 文件路径列表（若使用 combine_out，则只返回 [combine_out]）
    """
    if not pdf_paths:
        return []
    for p in pdf_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"PDF not found: {p}")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    device, dtype = choose_device_dtype(force_cpu=cpu)
    print(f"[Env] device={device}, dtype={dtype}")
    tok, model = load_deepseek(model_dir, device, dtype)

    outputs: list[str] = []
    combined_fh = None
    if combine_out is not None:
        os.makedirs(os.path.dirname(combine_out), exist_ok=True)
        combined_fh = open(combine_out, "w", encoding="utf-8")

    try:
        for pdf_path in pdf_paths:
            stem = os.path.splitext(os.path.basename(pdf_path))[0]
            out_md = os.path.join(out_dir, f"{stem}.md")
            tmpdir, imgs = pdf_to_images(pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)
            try:
                # 先生成单份 md（便于调试与合并）
                _run_ocr_with_loaded(
                    tok=tok,
                    model=model,
                    images=imgs,
                    out_md=out_md,
                    remove_back_matter=(not keep_refs),
                )
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            if combined_fh is not None:
                # 合并写入
                combined_fh.write(f"\n\n<!-- FILE: {os.path.basename(pdf_path)} -->\n\n")
                with open(out_md, "r", encoding="utf-8", errors="ignore") as fh:
                    combined_fh.write(fh.read())
                combined_fh.flush()
            else:
                outputs.append(out_md)

        if combined_fh is not None:
            combined_fh.close()
            outputs = [combine_out]
        return outputs
    finally:
        # 如果上面异常导致 combined_fh 未关闭，这里兜底
        try:
            if combined_fh and not combined_fh.closed:
                combined_fh.close()
        except Exception:
            pass
def main():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR PDF -> Markdown")
    parser.add_argument("--pdf", required=True, help="输入 PDF 路径")
    parser.add_argument("--out", required=True, help="输出 Markdown 路径")
    parser.add_argument("--model", required=True, help="DeepSeek-OCR 本地目录（含权重与代码）")
    parser.add_argument("--dpi", type=int, default=220, help="渲染 DPI（更大更清晰但更占显存/内存）")
    parser.add_argument("--first-page", type=int, default=1, help="起始页（1-based）")
    parser.add_argument("--last-page", type=int, default=None, help="结束页（默认到最后一页）")
    parser.add_argument("--keep_refs", action="store_true", help="保留参考文献/致谢等尾部内容")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 推理")
    args = parser.parse_args()

    device, dtype = choose_device_dtype(force_cpu=args.cpu)
    print(f"[Env] device={device}, dtype={dtype}")

    tmpdir, imgs = pdf_to_images(args.pdf, dpi=args.dpi, first_page=args.first_page, last_page=args.last_page)
    try:
        run_ocr(
            model_path=args.model,
            images=imgs,
            out_md=args.out,
            remove_back_matter=(not args.keep_refs),
            device=device,
            dtype=dtype,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    # 纯离线：若你机器默认会联网，可手动打开此开关
    # os.environ["HF_HUB_OFFLINE"] = "1"
    main()