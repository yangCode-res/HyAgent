from pathlib import Path

from markdown_it import MarkdownIt

"""
md_preprocess.py

功能概述：
    1. 从一篇论文 OCR 得到的 Markdown 文件中，删除参考文献部分：
       - 识别形如 "## References" 或 "## 参考文献" 的二级标题（H2）
       - 从该标题开始到文件末尾的内容全部删除，只保留正文部分
    2. 在删除参考文献之后，将正文按“约 1200 词一块”进行分段：
       - 实际切分点只发生在段落边界（段落之间的空行处）
       - 尽量保证上下文连续，不在段落内部硬切

主要函数：
    - trim_md_inplace(path: str) -> None
        就地修改单个 .md 文件，删除参考文献之后的部分。
        参数：
            path: Markdown 文件路径（字符串）
        行为：
            如找到 "## References" 或 "## 参考文献" 等 H2 标题，则截断；
            如找不到，则不做任何修改。

    - split_md_after_trim(path: str, target_words: int = 1200) -> dict[str, list[str]]
        调用 trim_md_inplace 删除参考文献后，将正文按段落 + 约 target_words 词切块。
        参数：
            path: Markdown 文件路径（字符串）
            target_words: 每块目标单词数（默认约 1200）
        返回：
            一个字典：{ 文件名: [chunk1, chunk2, ...] }
            其中每个 chunk 是一段文本，由若干完整段落组成。

使用示例：
    from md_preprocess import split_md_after_trim

    result = split_md_after_trim("paper_ocr.md", target_words=1200)
    filename, chunks = next(iter(result.items()))
    print(filename, len(chunks))
    for i, text in enumerate(chunks):
        print(f"=== chunk {i} ===")
        print(text[:300])  # 查看每块前 300 个字符
"""
# 识别参考文献标题
REF_TITLES = {"references", "参考文献"}


def trim_md_inplace(path: str):
    """
    就地删除 md 中从第一个 `## References` / `## 参考文献` 开始的内容
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")

    md = MarkdownIt()
    tokens = md.parse(text)

    cut_line = None
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.tag == "h2" and tok.map:
            title = tokens[i + 1].content.strip().lower()
            title = title.replace(":", "").replace("：", "")
            if title in REF_TITLES or any(title.startswith(t) for t in REF_TITLES):
                cut_line = tok.map[0]
                break

    # 没找到就不改
    if cut_line is None:
        return

    lines = text.splitlines(keepends=True)
    p.write_text("".join(lines[:cut_line]), encoding="utf-8")


def split_md_after_trim(path: str, target_words: int = 1200) -> dict:
    """
    1) 调用 trim_md_inplace 去掉参考文献部分
    2) 将剩余内容按“段落 + 约 target_words 词”切块
       - 只在段落之间切分（段落由空行分隔）
    3) 返回 {filename: [chunk1, chunk2, ...]}
    """
    p = Path(path)

    # 第一步：去掉参考文献
    trim_md_inplace(str(p))

    # 第二步：读取文本
    text = p.read_text(encoding="utf-8", errors="ignore")

    # 按空行拆成段落（保留段内换行）
    lines = text.splitlines()
    paragraphs = []
    buf = []
    for line in lines:
        if line.strip() == "":
            if buf:
                paragraphs.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        paragraphs.append("\n".join(buf))

    # 按“约 target_words 词” + 段落边界切块
    chunks = []
    cur_paras = []
    cur_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        # 如果已经有内容，再加这个段落会明显超过 target，就在前一个段落边界切一块
        if cur_paras and cur_words + para_words > target_words:
            chunks.append("\n\n".join(cur_paras).strip())
            cur_paras = [para]
            cur_words = para_words
        else:
            cur_paras.append(para)
            cur_words += para_words

    # 收尾
    if cur_paras:
        chunks.append("\n\n".join(cur_paras).strip())

    return {p.name.replace(".md", ""): chunks}


if __name__ == "__main__":
    import sys

    md_path = sys.argv[1]
    result = split_md_after_trim(md_path, target_words=1200)

    fname, blocks = next(iter(result.items()))
    print(f"{fname} -> {len(blocks)} chunks")
    # 你也可以在这里打印每块的前几行做检查
    for i, b in enumerate(blocks):
        print("="*20, i, "="*20)
        print("\n".join(b.splitlines()[:5]))