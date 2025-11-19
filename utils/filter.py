from pathlib import Path

from metapub import FindIt, PubMedFetcher

from utils.download import save_pdfs_from_url_list
from utils.pdf2md import deepseek_pdf_to_md_batch
from utils.search import batch_search_reviews_from_user_query

fetch = PubMedFetcher()


def format_reviews(reviews_metadata):  # 将多篇文章格式化为字符串
    formatted_reviews = []
    for review in reviews_metadata:
        formatted_reviews.append(format_review(review))
    return "\n\n".join(formatted_reviews)


def format_review(article):  # 将标题、日期、引用量、摘要、文章id喂给模型
    return f"""
    标题: {article.title}
    发表日期: {article.pubdate}
    引用量: {fetch.related_pmids(article.pmid).__len__()}
    摘要: {article.abstract}
    文章id: {article.pmid}
    """


def ReviewSelection(reviews_metadata, topk=5) -> list:  # 选择最合适的文章
    selection_prompt = f"""
    从以下{len(reviews_metadata)}篇综述中选择最相关的{topk}篇:
    {format_reviews(reviews_metadata)}
    选择标准:
    1. 覆盖查询主题的不同⽅⾯
    2. ⾼引⽤量和影响因⼦
    3. 最新发表⽇期
    4. 包含机制研究和临床应⽤
    请用,隔开的形式返回所选择的{topk}篇综述的pid，不需要其他额外叙述。
    """
    selected_str = str(generate_text(selection_prompt))
    selected_str = selected_str.replace("[", "").replace("]", "")
    selected_5 = [pid.strip() for pid in selected_str.split(",") if pid.strip()]
    return selected_5


def extract_pdf_paths(download_results) -> list[str]:
    """
    从 save_pdfs_from_url_list 的结果中提取成功的本地 PDF 路径列表。
    """
    pdfs = []
    for item in download_results:
        if item.get("status") in {"OK", "EXISTS"} and item.get("path_or_msg"):
            p = Path(item["path_or_msg"])
            if p.is_file() and p.suffix.lower() == ".pdf":
                pdfs.append(str(p))
    return pdfs


if __name__ == "__main__":
    user_query = "Causal mechanisms linking diabetes and cardiovascular disease and potential therapeutic targets"
    print(f"🔍 Query: {user_query}\n")

    # 1) 检索候选综述
    results = batch_search_reviews_from_user_query(
        user_query=user_query,     # 传入问题
        years_back=5,              # 近5年
        batch_size=60,             # 每批候选
        topk_batch=10,             # 每批取10
        K_total=30,                # 总共要30
        strong_review=False,       # 是否使用强综述：此处否
        lang_filter=["english"],   # 只要英文文献（可去掉此参数）
        lambda_decay=0.35,         # 越大越偏近期
        year_max_ratio=0.4,        # 单年最多占40%
    )

    # 2) 选择 topK 篇综述
    selected_pmids = ReviewSelection(results, topk=10)  # 有些可能没有全文，故取 10
    print("Selected PMIDs:", selected_pmids)

    # 3) 找到全文 URL 并下载 PDF 到本地
    selected_reviews = [FindIt(pmid).url for pmid in selected_pmids]
    print("Selected Reviews URLs:", selected_reviews)

    download_results = save_pdfs_from_url_list(
        selected_reviews,
        outdir="downloaded_pdfs",
        overwrite=False,
        timeout=20,
    )

    # 4) 提取成功下载的 PDF 本地路径
    pdf_paths = extract_pdf_paths(download_results)
    if not pdf_paths:
        print("⚠️ 没有成功下载到可用的 PDF。")
        exit(0)

    # 5) 批量 OCR → Markdown（模型只加载一次，效率更高）
    #    - 默认模型位置在 utils.pdf2md 里有常量 DEFAULT_DEEPSEEK_MODEL_DIR
    #    - 3090 建议走 GPU（cpu=False）
    markdown_dir = Path(__file__).resolve().parent / "markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)

    md_outputs = deepseek_pdf_to_md_batch(
        pdf_paths=pdf_paths,
        out_dir=str(markdown_dir),
        first_page=1,          # 如需只测前几页可设 last_page，例如 last_page=3
        last_page=None,
        dpi=220,
        keep_refs=False,       # 不保留参考文献/致谢等
        cpu=False,             # 3090 走 GPU；若想走 CPU，改为 True
        # model_dir 不传就用 utils.pdf2md 里的默认：/home/nas2/path/yangmingjian/DeepSeek-OCR
        # combine_out 可传一个路径把多篇合并到一个 md；这里按篇输出
    )

    if not md_outputs:
        print("⚠️ 未成功生成 Markdown 文件。")
        exit(0)

    print("✅ 生成的 Markdown 文件：")
    for p in md_outputs:
        print("  -", p)

    # 6) （可选）做一次轻量清洗：去尾部引用/噪声，保守策略
    #    - 你的 clean_markdown 支持 tail_portion/min_ref_block/min_keep_ratio，可自行调整
    # for md_file in md_outputs:
    #     try:
    #         raw_md = Path(md_file).read_text(encoding="utf-8", errors="ignore")
    #         cleaned_md, _ = clean_markdown(
    #             raw_md,
    #             tail_portion=0.30,   # 仅对文末 30% 进行参考文献/附录识别
    #             min_ref_block=10,    # 参考条目最小块大小
    #             min_keep_ratio=0.60, # 至少保留 60% 正文，避免误删过多
    #         )
    #         Path(md_file).write_text(cleaned_md, encoding="utf-8")
    #     except Exception as e:
    #         print(f"[WARN] 清洗 {md_file} 时出错：{e}")