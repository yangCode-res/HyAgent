from pathlib import Path
from metapub import FindIt, PubMedFetcher
from pdf_to_paragraphs import generate_text
from utils.download import save_pdfs_from_url_list
from utils.pdf2md import deepseek_pdf_to_md_batch  # ✅ 用我们刚写的批量 OCR 函数
from utils.process_markdown import clean_markdown  # ✅ 可选清洗

fetch = PubMedFetcher()

def format_reviews(reviews_metadata):  # 将多篇文章格式化为字符串
    formatted_reviews = []
    for review in reviews_metadata:
        formatted_reviews.append(format_review(review))
    return "\n\n".join(formatted_reviews)

def ReviewSearch(user_query,maxlen=20):#生成搜索策略并检索文章
    prompt = f"""
    作为⽣物医学检索专家,为以下研究问题⽣成PubMed检索策略:
    问题: {user_query}
    要求:
    1. 使⽤MeSH术语
    2. 结合⾃由词检索
    3. 使⽤布尔运算符(AND/OR/NOT)
    4. 限定⽂献类型为综述(Review)
    5. 限定近5年⽂献
    注意事项：
    请仅仅返回检索策略即可，不要任何的说明。
    """
    result=generate_text(prompt)
    conversation_history.pop()
    pmids=fetch.pmids_for_query(str(result),retmax=maxlen)
    reviews_metadata = [fetch.article_by_pmid(pmid) for pmid in pmids]
    return reviews_metadata

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

