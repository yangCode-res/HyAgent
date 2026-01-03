import os
import sys
from pathlib import Path
from typing import List

from dotenv import find_dotenv, load_dotenv
from metapub import FindIt, PubMedFetcher
from openai import OpenAI
from pyexpat import model

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Subgraph
from Store.index import get_memory
from utils.download import save_pdfs_from_url_list
from utils.filter import extract_pdf_paths
from utils.pdf2md import deepseek_pdf_to_md_batch
from utils.pdf2mdOCR import ocr_to_md_files
from utils.process_markdown import split_md_by_mixed_count

fetch=PubMedFetcher()
class ReviewFetcherAgent(Agent):
    def __init__(self,client:OpenAI,model_name:str) -> None:
        self.system_prompt="""You are a specialized Review Fetcher Agent for biomedical knowledge graphs. Your task is to fetch relevant literature reviews based on the user's query.
        You are requested to do the following:
        1.Understand the user's query and identify key biomedical concepts, generate MeSH search strategy.
        2.Based on the given abstracts of the reviews, select the most relevant ones that comprehensively cover different aspects of the query.
        """
        self.memory=get_memory()
        self.logger=get_global_logger()
        self.fetch=PubMedFetcher()
        self.k=2
        super().__init__(client,model_name,self.system_prompt)
    
    def process(self, user_query: str, save_dir: str | None = None):
        strategy = self.generateMeSHStrategy(user_query)
        reviews_metadata = self.fetchReviews(strategy, maxlen=30)
        selected_reviews = self.selectReviews(reviews_metadata, query=user_query, topk=10)
        review_urls = []
        for pmid in selected_reviews:
            try:
                review_urls.append(FindIt(pmid).url)
            except:
                self.logger.warning(f"Failed to fetch URL for PMID: {pmid}")
        self.logger.debug(f"review_urls=> {review_urls}")
        review_urls=[url for url in review_urls if url is not None]
        # 保存 OCR→MD 到指定目录，默认使用项目根下的 ocr_md_outputs
        md_outputs = ocr_to_md_files(review_urls, save_dir=save_dir or "ocr_md_outputs")
        # md_outputs=["/home/nas3/biod/dongkun/HyAgent/ocr_md_outputs/ocr_result_1.md"]
        # 过滤掉 None 值（OCR 失败的情况）
        md_outputs = [md for md in md_outputs if md is not None]
        self.logger.debug(f"md_outputs=> {md_outputs}")
        for md_output in md_outputs[0:self.k]:
            paragraphs=split_md_by_mixed_count(md_output)

            # paragraphs=split_md_by_h2(md_output)
            for id,content in paragraphs.items():
                for i,content_chunk in enumerate(content):
                    subgraph_id=f"{id}_{i}"
                    meta={"text":content_chunk,"source":id}
                    s = Subgraph(subgraph_id=subgraph_id,meta=meta)
                    self.memory.register_subgraph(s)
        if len(review_urls) == 0:
            self.logger.warning("No review URLs found")
            sys.exit(1)
        # 返回生成的 MD 文件路径，便于上层批处理逻辑组织输出
        return md_outputs


    def generateMeSHStrategy(self,user_query:str)->str:
        prompt = f"""
As an expert Biomedical Information Specialist, generate a high-recall PubMed search strategy to find **Reviews** and **Systematic Reviews** relevant to the user's query.

**User Query:** {user_query}

**Critical Rules for Logic:**
1. **Simplify Concepts:** Extract ONLY the **2 most critical concepts** (usually **Intervention** and **Outcome**).
   - *Example:* For "Tirzepatide vs Semaglutide in obesity for CV outcomes in US insurance data", the ONLY concepts are: (Tirzepatide OR Semaglutide) AND (Cardiovascular Outcomes).
   - **IGNORE** geographic locations (e.g., "USA", "China").
   - **IGNORE** data sources (e.g., "insurance claims", "hospital records").
   - **IGNORE** specific study designs in the query text (e.g., "cohort", "RCT") because we will apply a Review filter later.
   - **IGNORE** specific dates mentioned in the text (e.g., "2018-2025") as we will use a date filter.

2. **Handle Comparisons:** If the query compares two drugs (e.g., Drug A vs Drug B), combine them into **ONE** concept using **OR** (e.g., `("Drug A" OR "Drug B")`), rather than splitting them with AND. This ensures we catch reviews discussing the whole class or either drug.

3. **Handle Population (P):** Do NOT create a separate 'AND' block for the disease (e.g., Diabetes/Obesity) *unless* the drugs are used for multiple wildly different conditions. For GLP-1s, the disease is implied by the outcome (CV risk), so adding "AND Diabetes" might restrict results too much. **Prioritize Broad Search.**

**Step-by-Step Construction:**
1. **Concept 1 (Intervention/Exposure):** Expand with MeSH + Keywords + Brand Names + CAS/Drug Codes.
2. **Concept 2 (Outcome/Main Topic):** Expand with MeSH + Keywords (Synonyms).
3. **Combine:** (Concept 1) AND (Concept 2).
4. **Filters:**
   - Apply "Review"[Publication Type] OR "Systematic Review"[Publication Type].
   - Apply Date Range: "2018/01/01"[Date - Publication] : "2023/12/31"[Date - Publication] (Last 5+ years).

**Output format:**
Return ONLY the raw search query string. No markdown, no explanations.
"""
        result=self.call_llm(prompt)
        self.logger.debug(f"mesh strategy=> {result}")
        return str(result)
    
    def fetchReviews(self,search_strategy:str,maxlen=1):
        pmids=self.fetch.pmids_for_query(str(search_strategy),retmax=maxlen)
        reviews_metadata = [self.fetch.article_by_pmid(pmid) for pmid in pmids]
        return reviews_metadata
    
    def selectReviews(self,reviews_metadata, query='',topk=1) -> List:
        review_str='\n'.join(self.format_review(review) for review in reviews_metadata)
        self.logger.debug(f"review_str=> {review_str}")
        selection_prompt = f"""
        here is the user query: {query}, and here are the reviews:
        From the following {len(reviews_metadata)} reviews, select the most relevant {topk} ones:
        {review_str}
        Selection criteria:
        1. Cover different aspects of the query topic
        2. High citation count and impact factor
        3. Recent publication date
        4. Include mechanism studies and clinical applications
        5. Select the reviews that most caters to the user query.
        6. You should also take the richness of the review(identified as the range of pages here however the page range is not always available) into consideration so that we could build a knowledge graph with more triples.
        If the page range is not available, you should put other requirements first.
        Please return the selected {topk} review pmids in a comma-separated format without any additional description.
        """
        selected_str = str(self.call_llm(selection_prompt))
        selected_str = selected_str.replace("[", "").replace("]", "")
        selected_5 = [pid.strip() for pid in selected_str.split(",") if pid.strip()]
        return selected_5
    def format_review(self,article):#将标题、日期、引用量、摘要、文章id喂给模型
        return f"""
        title: {article.title}
        pubdate: {article.pubdate}
        citation_count: {fetch.related_pmids(article.pmid).__len__()}
        abstract: {article.abstract}
        pmid: {article.pmid}
        page-range:{article.pages}
        """
if __name__ == "__main__":
    from openai import OpenAI
    try:
        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
            pass
    open_ai_api=os.environ.get("OPENAI_API_KEY")
    open_ai_url=os.environ.get("OPENAI_API_BASE_URL")
    model_name=os.environ.get("OPENAI_MODEL")
    client = OpenAI(api_key=open_ai_api, base_url=open_ai_url)
    agent = ReviewFetcherAgent(client, model_name=model_name)
    user_query = "What are the latest advancements in CRISPR-Cas9 gene editing technology for treating genetic disorders?"
    agent.process(user_query)
    agent.memory.dump_json("./snapshots")