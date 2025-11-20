import os
from pathlib import Path
from pyexpat import model
from typing import List

from dotenv import find_dotenv, load_dotenv
from metapub import FindIt, PubMedFetcher
from openai import OpenAI

from Core.Agent import Agent
from utils.pdf2mdOCR import ocr_to_md_files
from Logger.index import get_global_logger
from Memory.index import Subgraph
from Store.index import get_memory
from utils.download import save_pdfs_from_url_list
from utils.filter import extract_pdf_paths
from utils.pdf2md import deepseek_pdf_to_md_batch
from utils.process_markdown import split_md_after_trim


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
        super().__init__(client,model_name,self.system_prompt)
    
    def process(self,user_query:str):
        strategy = self.generateMeSHStrategy(user_query)
        reviews_metadata = self.fetchReviews(strategy, maxlen=20)
        selected_reviews = self.selectReviews(reviews_metadata, topk=2)
        review_urls = []
        for pmid in selected_reviews:
            try:
                review_urls.append(FindIt(pmid).url)
            except:
                self.logger.warning(f"Failed to fetch URL for PMID: {pmid}")
        md_outputs=ocr_to_md_files(review_urls)
        for md_output in md_outputs:
            paragraphs=split_md_after_trim(md_output)
            for id,content in paragraphs.items():
                for i,content_chunk in enumerate(content):
                    subgraph_id=f"{id}_{i}"
                    meta={"text":content_chunk,"source":id}
                    s = Subgraph(subgraph_id=subgraph_id,meta=meta)
                    self.memory.register_subgraph(s)
        return


    def generateMeSHStrategy(self,user_query:str)->str:
        prompt=f"""
        As a biomedical literature search expert, generate a PubMed search strategy using MeSH terms for the following research question:
        Question: {user_query}
        Requirements:
        1. Use MeSH terms
        2. Combine free-text terms
        3. Use Boolean operators (AND/OR/NOT)
        4. Limit document type to reviews
        5. Limit to articles from the last 5 years
        Note:
        Please only return the search strategy without any explanations.
        """
        result=self.call_llm(prompt)
        return str(result)
    
    def fetchReviews(self,search_strategy:str,maxlen=20):
        pmids=self.fetch.pmids_for_query(str(search_strategy),retmax=maxlen)
        reviews_metadata = [self.fetch.article_by_pmid(pmid) for pmid in pmids]
        return reviews_metadata
    
    def selectReviews(self,reviews_metadata, topk=5) -> List:
        review_str='\n'.join(review.__str__() for review in reviews_metadata)
        selection_prompt = f"""
        From the following {len(reviews_metadata)} reviews, select the most relevant {topk} ones:
        {review_str}
        Selection criteria:
        1. Cover different aspects of the query topic
        2. High citation count and impact factor
        3. Recent publication date
        4. Include mechanism studies and clinical applications
        Please return the selected {topk} review pmids in a comma-separated format without any additional description.
        """
        selected_str = str(self.call_llm(selection_prompt))
        selected_str = selected_str.replace("[", "").replace("]", "")
        selected_5 = [pid.strip() for pid in selected_str.split(",") if pid.strip()]
        return selected_5

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