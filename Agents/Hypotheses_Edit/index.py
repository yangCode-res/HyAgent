from datetime import date
import datetime
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json

from Core.Agent import Agent
from Logger.index import get_global_logger
from Memory.index import Memory, load_memory_from_json
from Store.index import get_memory
from TypeDefinitions.EntityTypeDefinitions.index import KGEntity
from TypeDefinitions.TripleDefinitions.KGTriple import KGTriple


class HypothesisGenerationAgent(Agent):
    """
    基于：
      - 用户 query
      - 假设 feedback
      - PathExtractionAgent 抽取的 KG 路径（节点 + 边）
      - 上下文信息（path 相关的文献片段等）

    调用 LLM 修改和生成新的假设。
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        query: str,
        memory: Optional[Memory] = None,
    ):
        system_prompt = """
    You are an expert biomedical AI4Science assistant specializing in hypothesis refinement and critical analysis.
    
    Your task is to REVISE and IMPROVE a previously generated scientific hypothesis based on specific feedback or critique provided.
    You must address the issues raised in the feedback (e.g., lack of mechanistic detail, weak experimental design, logical gaps) while maintaining the relevance to the user's original query.

    The input will be a JSON payload containing:
    1. The original user query.
    2. The original hypothesis that needs improvement.
    3. The specific feedback/critique to address.
    4. (Optional) Additional context or KG paths if provided.

    The input format is as follows:
    Task:
    {{
        "task": "refine a hypothesis based on feedback",
        "query": "query",
        "original_hypothesis_data": {{
            "title": "original title",
            "hypothesis": "original hypothesis statement",
            "mechanism_explanation": "original explanation",
            "experimental_suggestion": "original experiment"
        }},
        "feedback": "Specific instructions on what to improve. Examples: 'The mechanism is too vague regarding the receptor interaction', 'The experimental suggestion lacks a negative control', or 'Incorporate the role of Gene X mentioned in the context'.",
        "contexts": "contexts"
    }}

    Your Goal:
    - GENERATE A REVISED VERSION of the hypothesis.
    - ENSURE the new hypothesis explicitly resolves the issues mentioned in the "feedback".
    - STRENGTHEN the mechanistic logic and experimental rigor.
    - UPDATE the confidence score if the refinement makes the hypothesis more plausible.

    You should respond ONLY with valid JSON in the following format:

    {{
    "hypotheses": [
        {{
        "title": "REVISED short hypothesis title",
        "hypothesis": "REVISED full hypothesis statement (more detailed and robust)",
        "mechanism_explanation": "REVISED explanation explicitly addressing the feedback mechanics",
        "experimental_suggestion": "REVISED experimental idea (concrete, including controls/methods if requested)",
        "relevance_to_query": "Reiteration of relevance, updated if necessary",
        "confidence": 0.0,
        "refinement_rationale": "Briefly explain how you addressed the feedback (e.g., 'Added details on phosphorylation pathway as requested')"
        }}
    ]
    }}

    Do not include any text outside the JSON response.
    """


        super().__init__(client, model_name, system_prompt)

        self.logger = get_global_logger()
        self.memory: Memory = memory or get_memory()
        self.query = query


    def serialize_hypothesis(self, hypothesis: List[Dict[str, Any]]) -> str:
        lines = []
        for hypo in hypothesis:
            lines.append(
                f"Title: {hypo.get('title', '')}\n"
                f"Hypothesis: {hypo.get('modified_hypothesis', '')}\n"
                f"Mechanism Explanation: {hypo.get('mechanism_explanation', '')}\n"
                f"Experimental Suggestion: {hypo.get('experimental_suggestion', '')}\n"
                f"Relevance to Query: {hypo.get('relevance_to_query', '')}\n"
                f"Confidence: {hypo.get('confidence', 0.0)}\n"
            )
        return "\n".join(lines)
    

    def process(self,original_hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for original_hypothesis in original_hypotheses:
            self.logger.info(
                f"[HypothesisGeneration] Processing original hypothesis: {original_hypothesis.get('title', '')}"
            )
        for hypothesis in original_hypotheses:
            modified_hyps=hypothesis.get("modified_hypotheses",[])
            hypo_str=self.serialize_hypothesis([modified_hyps])
            feedback_str=json.dumps(hypothesis.get("feedback",""), ensure_ascii=False)
            payload={
                "task": "refine a hypothesis based on feedback",
                "query": self.query,
                "original_hypothesis_data": hypo_str,
                "feedback": feedback_str,
                "contexts": hypothesis.get("contexts","")
            }
            prompt=f"""Now you need to refine the following hypothesis based on the feedback provided.
            Here is the input JSON payload:
            {json.dumps(payload, ensure_ascii=False)}
            Please provide the revised hypothesis in the specified JSON format.
            """
            raw=self.call_llm(prompt)
            try:
                response=json.loads(raw)
                refined_hyps=response.get("hypotheses",[])
                hypothesis["refined_hypotheses"]=refined_hyps
                self.logger.info(
                    f"[HypothesisGeneration] Refined hypotheses generated for: {hypothesis.get('title', '')}"
                )
            except Exception as e:
                self.logger.error(
                    f"[HypothesisGeneration][LLM process] JSON parse failed for hypothesis: {hypothesis.get('title', '')}, error: {e}"
                )
        timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(f'../../output/output_{timestamp}_modified.json', 'w', encoding='utf-8') as f:
            json.dump(
                original_hypotheses, 
                f, 
                ensure_ascii=False, 
                indent=4, 
                # 只需要这一行 lambda
                default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o)
            )
        if self.memory:
            self.memory.add_hypothesesDir(f'output/output_{timestamp}_modified.json')
        return original_hypotheses
        