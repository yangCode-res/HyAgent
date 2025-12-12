import json
import re
from typing import Any, Dict, List, Optional

from Core.Agent import Agent
from openai import OpenAI
from Memory.index import Memory
from Store.index import get_memory


class ReflectionAgent(Agent):
    """
    Input: hypothesis (dict)
    Output: reflection report (Strict Structured JSON Dictionary)
    
    Key Features:
    - Enforces PascalCase keys (e.g., SafetyEthics) for code stability.
    - Validates JSON schema before returning.
    - returns a Python Dictionary, not a string.
    """

    _RUBRIC_ANCHORS = r"""
    You must evaluate the hypothesis using the following 6 criteria.

    For EACH criterion, you must provide a JSON object with:
    - Score: string "1/5" to "5/5"
    - Rationale: list of strings (bullet points of evidence)
    - Concerns: list of strings (concrete issues, empty if none)
    - Suggestions: list of strings (actionable fixes, empty if none)

    Scoring anchors (1–5):

    (1) Novelty (Key: Novelty)
    1. Almost no novelty: common trope; trivial recombination.
    2. Weak novelty: minor variant; straightforward extension.
    3. Moderate novelty: identifiable new linkage but close to existing work.
    4. Clear novelty: under-discussed mechanism; explicit differences.
    5. High novelty: non-obvious mechanism with testable predictions.

    (2) Plausibility (Key: Plausibility)
    1. Implausible: contradicts basic knowledge; broken causal chain.
    2. Low plausibility: unstated strong assumptions; confused direction.
    3. Partly plausible: coherent but key steps missing.
    4. Plausible: coherent chain; assumptions stated.
    5. Highly plausible: complete mechanism; handles alternatives.

    (3) Grounding (Key: Grounding)
    1. No evidence: speculation; references missing.
    2. Weak evidence: scattered hints; claims not supported.
    3. Moderate: some claims supported; gaps remain.
    4. Strong: key claims supported by evidence/reasoning.
    5. Very strong: complete traceable chain.

    (4) Testability (Key: Testability)
    1. Not testable: too abstract; no measurable endpoints.
    2. Hard to test: high cost; metrics unclear.
    3. Partly testable: directions exist but vague.
    4. Testable: executable experiments with clear controls.
    5. Highly testable: multiple validation routes; reproducibility.

    (5) Specificity (Key: Specificity)
    1. Too vague: lacks actor/condition/scope.
    2. Somewhat vague: missing constraints (context, timing).
    3. Moderate: actors stated; exclusions incomplete.
    4. Specific: scope bounded; clear mechanism.
    5. Very specific: directly translatable to protocol; constraints defined.

    (6) Safety & Ethics (Key: SafetyEthics)
    1. High risk: unsafe/unethical.
    2. Significant risk: ignores constraints.
    3. Manageable: needs compliance additions.
    4. Safe: risks identified and mitigated.
    5. Very safe: clear limitations and safer alternatives.

    General requirements:
    - Be strict: do not inflate scores.
    - Suggestions must be executable (e.g., “Add boundary condition X”).
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        hypothesis: Dict[str, Any],
        memory: Optional[Memory] = None,
    ):
        # 构造严格的 JSON 模板
        # 注意：这里定义了 EditInstructions 的具体结构，方便 RevisionAgent 自动执行
        json_template = json.dumps({
            "Novelty": {
                "Score": "3/5",
                "Rationale": ["Point 1", "Point 2"],
                "Concerns": ["Concern 1"],
                "Suggestions": ["Suggestion 1"]
            },
            "Plausibility": { "Score": "X/5", "Rationale": [], "Concerns": [], "Suggestions": [] },
            "Grounding": { "Score": "X/5", "Rationale": [], "Concerns": [], "Suggestions": [] },
            "Testability": { "Score": "X/5", "Rationale": [], "Concerns": [], "Suggestions": [] },
            "Specificity": { "Score": "X/5", "Rationale": [], "Concerns": [], "Suggestions": [] },
            "SafetyEthics": { "Score": "X/5", "Rationale": [], "Concerns": [], "Suggestions": [] },
            "OverallSummary": {
                "Strengths": ["Strength 1"],
                "Weaknesses": ["Weakness 1"],
                "PriorityMustFix": ["Critical issue 1"],
                "NiceToFix": ["Minor issue 1"],
                "RiskFlags": ["Risk 1"],
                "EditInstructions": [
                    {
                        "Target": "Section Name or Whole Hypothesis",
                        "Action": "Rewrite / Add / Delete",
                        "Description": "Specific instruction on what to change."
                    }
                ]
            }
        }, indent=2)

        system_prompt = f"""You are a rigorous scientific reviewer.

        {self._RUBRIC_ANCHORS}

        You will receive:
        - A hypothesis (as JSON text)
        - Optional context (as JSON text)

        IMPORTANT OUTPUT RULES:
        1. Output ONLY valid JSON.
        2. Do NOT output Markdown code fences (like ```json).
        3. Use PascalCase for ALL keys. No spaces, no ampersands (&).
        4. Follow this EXACT JSON structure and types:

        {json_template}
        """
        super().__init__(client, model_name, system_prompt)
        self.hypothesis = hypothesis
        self.memory = memory or get_memory()

    def process(self) -> Dict[str, Any]:
        """
        Executes the agent logic:
        1. Calls LLM.
        2. Cleans and extracts JSON from response.
        3. Validates schema.
        4. Returns structured Dict.
        """
        hypothesis_text = json.dumps(self.hypothesis, ensure_ascii=False, indent=2)

        user_message = (
            "Hypothesis (JSON):\n"
            f"{hypothesis_text}\n\n"
            "\n"
            "Review Task:\n"
            "Provide a critique in strict JSON format based on the rubric.\n"
            "Output JSON ONLY. No preamble. No markdown."
        )

        # 1. 调用 LLM
        raw_response = self.call_llm(user_message)

        if not isinstance(raw_response, str) or not raw_response.strip():
            raise ValueError("Empty reflection output from model.")

        # 2. 清理与提取 JSON
        cleaned_json_str = self._clean_and_extract_json(raw_response)

        # 3. 解析 JSON
        try:
            data = json.loads(cleaned_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON Parse Error: {e}\nRaw Output: {raw_response}")

        # 4. 结构校验 (Schema Validation)
        self._validate_schema(data)

        return data

    def _clean_and_extract_json(self, text: str) -> str:
        """
        使用正则提取最外层的 JSON 对象，忽略前后的 Markdown 标记或废话。
        """
        text = text.strip()
        # 尝试匹配第一个 { 和最后一个 } 之间的内容 (re.DOTALL 让 . 匹配换行符)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            # 备用清理逻辑
            cleaned = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE)
            cleaned = re.sub(r"```$", "", cleaned)
            return cleaned.strip()

    def _validate_schema(self, data: Dict[str, Any]) -> None:
        """
        检查返回的 Dict 是否包含所有必要的 Key 和正确的类型。
        这能防止 RevisionAgent 因为缺字段而崩溃。
        """
        # 1. Check Top-level Keys
        required_criteria = [
            "Novelty", "Plausibility", "Grounding", 
            "Testability", "Specificity", "SafetyEthics", 
            "OverallSummary"
        ]
        
        for key in required_criteria:
            if key not in data:
                raise ValueError(f"ReflectionAgent Output Missing required top-level key: '{key}'")

        # 2. Check Criteria Fields (Novelty...SafetyEthics)
        criteria_subfields = ["Score", "Rationale", "Concerns", "Suggestions"]
        
        for key in required_criteria[:-1]: # Exclude OverallSummary from this loop
            item = data[key]
            if not isinstance(item, dict):
                raise ValueError(f"Key '{key}' must be a dictionary.")
            
            for sub in criteria_subfields:
                if sub not in item:
                    raise ValueError(f"Key '{key}' missing subfield '{sub}'.")
            
            # 类型检查：Rationale 必须是列表，Score 必须是字符串
            if not isinstance(item["Rationale"], list):
                raise ValueError(f"'{key}.Rationale' must be a list of strings.")
            if not isinstance(item["Score"], str):
                 raise ValueError(f"'{key}.Score' must be a string (e.g. '3/5').")

        # 3. Check OverallSummary Fields
        summary_fields = [
            "Strengths", "Weaknesses", "PriorityMustFix", 
            "NiceToFix", "RiskFlags", "EditInstructions"
        ]
        summary = data["OverallSummary"]
        for field in summary_fields:
            if field not in summary:
                raise ValueError(f"OverallSummary missing field '{field}'.")
        
        # 检查 EditInstructions 必须是列表
        if not isinstance(summary["EditInstructions"], list):
             raise ValueError("OverallSummary.EditInstructions must be a list.")