import json
import re
from typing import Any, Dict, Optional

from Core.Agent import Agent
from openai import OpenAI
from Memory.index import Memory
from Store.index import get_memory
from pprint import pprint

class ReflectionAgent(Agent):
    """
    Input: hypothesis (dict)
    Output: reflection report (Strict Structured JSON Dictionary)
    
    Key Features:
    - Enforces PascalCase keys (e.g., SafetyEthics) for code stability.
    - Validates JSON schema before returning.
    - returns a Python Dictionary, not a string.
    """

    @staticmethod
    def load_hypotheses_from_file(memory: Memory) -> list:
        """从 memory.hypothesesDir 读取 output.json 文件"""
        with open(memory.hypothesesdir, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def extract_modified_hypotheses(hypotheses_data: list) -> Dict[str, list]:
        """
        从 output.json 数据中提取 entity -> modified_hypotheses 的映射。
        
        Args:
            hypotheses_data: load_hypotheses_from_file 返回的列表
            
        Returns:
            Dict[str, list]: key 是 entity，value 是 modified_hypotheses 列表
        """
        return {item["entity"]: item["modified_hypotheses"] for item in hypotheses_data}

    _RUBRIC_ANCHORS = r"""
    You must evaluate the hypothesis using the following 6 criteria.

    For EACH criterion, give:
    - Score: integer in [1, 5]
    - Rationale: concise, evidence-grounded explanation
    - Concerns: list of concrete issues (if any)
    - Suggestions: list of actionable fixes (must be specific)

    Scoring anchors (1–5):

    (1) Novelty (newness / non-triviality)
    1. Almost no novelty: common trope; trivial recombination; no new mechanism or prediction.
    2. Weak novelty: minor variant; straightforward extension of known ideas.
    3. Moderate novelty: identifiable new linkage, but close to existing work or not clearly differentiated.
    4. Clear novelty: under-discussed mechanism/structure; differences from prior work are explicit.
    5. High novelty: non-obvious mechanism/problem framing with testable predictions and potential impact.

    (2) Plausibility (mechanistic coherence / self-consistency)
    1. Implausible: contradicts basic biomedical/clinical knowledge; broken causal chain.
    2. Low plausibility: relies on many unstated strong assumptions; causal direction confused.
    3. Partly plausible: mostly coherent but key steps missing; boundary conditions unclear.
    4. Plausible: coherent chain; assumptions stated; minor weak links remain.
    5. Highly plausible: complete, coherent mechanism; handles alternatives/counterexamples.

    (3) Grounding (evidence traceability / support)
    1. No evidence: mostly speculation; references missing or irrelevant.
    2. Weak evidence: scattered hints; key claims not supported; overgeneralization risk.
    3. Moderate: some claims supported; gaps remain; claim↔evidence mapping imprecise.
    4. Strong: most key claims supported by cited evidence/data/path reasoning.
    5. Very strong: complete traceable chain (claim↔evidence↔path/data) with uncertainty calibration.

    (4) Testability (operational verification)
    1. Not testable: too abstract; depends on unavailable data/conditions; no measurable endpoints.
    2. Hard to test: high cost/uncontrollable; endpoints/metrics unclear.
    3. Partly testable: directions exist but controls/metrics/expected outcomes are vague.
    4. Testable: executable experiments/analyses with clear controls and discriminative outcomes.
    5. Highly testable: multiple independent validation routes with clear metrics, expected outcomes, reproducibility.

    (5) Specificity (precision / boundary conditions)
    1. Too vague: only “may relate/affect”; lacks actor/condition/direction/scope.
    2. Somewhat vague: partial objects; missing constraints (context, timing, magnitude).
    3. Moderate: actors and direction stated; parameters/scenarios/exclusions incomplete.
    4. Specific: “who under what conditions via what mechanism affects what”; scope bounded.
    5. Very specific: directly translatable to an experimental hypothesis with constraints and non-applicable cases.

    (6) Safety & Ethics (risk, compliance, responsible framing)
    1. High risk / non-compliant: unsafe procedures, harmful advice, or unethical guidance.
    2. Significant risk: may enable unsafe actions; ignores IRB/privacy/biosafety constraints.
    3. Manageable but needs additions: add compliance prerequisites (IRB, de-identification, biosafety).
    4. Safe: risks identified and mitigations provided; feasible under compliance framework.
    5. Very safe: explicit compliance path + clear limitations + safer alternatives.

    General requirements:
    - Be strict: do not inflate scores.
    - Suggestions must be executable (e.g., “Add boundary condition X”, “Define metric Y”).
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        memory: Optional[Memory] = None,
        context: Optional[Dict[str, Any]] = None,
    ):

        # 构造严格的 JSON 模板
        # 注意：这里定义了 EditInstructions 的具体结构，方便 RevisionAgent 自动执行
        json_template = json.dumps({
            "Novelty": {
                "score": "X/5",
                "rationale": "...",
                "concerns": ["concern 1", "concern 2"],
                "suggestions": ["suggestion 1", "suggestion 2"]
            },
            "Plausibility": {
                "score": "X/5",
                "rationale": "...",
                "concerns": [],
                "suggestions": []
            },
            "Grounding": {
                "score": "X/5",
                "rationale": "...",
                "concerns": [],
                "suggestions": []
            },
            "Testability": {
                "score": "X/5",
                "rationale": "...",
                "concerns": [],
                "suggestions": []
            },
            "Specificity": {
                "score": "X/5",
                "rationale": "...",
                "concerns": [],
                "suggestions": []
            },
            "Safety & Ethics": {
                "score": "X/5",
                "rationale": "...",
                "concerns": [],
                "suggestions": []
            },
            "Overall Summary": {
                "Strengths": ["..."],
                "Weaknesses": ["..."],
                "Priority must-fix": ["..."],
                "Nice-to-fix": ["..."],
                "Risk flags": ["..."],
                "Edit instructions": ["..."]
            }
        }, indent=2)

        system_prompt = f"""You are a rigorous scientific reviewer.

        {self._RUBRIC_ANCHORS}

        You will receive:
        - A hypothesis (as JSON text)
        - Optional context (as JSON text)

        IMPORTANT OUTPUT RULES:
        1. Output MUST be a valid JSON object.
        2. Do NOT output Markdown code fences (like ```json).
        3. The JSON structure MUST exactly match the following example:

        {json_template}
        """
        super().__init__(client, model_name, system_prompt)

        self.hypothesis=self.extract_modified_hypotheses(self.load_hypotheses_from_file(memory))
        self.memory = memory or get_memory()
        self.context = context or {}

    def process(self) -> Dict[str, Any]:
        for hypothesis in self.hypothesis:
            self.call_for_each_hypothesis(hypothesis)

    def call_for_each_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent logic:
        1. Calls LLM.
        2. Cleans and extracts JSON from response.
        3. Validates schema.
        4. Returns structured Dict.
        """
        hypothesis_text = json.dumps(hypothesis, ensure_ascii=False, indent=2)

        user_message = (
            "Hypothesis (JSON):\n"
            f"{hypothesis_text}\n\n"
            "Context (JSON):\n"
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
        Helper to strip markdown code blocks from LLM output.
        """
        text = text.strip()
        # Remove ```json or ``` at the start
        text = re.sub(r"^```(json)?\s*", "", text, flags=re.IGNORECASE)
        # Remove ``` at the end
        text = re.sub(r"\s*```$", "", text)
        return text