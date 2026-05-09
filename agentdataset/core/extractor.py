"""
AgentDataset Extractor
PDF/Markdown -> Parameters (Pydantic)
"""

import json
import logging
import os
import time
from typing import Dict, Any
import re
import fitz  # PyMuPDF
from litellm import completion
from agentdataset.models.schemas import Parameters, VariableParams, CorrelationParams, MetaParams

logger = logging.getLogger(__name__)

CAVEMAN_PROMPT = """
You expert statistician.
Strip prose.
Extract variables, distributions, correlations.
Output strict JSON matching this schema exactly:
{
  "variables": {
    "<name>": {"distribution": "normal|uniform|gamma", "mean": 0.0, "std": 1.0, "min": null, "max": null}
  },
  "correlations": {
    "<key>": {"var1": "<name>", "var2": "<name>", "correlation": 0.5, "direction": "positive|negative"}
  }
}
No fluff. No greeting. Output JSON only.
"""

# Separator between a keyword and its numeric value: "= 3.5", ": 3.5", "is 3.5", "of 3.5"
_SEP = r"\s*(?:is|=|:|of)?\s*"

# Pattern A: mean then std (e.g. "mean = 3.5 ... std = 1.2", "mean is 3.5 ... standard deviation is 1.2")
_PATTERN_MEAN_STD = re.compile(
    r"(?:mean|average|μ)" + _SEP + r"(\d+(?:\.\d+)?)"
    r".{0,80}?"
    r"(?:std|s\.?d\.?|standard\s+deviation|σ)" + _SEP + r"(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Pattern B: std then mean (e.g. "SD = 1.2, mean = 3.5")
_PATTERN_STD_MEAN = re.compile(
    r"(?:std|s\.?d\.?|standard\s+deviation|σ)" + _SEP + r"(\d+(?:\.\d+)?)"
    r".{0,80}?"
    r"(?:mean|average|μ)" + _SEP + r"(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class Extractor:
    def __init__(self, model: str = "gpt-4o", api_key: str = ""):
        self.model = model
        self.api_key = api_key

    def pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF to clean text."""
        doc = fitz.open(pdf_path)
        sections = []
        for page in doc:
            text = page.get_text("text")
            sections.append(text)
        doc.close()
        return "\n\n".join(sections)

    def _extract_with_llm(self, text: str) -> Dict[str, Any]:
        """Call litellm and return parsed JSON dict. Raises on any failure."""
        if self.api_key:
            os.environ["OPENAI_API_KEY"] = self.api_key

        prompt = f"Extract stats from this text:\n\n{text[:10000]}"
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": CAVEMAN_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def _parse_llm_result(self, data: Dict[str, Any]) -> tuple[Dict, Dict]:
        """Convert raw LLM JSON dict into (variables, correlations) dicts of Pydantic models."""
        variables: Dict[str, VariableParams] = {}
        for name, v in data.get("variables", {}).items():
            mean = float(v.get("mean", 0.0))
            std = float(v.get("std", 1.0))
            variables[name] = VariableParams(
                name=name,
                distribution=v.get("distribution", "normal"),
                mean=mean,
                std=std,
                min=float(v["min"]) if v.get("min") is not None else mean - 3 * std,
                max=float(v["max"]) if v.get("max") is not None else mean + 3 * std,
            )

        correlations: Dict[str, CorrelationParams] = {}
        for key, c in data.get("correlations", {}).items():
            correlations[key] = CorrelationParams(
                var1=c["var1"],
                var2=c["var2"],
                correlation=float(c.get("correlation", 0.0)),
                direction=c.get("direction", "positive"),
            )

        return variables, correlations

    def _extract_with_regex(self, text: str) -> tuple[Dict, Dict]:
        """Fallback regex extraction returning (variables, correlations)."""
        variables: Dict[str, VariableParams] = {}
        seen = set()

        for match in _PATTERN_MEAN_STD.finditer(text):
            mean, std = float(match.group(1)), float(match.group(2))
            key = (mean, std)
            if key not in seen:
                seen.add(key)
                name = f"var_{len(variables) + 1}"
                variables[name] = VariableParams(
                    name=name, distribution="normal",
                    mean=mean, std=std,
                    min=mean - 3 * std, max=mean + 3 * std,
                )

        for match in _PATTERN_STD_MEAN.finditer(text):
            std, mean = float(match.group(1)), float(match.group(2))
            key = (mean, std)
            if key not in seen:
                seen.add(key)
                name = f"var_{len(variables) + 1}"
                variables[name] = VariableParams(
                    name=name, distribution="normal",
                    mean=mean, std=std,
                    min=mean - 3 * std, max=mean + 3 * std,
                )

        return variables, {}

    def extract_parameters(self, text: str, source_name: str) -> Parameters:
        """Extract statistical parameters — LLM first, regex fallback."""
        method = "regex_fallback"
        variables: Dict[str, VariableParams] = {}
        correlations: Dict[str, CorrelationParams] = {}

        if self.api_key or os.environ.get("OPENAI_API_KEY"):
            try:
                data = self._extract_with_llm(text)
                variables, correlations = self._parse_llm_result(data)
                method = "llm"
                logger.info("LLM extraction succeeded: %d variables, %d correlations",
                            len(variables), len(correlations))
            except Exception as e:
                logger.warning("LLM extraction failed (%s); falling back to regex.", e)
                variables, correlations = self._extract_with_regex(text)
        else:
            variables, correlations = self._extract_with_regex(text)

        return Parameters(
            variables=variables,
            correlations=correlations,
            meta=MetaParams(
                source=source_name,
                extracted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                extraction_method=method,
            ),
        )

    def check_statistical_density(self, text: str) -> float:
        """Assess if the text is statistically dense (numbers vs words)."""
        words = len(re.findall(r'\w+', text))
        numbers = len(re.findall(r'\d+', text))
        if words == 0:
            return 0.0
        return numbers / words
