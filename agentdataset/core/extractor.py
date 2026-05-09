"""
AgentDataset Extractor
PDF/Markdown -> Parameters (Pydantic)
"""

import time
from pathlib import Path
from typing import Dict, Any
import re
import fitz  # PyMuPDF
from litellm import completion
from agentdataset.models.schemas import Parameters, VariableParams, CorrelationParams, MetaParams

CAVEMAN_PROMPT = """
You expert statistician.
Strip prose.
Extract variables, distributions, correlations.
Output strict JSON.
No fluff. No greeting.
[thing] [action] [reason].
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
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF to clean Markdown."""
        doc = fitz.open(pdf_path)
        sections = []
        for page in doc:
            text = page.get_text("text")
            sections.append(text)
        doc.close()
        return "\n\n".join(sections)

    def extract_parameters(self, text: str, source_name: str) -> Parameters:
        """Extract statistical parameters using LLM with Caveman protocol."""
        prompt = f"Extract stats from this text:\n\n{text[:10000]}"  # Truncate for safety

        # In a real scenario, we'd use litellm here.
        # response = completion(
        #     model=self.model,
        #     messages=[
        #         {"role": "system", "content": CAVEMAN_PROMPT},
        #         {"role": "user", "content": prompt}
        #     ],
        #     response_format={"type": "json_object"}
        # )
        # data = json.loads(response.choices[0].message.content)

        # Fallback to regex until LLM integration is enabled
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

        return Parameters(
            variables=variables,
            correlations={},
            meta=MetaParams(
                source=source_name,
                extracted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                extraction_method="regex_fallback",
            ),
        )

    def check_statistical_density(self, text: str) -> float:
        """Assess if the text is statistically dense (numbers vs words)."""
        words = len(re.findall(r'\w+', text))
        numbers = len(re.findall(r'\d+', text))
        if words == 0:
            return 0.0
        return numbers / words
