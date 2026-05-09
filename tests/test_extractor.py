import json
import pytest
from unittest.mock import MagicMock, patch
from agentdataset.core.extractor import Extractor


def test_extractor_init():
    ext = Extractor(model="test-model", api_key="sk-test")
    assert ext.model == "test-model"
    assert ext.api_key == "sk-test"


@patch('agentdataset.core.extractor.fitz')
def test_pdf_to_markdown(mock_fitz):
    mock_doc = MagicMock()
    mock_page = MagicMock()
    mock_page.get_text.return_value = "page text"
    mock_doc.__iter__.return_value = [mock_page]
    mock_fitz.open.return_value = mock_doc

    ext = Extractor()
    md = ext.pdf_to_markdown("test.pdf")
    assert md == "page text"


@patch('agentdataset.core.extractor.completion')
def test_extract_parameters_llm_path(mock_completion):
    """LLM path: completion returns valid JSON, variables populated from it."""
    llm_json = {
        "variables": {
            "income": {"distribution": "normal", "mean": 50000.0, "std": 12000.0, "min": None, "max": None}
        },
        "correlations": {}
    }
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(llm_json)
    mock_completion.return_value = mock_response

    ext = Extractor(model="gpt-4o", api_key="sk-test")
    params = ext.extract_parameters("Some research text.", "test_source")

    assert params.meta.extraction_method == "llm"
    assert "income" in params.variables
    assert params.variables["income"].mean == 50000.0
    assert params.variables["income"].std == 12000.0
    assert params.meta.source == "test_source"


@patch('agentdataset.core.extractor.completion')
def test_extract_parameters_llm_fallback(mock_completion):
    """LLM failure falls back to regex and labels method correctly."""
    mock_completion.side_effect = Exception("API error")

    ext = Extractor(model="gpt-4o", api_key="sk-test")
    text = "The mean is 10.5 and the standard deviation is 2.1."
    params = ext.extract_parameters(text, "test_source")

    assert params.meta.extraction_method == "regex_fallback"
    assert len(params.variables) == 1
    var = list(params.variables.values())[0]
    assert var.mean == 10.5
    assert var.std == 2.1


def test_extract_parameters_regex_no_key():
    """No API key → skips LLM entirely, uses regex."""
    import os
    os.environ.pop("OPENAI_API_KEY", None)

    ext = Extractor()  # no api_key
    text = "The mean is 10.5 and the standard deviation is 2.1."
    params = ext.extract_parameters(text, "test_source")

    assert params.meta.extraction_method == "regex_fallback"
    assert len(params.variables) == 1
    var = params.variables["var_1"]
    assert var.mean == 10.5
    assert var.std == 2.1
    assert params.meta.source == "test_source"


def test_check_statistical_density():
    ext = Extractor()
    text = "Word word 123 456 word"
    density = ext.check_statistical_density(text)
    # Regex findall r'\w+': ['Word', 'word', '123', '456', 'word'] -> 5
    # Regex findall r'\d+': ['123', '456'] -> 2
    assert density == 0.4
