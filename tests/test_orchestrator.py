import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from agentdataset.core.orchestrator import Orchestrator
from agentdataset.core.discovery import PDF_PATH_PREFIX
from agentdataset.models.schemas import Parameters, VariableParams, MetaParams, DiscoveryResult

@pytest.fixture
def mock_orchestrator(tmp_path):
    with patch('agentdataset.core.orchestrator.DiscoveryAgent'), \
         patch('agentdataset.core.orchestrator.Extractor'), \
         patch('agentdataset.core.orchestrator.Synthesizer'), \
         patch('agentdataset.core.orchestrator.Validator'):
        orc = Orchestrator(session_id="test_session", base_dir=str(tmp_path))
        return orc

def test_orchestrator_init(mock_orchestrator):
    assert mock_orchestrator.context.session_id == "test_session"

def test_run_discovery(mock_orchestrator):
    mock_orchestrator.discovery.search.return_value = [DiscoveryResult(title="T", url="U", source_type="pdf", relevance_score=1.0)]
    results = mock_orchestrator.run_discovery("query")
    assert len(results) == 1
    mock_orchestrator.discovery.search.assert_called_once_with("query")

def test_process_source_plain_text(mock_orchestrator):
    res = DiscoveryResult(title="T", url="U", source_type="html", relevance_score=1.0)
    mock_orchestrator.discovery.fetch_content.return_value = "plain text content"
    mock_orchestrator.extractor.extract_parameters.return_value = Parameters(
        variables={}, correlations={}, meta=MetaParams(source="S", extracted_at="N")
    )
    params = mock_orchestrator.process_source(res)
    assert isinstance(params, Parameters)
    mock_orchestrator.extractor.pdf_to_markdown.assert_not_called()
    mock_orchestrator.extractor.extract_parameters.assert_called_once_with("plain text content", "T")


def test_process_source_pdf_path(mock_orchestrator, tmp_path):
    """pdf:// prefix triggers pdf_to_markdown then cleans up the temp file."""
    fake_pdf = tmp_path / "doc.pdf"
    fake_pdf.write_bytes(b"PDF")

    res = DiscoveryResult(title="T", url="U", source_type="pdf", relevance_score=1.0)
    mock_orchestrator.discovery.fetch_content.return_value = PDF_PATH_PREFIX + str(fake_pdf)
    mock_orchestrator.extractor.pdf_to_markdown.return_value = "parsed markdown"
    mock_orchestrator.extractor.extract_parameters.return_value = Parameters(
        variables={}, correlations={}, meta=MetaParams(source="S", extracted_at="N")
    )

    params = mock_orchestrator.process_source(res)

    assert isinstance(params, Parameters)
    mock_orchestrator.extractor.pdf_to_markdown.assert_called_once_with(str(fake_pdf))
    mock_orchestrator.extractor.extract_parameters.assert_called_once_with("parsed markdown", "T")
    assert not fake_pdf.exists(), "Temp PDF should be deleted after processing"

def test_run_optimization_loop(mock_orchestrator):
    params = Parameters(
        variables={"v1": VariableParams(name="v1")},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N")
    )
    df = pd.DataFrame({"v1": [1, 2, 3]})
    mock_orchestrator.synthesizer.synthesize.return_value = df

    report = MagicMock()
    report.overall_score = 95.0
    mock_orchestrator.validator.validate.return_value = report
    mock_orchestrator.validator.generate_datacard.return_value = "mock datacard content"

    score, data = mock_orchestrator.run_optimization_loop(params, iterations=1)

    assert score == 95.0
    assert data.equals(df)
    assert mock_orchestrator.best_score == 95.0


def test_noise_pivot_strategy(mock_orchestrator):
    """Streak counter drives explore → exploit → reset transitions."""
    from agentdataset.core.orchestrator import PATIENCE, MAX_NOISE, MIN_NOISE

    params = Parameters(
        variables={"v1": VariableParams(name="v1")},
        correlations={},
        meta=MetaParams(source="S", extracted_at="N")
    )
    df = pd.DataFrame({"v1": [1, 2, 3]})
    mock_orchestrator.synthesizer.synthesize.return_value = df

    # Scores: first improves (streak reset), then 4 consecutive non-improvements
    # to exercise explore (streak 1), exploit (streak 2), explore (streak 3), reset (streak 4)
    scores = [95.0, 80.0, 79.0, 78.0, 77.0]
    reports = [MagicMock(overall_score=s) for s in scores]
    mock_orchestrator.validator.validate.side_effect = reports
    mock_orchestrator.validator.generate_datacard.return_value = ""

    # Capture noise_level passed to synthesize on each call
    noise_calls = []
    original_synthesize = mock_orchestrator.synthesizer.synthesize
    def capture_noise(params, noise_level):
        noise_calls.append(noise_level)
        return df
    mock_orchestrator.synthesizer.synthesize.side_effect = capture_noise

    mock_orchestrator.run_optimization_loop(params, iterations=5)

    initial = 0.1
    # iter 0: noise = 0.1, score 95 → keep, streak resets to 0
    assert noise_calls[0] == pytest.approx(initial)
    # iter 1: streak=1 (explore) → noise *= 1.1
    assert noise_calls[1] == pytest.approx(initial)
    explore_noise = initial * 1.1
    # iter 2: streak=2 (exploit, streak % PATIENCE == 0) → noise *= 0.5
    assert noise_calls[2] == pytest.approx(explore_noise)
    exploit_noise = max(explore_noise * 0.5, MIN_NOISE)
    # iter 3: streak=3 (explore again)
    assert noise_calls[3] == pytest.approx(exploit_noise)
    # iter 4: streak=4 (full cycle, streak % (PATIENCE*2) == 0) → reset
    assert noise_calls[4] == pytest.approx(min(exploit_noise * 1.1, MAX_NOISE))
