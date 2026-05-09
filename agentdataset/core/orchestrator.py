"""
AgentDataset Orchestrator
The Autonomous Engine (Brain)
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, List
import pandas as pd
from agentdataset.models.schemas import SessionContext, Parameters, FidelityReport, DiscoveryResult
from agentdataset.core.discovery import DiscoveryAgent, PDF_PATH_PREFIX
from agentdataset.core.extractor import Extractor
from agentdataset.core.synthesizer import Synthesizer
from agentdataset.core.validator import Validator

logger = logging.getLogger(__name__)

MAX_NOISE = 2.0
MIN_NOISE = 0.01
PATIENCE = 2  # non-improvement streak length that triggers a pivot

class Orchestrator:
    def __init__(self, session_id: str, base_dir: str = "sessions", model: str = "gpt-4o",
                 api_key: str = "", env_var: str = "OPENAI_API_KEY"):
        self.context = SessionContext(
            session_id=session_id,
            path=str(Path(base_dir) / session_id)
        )
        os.makedirs(self.context.path, exist_ok=True)

        self.discovery = DiscoveryAgent()
        self.extractor = Extractor(model=model, api_key=api_key, env_var=env_var)
        self.synthesizer = Synthesizer()
        self.validator = Validator()

        self.best_score = 0.0
        self.best_params: Optional[Parameters] = None
        self.best_data: Optional[pd.DataFrame] = None

    def run_discovery(self, query: str) -> List[DiscoveryResult]:
        """Phase 0: Discovery."""
        return self.discovery.search(query)

    def process_source(self, result: DiscoveryResult) -> Parameters:
        """Phase 1: Extraction."""
        content = self.discovery.fetch_content(result)

        if content.startswith(PDF_PATH_PREFIX):
            pdf_path = content[len(PDF_PATH_PREFIX):]
            try:
                text = self.extractor.pdf_to_markdown(pdf_path)
            finally:
                # Clean up temp file regardless of extraction outcome
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass
        else:
            text = content

        return self.extractor.extract_parameters(text, result.title)

    def run_optimization_loop(self, parameters: Parameters, iterations: int = 5):
        """Phase 2 & 3: The Engine (Synthesis-Validation Loop).

        Noise pivot strategy — patience + reset:
          - Every PATIENCE consecutive non-improvements  → exploit: halve noise
          - Every PATIENCE*2 consecutive non-improvements → reset to initial noise
          - Single non-improvement steps (streak % PATIENCE != 0) → explore: raise noise
        """
        current_params = parameters
        initial_noise = 0.1
        noise_level = initial_noise
        no_improve_streak = 0

        for i in range(iterations):
            logger.info("Loop %d/%d (noise=%.4f)...", i + 1, iterations, noise_level)

            # Synthesis
            df = self.synthesizer.synthesize(current_params, noise_level=noise_level)

            # Validation
            report = self.validator.validate(df, current_params)
            logger.info("  Fidelity Score: %s", report.overall_score)

            # Ratchet Logic
            if report.overall_score > self.best_score:
                logger.info("  [KEEP] New best score!")
                self.best_score = report.overall_score
                self.best_params = current_params
                self.best_data = df
                no_improve_streak = 0

                # Save artifacts
                df.to_csv(Path(self.context.path) / "data.csv", index=False)
                with open(Path(self.context.path) / "parameters.json", "w") as f:
                    f.write(current_params.model_dump_json(indent=2))

                # Generate and save DATACARD
                datacard = self.validator.generate_datacard(report, current_params, df)
                with open(Path(self.context.path) / "DATACARD.md", "w") as f:
                    f.write(datacard)
            else:
                no_improve_streak += 1
                full_cycle = PATIENCE * 2

                if no_improve_streak % full_cycle == 0:
                    # Full cycle with no gain — reset to initial noise
                    noise_level = initial_noise
                    logger.info("  [DISCARD] Streak=%d — reset noise to %.4f",
                                no_improve_streak, noise_level)
                elif no_improve_streak % PATIENCE == 0:
                    # Exploit phase: tighten noise to improve fit
                    noise_level = max(noise_level * 0.5, MIN_NOISE)
                    logger.info("  [DISCARD] Streak=%d — exploit: noise → %.4f",
                                no_improve_streak, noise_level)
                else:
                    # Explore phase: expand noise for more variance
                    noise_level = min(noise_level * 1.1, MAX_NOISE)
                    logger.info("  [DISCARD] Streak=%d — explore: noise → %.4f",
                                no_improve_streak, noise_level)

        return self.best_score, self.best_data
