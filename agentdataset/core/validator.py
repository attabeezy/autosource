"""
AgentDataset Validator
Data + Parameters -> FidelityReport
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from agentdataset.models.schemas import Parameters, FidelityReport

MIN_KS_PVALUE = 0.05
MIN_FIDELITY_SCORE = 90.0

class Validator:
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "ks_pvalue": 0.05,
            "corr_similarity": 0.8,
            "fidelity_score": 90.0
        }

    def compute_ks_test(self, df: pd.DataFrame, parameters: Parameters) -> Dict[str, float]:
        """Compute KS-test p-values."""
        results = {}
        for name, var_params in parameters.variables.items():
            if name not in df.columns: continue
            data = df[name].values
            
            # Theoretical CDF mapping — use default-arg capture to avoid late-binding closure bug
            if var_params.distribution == "normal":
                theoretical_cdf = lambda x, m=var_params.mean, s=var_params.std: stats.norm.cdf(x, loc=m, scale=s)
            elif var_params.distribution == "uniform":
                low = var_params.min if var_params.min is not None else var_params.mean - 2*var_params.std
                high = var_params.max if var_params.max is not None else var_params.mean + 2*var_params.std
                theoretical_cdf = lambda x, l=low, h=high: stats.uniform.cdf(x, loc=l, scale=h-l)
            else:
                theoretical_cdf = lambda x, m=var_params.mean, s=var_params.std: stats.norm.cdf(x, loc=m, scale=s)
            
            _, p_val = stats.kstest(data, theoretical_cdf)
            results[name] = float(p_val)
        return results

    def compute_correlation_similarity(self, df: pd.DataFrame, parameters: Parameters) -> float:
        """Compute cosine similarity of correlation matrices."""
        var_names = list(parameters.variables.keys())
        if len(var_names) < 2: return 1.0
        
        synthetic_corr = df[var_names].corr().fillna(0).values
        target_corr = np.eye(len(var_names))
        
        for key, corr_params in parameters.correlations.items():
            v1, v2 = corr_params.var1, corr_params.var2
            if v1 in var_names and v2 in var_names:
                idx1, idx2 = var_names.index(v1), var_names.index(v2)
                target_corr[idx1, idx2] = corr_params.correlation
                target_corr[idx2, idx1] = corr_params.correlation
        
        cos_sim = np.sum(synthetic_corr * target_corr) / (
            np.sqrt(np.sum(synthetic_corr**2)) * np.sqrt(np.sum(target_corr**2))
        )
        return float(max(0.0, min(1.0, (cos_sim + 1) / 2)))

    def validate(self, df: pd.DataFrame, parameters: Parameters) -> FidelityReport:
        """Run full validation suite."""
        ks_pvalues = self.compute_ks_test(df, parameters)
        corr_sim = self.compute_correlation_similarity(df, parameters)
        
        # Simple bias score based on mean deviation
        bias_count = 0
        for name, var_params in parameters.variables.items():
            if name in df.columns:
                if abs(df[name].mean() - var_params.mean) / (var_params.mean or 1.0) > 0.2:
                    bias_count += 1
        bias_score = 1.0 - (bias_count / len(parameters.variables)) if parameters.variables else 1.0
        
        # Overall Score
        ks_score = (min(1.0, sum(ks_pvalues.values()) / (len(ks_pvalues) or 1) / 0.05)) * 100
        overall_score = 0.4 * ks_score + 0.4 * (corr_sim * 100) + 0.2 * (bias_score * 100)
        
        return FidelityReport(
            overall_score=round(overall_score, 2),
            ks_score=round(ks_score, 2),
            corr_score=round(corr_sim * 100, 2),
            bias_score=round(bias_score * 100, 2),
            ks_pvalues=ks_pvalues,
            bias_details={}, # Simplified
            privacy_details={"avg_min_dist": 0.1}, # Placeholder
            approved=overall_score >= self.thresholds["fidelity_score"]
        )

    def generate_datacard(self, report: FidelityReport, parameters: Parameters, df: pd.DataFrame) -> str:
        """Generate a Markdown DATACARD report."""
        source = parameters.meta.source
        extracted_at = parameters.meta.extracted_at
        
        var_details = []
        for name, p_val in report.ks_pvalues.items():
            status = "[PASS]" if p_val >= self.thresholds["ks_pvalue"] else "[FAIL]"
            var_details.append(f"- {status} **{name}**: KS p-value={p_val:.4f}")

        card = f"""# DATACARD: Synthetic Dataset

## Overview
- **Source**: {source}
- **Generated**: {extracted_at}
- **Rows**: {len(df)}
- **Columns**: {len(df.columns)}
- **Fidelity Score**: **{report.overall_score}**/100

## Statistical Fidelity

### Distribution Fit (KS-test)
{"\n".join(var_details)}

### Correlation Preservation
**Correlation Similarity**: {report.corr_score/100:.4f}

## Bias Detection
**Bias Score**: {report.bias_score:.2f}%

---
*DATACARD generated by AgentDataset*
"""
        return card
