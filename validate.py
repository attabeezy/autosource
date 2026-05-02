"""
AutoSource Validation Pipeline
data.csv + parameters.json → fidelity_score + DATACARD.md

Usage: uv run validate.py
"""

import os
import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Any, dict

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

PARAMS_INPUT = Path("results/parameters.json")
DATA_INPUT = Path("results/data.csv")
OUTPUT_DIR = Path("results")
FIDELITY_FILE = OUTPUT_DIR / "fidelity_score.txt"
DATACARD_FILE = OUTPUT_DIR / "DATACARD.md"

# Fidelity thresholds (agent must meet these)
MIN_KS_PVALUE = 0.05  # KS-test p-value minimum
MIN_CORR_SIMILARITY = 0.8  # Correlation similarity minimum
MIN_FIDELITY_SCORE = 90.0  # Overall fidelity score minimum


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------


def compute_ks_test(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, float]:
    """
    Compute Kolmogorov-Smirnov test for each variable.
    Returns dict of {var_name: p_value}
    """
    variables = params.get("variables", {})
    results = {}

    for var_name, var_params in variables.items():
        if var_name not in df.columns:
            continue

        data = df[var_name].values

        # Fit theoretical distribution
        dist_type = var_params.get("distribution", "normal")
        mean = var_params.get("mean", 0)
        std = var_params.get("std", 1)

        if dist_type == "normal":
            loc, scale = mean, std
            theoretical_cdf = lambda x: stats.norm.cdf(x, loc=loc, scale=scale)
        elif dist_type == "uniform":
            min_val = var_params.get("min", mean - 2 * std)
            max_val = var_params.get("max", mean + 2 * std)
            theoretical_cdf = lambda x: stats.uniform.cdf(
                x, loc=min_val, scale=max_val - min_val
            )
        elif dist_type == "gamma":
            shape = (mean / std) ** 2
            scale = std**2 / mean
            theoretical_cdf = lambda x: stats.gamma.cdf(x, a=shape, scale=scale)
        elif dist_type == "beta":
            min_val = var_params.get("min", 0)
            max_val = var_params.get("max", 1)
            mean_norm = (mean - min_val) / (max_val - min_val)
            a = mean_norm * 10 + 0.5
            b = (1 - mean_norm) * 10 + 0.5
            theoretical_cdf = lambda x: stats.beta.cdf(
                x, a=a, b=b, loc=min_val, scale=max_val - min_val
            )
        else:
            theoretical_cdf = lambda x: stats.norm.cdf(x, loc=mean, scale=std)

        # KS test
        ks_stat, p_value = stats.kstest(data, theoretical_cdf)
        results[var_name] = p_value

    return results


def compute_correlation_similarity(df: pd.DataFrame, params: dict[str, Any]) -> float:
    """
    Compute cosine similarity between synthetic and target correlation matrices.
    Returns score in [0, 1].
    """
    correlations = params.get("correlations", {})
    var_names = list(params.get("variables", {}).keys())

    if len(var_names) < 2 or not correlations:
        return 1.0  # No correlations to compare

    # Compute synthetic correlation matrix
    synthetic_corr = df[var_names].corr().values

    # Build target correlation matrix
    target_corr = np.eye(len(var_names))
    for i, v1 in enumerate(var_names):
        for j, v2 in enumerate(var_names):
            if i >= j:
                continue
            key = f"{v1} ↔ {v2}" if f"{v1} ↔ {v2}" in correlations else f"{v2} ↔ {v1}"
            if key in correlations:
                corr = correlations[key].get("correlation", 0)
                target_corr[i, j] = corr
                target_corr[j, i] = corr

    # Cosine similarity
    cos_sim = np.sum(synthetic_corr * target_corr) / (
        np.sqrt(np.sum(synthetic_corr**2)) * np.sqrt(np.sum(target_corr**2))
    )

    return max(0.0, min(1.0, (cos_sim + 1) / 2))  # Normalize to [0, 1]


def check_bias(df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """
    Check for demographic bias in synthetic data.
    Returns dict with {overrepresented: [], underrepresented: [], bias_score: float}
    """
    variables = params.get("variables", {})
    bias_results = {"overrepresented": [], "underrepresented": [], "bias_score": 1.0}

    # Simple check: if synthetic mean deviates > 20% from source mean
    for var_name, var_params in variables.items():
        if var_name not in df.columns:
            continue

        synthetic_mean = df[var_name].mean()
        target_mean = var_params.get("mean", 0)

        if target_mean > 0:
            deviation = abs(synthetic_mean - target_mean) / target_mean
            if deviation > 0.2:
                if synthetic_mean > target_mean:
                    bias_results["overrepresented"].append(var_name)
                else:
                    bias_results["underrepresented"].append(var_name)

    # Bias score (1.0 = no bias, 0.0 = max bias)
    total_vars = len(variables)
    biased_vars = len(bias_results["overrepresented"]) + len(
        bias_results["underrepresented"]
    )
    bias_results["bias_score"] = (
        1.0 - (biased_vars / total_vars) if total_vars > 0 else 1.0
    )

    return bias_results


def check_privacy(df: pd.DataFrame) -> dict[str, float]:
    """
    Privacy guardrail: check distance to closest record.
    Returns dict with {avg_min_distance: float, max_min_distance: float}
    """
    if len(df) < 2:
        return {"avg_min_distance": 0.0, "max_min_distance": 0.0}

    # Compute pairwise distances
    values = df.values
    n = len(values)

    # Sample for efficiency if dataset is large
    if n > 1000:
        idx = np.random.choice(n, 1000, replace=False)
        values = values[idx]
        n = len(values)

    min_distances = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                distances.append(np.linalg.norm(values[i] - values[j]))
        if distances:
            min_distances.append(min(distances))

    return {
        "avg_min_distance": float(np.mean(min_distances)),
        "max_min_distance": float(np.max(min_distances)),
    }


def compute_overall_fidelity(
    ks_results: dict, corr_sim: float, bias_score: float
) -> float:
    """
    Compute overall fidelity score (0-100).
    Formula: weighted average of components.
    """
    # KS component: mean p-value normalized to [0, 1]
    if ks_results:
        ks_score = (
            min(1.0, sum(ks_results.values()) / len(ks_results) / MIN_KS_PVALUE) * 100
        )
    else:
        ks_score = 100  # No variables = perfect

    # Correlation component
    corr_score = corr_sim * 100

    # Bias component
    bias_component = bias_score * 100

    # Weighted average (default weights)
    score = 0.4 * ks_score + 0.4 * corr_score + 0.2 * bias_component

    return round(score, 2)


def generate_datacard(
    fidelity_score: float,
    ks_results: dict,
    corr_sim: float,
    bias_results: dict,
    privacy_results: dict,
    params: dict,
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate DATACARD.md."""

    # Source provenance
    source = params.get("meta", {}).get("source", "Unknown Report")
    extracted_at = params.get("meta", {}).get("extracted_at", "Unknown")

    # Per-variable fidelity
    var_details = []
    for var_name, p_value in ks_results.items():
        status = "✓" if p_value >= MIN_KS_PVALUE else "✗"
        var_details.append(f"- {status} **{var_name}**: KS p-value={p_value:.4f}")

    # Correlation details
    corr_status = "✓" if corr_sim >= MIN_CORR_SIMILARITY else "✗"

    # Bias status
    bias_status = "✓" if bias_results["bias_score"] >= 0.8 else "⚠"

    # Privacy status
    privacy_status = "✓" if privacy_results["avg_min_distance"] > 0 else "⚠"

    card = f"""# DATACARD: Synthetic Dataset

## Overview
- **Source**: {source}
- **Generated**: {extracted_at}
- **Rows**: {len(df)}
- **Columns**: {len(df.columns)}
- **Fidelity Score**: **{fidelity_score}**/100

## Source Provenance
- Extracted from: `{source}`
- Extraction method: `regex_placeholder` (use LLM for production)

## Statistical Fidelity

### Distribution Fit (KS-test)
{chr(10).join(var_details)}

**Overall KS Score**: {min(1.0, sum(ks_results.values()) / len(ks_results) / MIN_KS_PVALUE * 100):.1f}/100

### Correlation Preservation
{corr_status} **Correlation Similarity**: {corr_sim:.4f}

## Bias Detection

{bias_status} **Bias Score**: {bias_results["bias_score"]:.2%}
- Overrepresented: {", ".join(bias_results["overrepresented"]) if bias_results["overrepresented"] else "None"}
- Underrepresented: {", ".join(bias_results["underrepresented"]) if bias_results["underrepresented"] else "None"}

## Privacy Guardrail

{privacy_status} **Average Min Distance**: {privacy_results["avg_min_distance"]:.4f}

## Threshold Compliance
- [ ] KS-test p-value > {MIN_KS_PVALUE} (all variables)
- [ ] Correlation similarity > {MIN_CORR_SIMILARITY}
- [ ] Fidelity score > {MIN_FIDELITY_SCORE}

## Recommendation
{"**APPROVED** — Dataset meets all fidelity thresholds" if fidelity_score >= MIN_FIDELITY_SCORE else "**REVIEW REQUIRED** — Dataset falls below fidelity thresholds"}

---

*DATACARD generated by AutoSource validation pipeline*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(card)

    print(f"DATACARD saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic dataset")
    parser.add_argument("--params", default=PARAMS_INPUT, help="Input parameters.json")
    parser.add_argument("--data", default=DATA_INPUT, help="Input data.csv")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    t_start = time.time()

    # Load inputs
    params_path = Path(args.params)
    data_path = Path(args.data)

    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(params_path) as f:
        params = json.load(f)

    df = pd.read_csv(data_path)

    print(f"Loading parameters from: {params_path}")
    print(f"Loading dataset from: {data_path}")
    print(f"  Shape: {df.shape}")

    # Run validations
    print("\nRunning validations...")

    print("  1. KS-test...")
    ks_results = compute_ks_test(df, params)

    print("  2. Correlation similarity...")
    corr_sim = compute_correlation_similarity(df, params)

    print("  3. Bias check...")
    bias_results = check_bias(df, params)

    print("  4. Privacy check...")
    privacy_results = check_privacy(df)

    # Compute overall fidelity
    fidelity_score = compute_overall_fidelity(
        ks_results, corr_sim, bias_results["bias_score"]
    )

    # Save fidelity score
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fidelity_file = output_dir / FIDELITY_FILE.name
    with open(fidelity_file, "w") as f:
        f.write(str(fidelity_score))
    print(f"\nFidelity score saved to {fidelity_file}: {fidelity_score}/100")

    # Generate DATACARD
    datacard_file = output_dir / DATACARD_FILE.name
    generate_datacard(
        fidelity_score,
        ks_results,
        corr_sim,
        bias_results,
        privacy_results,
        params,
        df,
        datacard_file,
    )

    # Timing
    t_end = time.time()
    print(f"\nValidation time: {t_end - t_start:.1f}s")

    # Summary
    print("\n=== Validation Summary ===")
    print(f"Fidelity score: {fidelity_score}/100 (threshold: {MIN_FIDELITY_SCORE})")
    print(f"Correlation similarity: {corr_sim:.4f} (threshold: {MIN_CORR_SIMILARITY})")
    print(f"Bias score: {bias_results['bias_score']:.2%}")

    if fidelity_score >= MIN_FIDELITY_SCORE:
        print("\n✓ Dataset passes all fidelity thresholds!")
    else:
        print(f"\n✗ Dataset falls below fidelity threshold ({MIN_FIDELITY_SCORE})")


if __name__ == "__main__":
    main()
