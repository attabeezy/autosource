"""
AutoSource Synthesis Pipeline
parameters.json → data.csv (agent-modifiable)

Usage: uv run synthesize.py
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Any

# ---------------------------------------------------------------------------
# Hyperparameters (agent edits this section - follow AutoResearch pattern)
# ---------------------------------------------------------------------------

# Generation settings
N_ROWS = 10_000  # number of synthetic rows
SEED = 42  # random seed
NOISE_LEVEL = 0.1  # noise injection (0.0 = exact, 1.0 = max noise)

# Distribution choices (agent modifies these)
DEFAULT_DISTRIBUTION = "normal"  # Options: normal, uniform, gamma, beta, lognorm
CORRELATION_METHOD = "gaussian"  # Options: gaussian, t (for heavy tails)

# Time budget (agent must finish within this)
TIME_BUDGET = 600  # 10 minutes in seconds

# Output
OUTPUT_CSV = Path("results/data.csv")
PARAMS_INPUT = Path("results/parameters.json")


# ---------------------------------------------------------------------------
# Synthesis Functions
# ---------------------------------------------------------------------------


def generate_variable(
    params: dict[str, Any], size: int, noise_level: float
) -> np.ndarray:
    """Generate single variable based on distribution parameters."""
    dist_type = params.get("distribution", DEFAULT_DISTRIBUTION)
    mean = params.get("mean", 0)
    std = params.get("std", 1)

    # Add noise (agent can adjust noise_level)
    noise_std = std * noise_level

    if dist_type == "normal":
        data = np.random.normal(mean, std * (1 + noise_std), size)
    elif dist_type == "uniform":
        min_val = params.get("min", mean - 2 * std)
        max_val = params.get("max", mean + 2 * std)
        data = np.random.uniform(min_val, max_val, size)
    elif dist_type == "gamma":
        # Convert mean/std to gamma shape/scale
        shape = (mean / std) ** 2
        scale = std**2 / mean
        data = np.random.gamma(shape, scale, size)
    elif dist_type == "beta":
        # Convert mean/std to beta a/b (simplified)
        mean_norm = (mean - params.get("min", 0)) / (
            params.get("max", 1) - params.get("min", 0)
        )
        a = mean_norm * 10 + 0.5
        b = (1 - mean_norm) * 10 + 0.5
        data = np.random.beta(a, b, size)
        # Scale to original range
        data = data * (params.get("max", 1) - params.get("min", 0)) + params.get(
            "min", 0
        )
    else:
        data = np.random.normal(mean, std, size)

    return data


def generate_correlated_dataset(
    stats: dict[str, Any], n_rows: int, noise_level: float
) -> pd.DataFrame:
    """Generate dataset with correlations using Gaussian Copula."""
    variables = stats.get("variables", {})
    correlations = stats.get("correlations", {})

    var_names = list(variables.keys())
    n_vars = len(var_names)

    if n_vars == 0:
        raise ValueError("No variables found in parameters.json")

    # Step 1: Generate base normal variables
    np.random.seed(SEED)
    base_data = np.random.randn(n_rows, n_vars)

    # Step 2: Apply correlation structure (Gaussian Copula)
    if correlations:
        # Build correlation matrix
        corr_matrix = np.eye(n_vars)
        for i, v1 in enumerate(var_names):
            for j, v2 in enumerate(var_names):
                if i >= j:
                    continue
                key = (
                    f"{v1} ↔ {v2}" if f"{v1} ↔ {v2}" in correlations else f"{v2} ↔ {v1}"
                )
                if key in correlations:
                    corr = correlations[key].get("correlation", 0)
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        # Cholesky decomposition for correlation
        try:
            L = np.linalg.cholesky(corr_matrix)
            base_data = base_data @ L.T
        except np.linalg.LinAlgError:
            print(
                "  Warning: Correlation matrix not positive definite, using independence"
            )

    # Step 3: Transform to marginal distributions
    data_dict = {}
    for i, var_name in enumerate(var_names):
        params = variables[var_name]
        data_dict[var_name] = generate_variable(params, n_rows, noise_level)

        # Apply correlation transform (rank substitution)
        if correlations:
            ranks = stats.rankdata(base_data[:, i])
            data_dict[var_name] = np.take(
                np.sort(data_dict[var_name]), np.argsort(ranks).argsort()
            )

    return pd.DataFrame(data_dict)


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save dataset to CSV and print summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset from parameters"
    )
    parser.add_argument("--input", default=PARAMS_INPUT, help="Input parameters.json")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output data.csv")
    parser.add_argument(
        "--n-rows", type=int, default=N_ROWS, help="Number of rows to generate"
    )
    parser.add_argument(
        "--noise", type=float, default=NOISE_LEVEL, help="Noise level (0-1)"
    )
    args = parser.parse_args()

    t_start = time.time()

    # Load parameters
    params_path = Path(args.input)
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    with open(params_path) as f:
        stats = json.load(f)

    print(f"Loading parameters from: {params_path}")
    print(f"  Found {len(stats.get('variables', {}))} variables")
    print(f"  Found {len(stats.get('correlations', {}))} correlations")
    print(f"  Target rows: {args.n_rows}")

    # Generate dataset
    print("\nSynthesizing data...")
    df = generate_correlated_dataset(stats, args.n_rows, args.noise)

    # Save
    save_dataset(df, Path(args.output))

    # Summary stats
    t_end = time.time()
    print(f"\nGeneration time: {t_end - t_start:.1f}s")
    print("\nDataset Summary Statistics:")
    print(df.describe())

    # Timing output for agent loop
    print(f"\n---")
    print(f"generation_seconds: {t_end - t_start:.1f}")
    print(f"n_rows: {len(df)}")
    print(f"n_columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
