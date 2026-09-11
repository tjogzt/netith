"""
compare_ith.py — Compare NetITH against traditional ITH metrics.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : NetITH results CSV; traditional ITH metrics (MAF, expression, clinical)
Outputs : correlation tables; incremental-value (likelihood-ratio) statistics
Module  : src.benchmarking
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


def load_netith_results(path: str) -> pd.DataFrame:
    """Load NetITH results from CSV.

    Expected columns: sample_id, cancer_type, netith, mean_entropy,
                     var_entropy, skewness, n_cells_used.
    """
    return pd.read_csv(path)


def load_traditional_ith(
    tcga_maf_path: Optional[str] = None,
    tcga_expr_path: Optional[str] = None,
    clinical_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load traditional ITH metrics.

    This function should be customized based on available data.
    Placeholder implementation.

    Args:
        tcga_maf_path: Path to TCGA mutation MAF file.
        tcga_expr_path: Path to TCGA expression matrix.
        clinical_path: Path to TCGA clinical data.

    Returns:
        DataFrame with columns: sample_id, MATH, shannon, depth, estimate.
    """
    # This is a scaffold - actual implementation depends on data availability
    # For now, returns empty DataFrame with expected columns
    return pd.DataFrame(
        columns=["sample_id", "MATH", "shannon", "depth", "estimate"]
    )


def compare_metrics(
    netith_df: pd.DataFrame,
    trad_ith_df: pd.DataFrame,
    merge_on: str = "sample_id",
) -> pd.DataFrame:
    """Merge and compare NetITH with traditional ITH metrics.

    Returns:
        Correlation matrix DataFrame.
    """
    # Inner join on sample ID keeps only samples present in both tables
    merged = netith_df.merge(trad_ith_df, on=merge_on, how="inner")
    return merged


def compute_correlations(
    merged_df: pd.DataFrame,
    netith_col: str = "netith",
    trad_cols: list = ["MATH", "shannon", "depth", "estimate"],
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute correlations between NetITH and traditional metrics.

    Args:
        merged_df: DataFrame with both NetITH and traditional metrics.
        netith_col: Column name for NetITH.
        trad_cols: Column names for traditional metrics.
        method: 'pearson' or 'spearman'.

    Returns:
        DataFrame with correlation coefficients and p-values.
    """
    corr_func = spearmanr if method == "spearman" else pearsonr
    results = []
    for col in trad_cols:
        if col not in merged_df.columns:
            continue
        # Drop rows with missing values before correlating
        valid = merged_df[[netith_col, col]].dropna()
        if len(valid) < 10:
            continue
        r, p = corr_func(valid[netith_col], valid[col])
        results.append(
            {
                "metric": col,
                "correlation": r,
                "p_value": p,
                "n_samples": len(valid),
                "method": method,
            }
        )
    return pd.DataFrame(results)


def incremental_value_test(
    merged_df: pd.DataFrame,
    outcome_col: str,
    netith_col: str = "netith",
    trad_cols: list = ["MATH", "depth"],
    clinical_covariates: Optional[list] = None,
) -> Dict:
    """Test incremental predictive value of NetITH beyond traditional metrics.

    Uses hierarchical regression approach.

    Args:
        merged_df: Data with all variables.
        outcome_col: Column name for outcome (e.g., survival time).
        netith_col: Column name for NetITH.
        trad_cols: Traditional ITH metric columns.
        clinical_covariates: Additional clinical covariates (age, stage, etc.).

    Returns:
        Dict with R² values and likelihood ratio test results.
    """
    import statsmodels.api as sm

    # Prepare data
    cols_to_use = [netith_col] + trad_cols
    if clinical_covariates:
        cols_to_use += clinical_covariates
    cols_to_use.append(outcome_col)

    data = merged_df[cols_to_use].dropna()

    # Base model: traditional metrics + clinical covariates only
    X_base = data[trad_cols + (clinical_covariates or [])]
    X_base = sm.add_constant(X_base)

    # Full model: base + NetITH
    X_full = data[[netith_col] + trad_cols + (clinical_covariates or [])]
    X_full = sm.add_constant(X_full)

    y = data[outcome_col]

    model_base = sm.OLS(y, X_base).fit()
    model_full = sm.OLS(y, X_full).fit()

    # Likelihood ratio test
    from scipy.stats import chi2

    lr_stat = -2 * (model_base.llf - model_full.llf)
    lr_pvalue = chi2.sf(lr_stat, df=1)  # 1 additional parameter (netith)

    return {
        "R2_base": model_base.rsquared,
        "R2_full": model_full.rsquared,
        "delta_R2": model_full.rsquared - model_base.rsquared,
        "LR_stat": lr_stat,
        "LR_pvalue": lr_pvalue,
    }
