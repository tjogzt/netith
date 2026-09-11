"""
run_netith_threshold.py — Find nonlinear NetITH drug-resistance thresholds via piecewise regression and GAM.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  :
    - results/gdsc/gdsc_netith_cell_lines.csv: precomputed NetITH per cell line
    - results/gdsc/gdsc_drug_netith_correlations.csv: NetITH x IC50 correlations
    - <DATA_ROOT>/gdsc_download/GDSC2_IC50_all.csv: GDSC2 drug sensitivity (LN_IC50)
Outputs :
    - results/depmap/netith_drug_thresholds.csv, netith_drug_nonlinearity.csv
    - results/depmap/figures/netith_nonlinear_threshold.png
Pipeline: stage 2 — see repository README for the full pipeline order
"""


import numpy as np
import pandas as pd
import os, sys, warnings
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

GDSC_OUTPUT = Path(f"{ROOT}/results/gdsc")
OUTPUT_DIR = Path(f"{ROOT}/results/depmap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

SEED = 42
N_TOP_DRUGS = 30   # number of top drugs (by |rho|) tested for a resistance threshold
N_GAM_DRUGS = 20   # number of top drugs smoothed with the spline (GAM) nonlinearity check
MIN_CELLS = 50     # minimum cell lines per drug to fit the piecewise regression
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════
# 1. Load data
# ═══════════════════════════════════════════════════════════
print("[1/4] Loading GDSC NetITH and drug sensitivity data...")

# Load NetITH
netith_df = pd.read_csv(GDSC_OUTPUT / "gdsc_netith_cell_lines.csv", index_col=0)

# Load IC50
GDSC_DIR_IC50 = f"{DATA_ROOT}/gdsc_download"
ic50_raw = pd.read_csv(f"{GDSC_DIR_IC50}/GDSC2_IC50_all.csv")
ic50_mat = ic50_raw.pivot_table(
    index='CELL_LINE_NAME', columns='DRUG_NAME',
    values='LN_IC50', aggfunc='mean'
)

# Load drug correlations (from §3.6)
drug_corr = pd.read_csv(GDSC_OUTPUT / "gdsc_drug_netith_correlations.csv")

# Align
common_cells = sorted(set(netith_df.index) & set(ic50_mat.index))
print(f"  Common cell lines: {len(common_cells)}")

netith_aligned = netith_df.loc[common_cells, 'NetITH']
ic50_aligned = ic50_mat.loc[common_cells]

# ═══════════════════════════════════════════════════════════
# 2. Piecewise regression per drug — find threshold
# ═══════════════════════════════════════════════════════════
print("\n[2/4] Piecewise regression: finding NetITH resistance thresholds...")

from sklearn.linear_model import LinearRegression

# Focus on top drugs (strongest NetITH association)
drug_corr['abs_rho'] = drug_corr['rho'].abs()
top_drugs = drug_corr.nlargest(N_TOP_DRUGS, 'abs_rho')['drug'].values
print(f"  Testing {len(top_drugs)} top drugs")

threshold_results = []

for drug in top_drugs:
    if drug not in ic50_aligned.columns:
        continue
    
    ic50_vals = ic50_aligned[drug].dropna()
    common_idx = ic50_vals.index.intersection(common_cells)
    if len(common_idx) < MIN_CELLS:
        continue
    
    x = netith_aligned.loc[common_idx].values
    y = ic50_vals.loc[common_idx].values
    
    # Sort by NetITH
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    n = len(x_sorted)
    
    # Test all possible breakpoints (between 20% and 80% of data)
    best_r2 = -np.inf
    best_bp = None
    best_models = None
    
    for bp_percentile in range(20, 81, 2):
        bp_idx = int(n * bp_percentile / 100)
        bp_value = x_sorted[bp_idx]
        
        # Segment 1: x <= bp
        mask1 = x_sorted <= bp_value
        mask2 = x_sorted > bp_value
        
        if mask1.sum() < 20 or mask2.sum() < 20:
            continue
        
        # Fit two lines
        x1 = x_sorted[mask1].reshape(-1, 1)
        y1 = y_sorted[mask1]
        x2 = x_sorted[mask2].reshape(-1, 1)
        y2 = y_sorted[mask2]
        
        lr1 = LinearRegression().fit(x1, y1)
        lr2 = LinearRegression().fit(x2, y2)
        
        # Combined R²
        pred1 = lr1.predict(x1)
        pred2 = lr2.predict(x2)
        y_pred = np.concatenate([pred1, pred2])
        y_all = np.concatenate([y1, y2])
        
        ss_res = np.sum((y_all - y_pred) ** 2)
        ss_tot = np.sum((y_all - y_all.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        if r2 > best_r2:
            best_r2 = r2
            best_bp = bp_value
            best_models = (lr1, lr2)
    
    if best_bp is not None:
        # Single-line R²
        lr_single = LinearRegression().fit(x_sorted.reshape(-1, 1), y_sorted)
        r2_single = lr_single.score(x_sorted.reshape(-1, 1), y_sorted)
        
        # Slope change
        slope1 = best_models[0].coef_[0]
        slope2 = best_models[1].coef_[0]
        slope_delta = slope2 - slope1
        
        threshold_results.append({
            'drug': drug,
            'bp_NetITH': best_bp,
            'r2_single': r2_single,
            'r2_piecewise': best_r2,
            'delta_r2': best_r2 - r2_single,
            'slope_before': slope1,
            'slope_after': slope2,
            'slope_delta': slope_delta,
            'n': n,
        })

thresh_df = pd.DataFrame(threshold_results).sort_values('delta_r2', ascending=False)
print(f"  Drugs with threshold detected: {len(thresh_df)}")
print(f"\n  Top 10 drugs with strongest threshold effects:")
for _, row in thresh_df.head(10).iterrows():
    print(f"    {row['drug']:30s}: BP={row['bp_NetITH']:.2f}, "
          f"R² single={row['r2_single']:.3f}→piecewise={row['r2_piecewise']:.3f}, "
          f"Δ slope={row['slope_delta']:+.3f}")

# ═══════════════════════════════════════════════════════════
# 3. GAM smoothing — visualize nonlinearity
# ═══════════════════════════════════════════════════════════
print("\n[3/4] GAM smoothing for nonlinearity detection...")

# For top drugs, compute smoothed curves
gam_results = []

for drug in top_drugs[:N_GAM_DRUGS]:
    if drug not in ic50_aligned.columns:
        continue
    
    ic50_vals = ic50_aligned[drug].dropna()
    common_idx = ic50_vals.index.intersection(common_cells)
    if len(common_idx) < MIN_CELLS:
        continue
    
    x = netith_aligned.loc[common_idx].values
    y = ic50_vals.loc[common_idx].values
    
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Linear fit
    lr = LinearRegression().fit(x_sorted.reshape(-1, 1), y_sorted)
    y_linear = lr.predict(x_sorted.reshape(-1, 1))
    
    # GAM smoothing (spline)
    try:
        # UnivariateSpline with smoothing
        spline = UnivariateSpline(x_sorted, y_sorted, s=len(x_sorted)*0.1)
        y_smooth = spline(x_sorted)
        
        # Deviation from linear
        deviation = y_smooth - y_linear
        max_dev = np.max(np.abs(deviation))
        
        # Nonlinearity score: variance of deviation / variance of y
        nonlinearity = np.var(deviation) / np.var(y_sorted) if np.var(y_sorted) > 0 else 0
        
        gam_results.append({
            'drug': drug,
            'nonlinearity': nonlinearity,
            'max_deviation': max_dev,
            'n': len(common_idx),
        })
    except Exception:
        pass

gam_df = pd.DataFrame(gam_results).sort_values('nonlinearity', ascending=False)
print(f"  GAM analyzed: {len(gam_df)} drugs")
print(f"\n  Top 10 drugs with strongest nonlinearity:")
for _, row in gam_df.head(10).iterrows():
    print(f"    {row['drug']:30s}: nonlinearity={row['nonlinearity']:.4f}, max_dev={row['max_deviation']:.3f}")

# ═══════════════════════════════════════════════════════════
# 4. Consensus threshold analysis
# ═══════════════════════════════════════════════════════════
print("\n[4/4] Consensus NetITH threshold across drugs...")

if len(thresh_df) > 0:
    # Distribution of breakpoints
    bp_values = thresh_df['bp_NetITH'].values
    print(f"  Breakpoint distribution:")
    print(f"    Mean: {bp_values.mean():.2f}")
    print(f"    Median: {np.median(bp_values):.2f}")
    print(f"    Std: {bp_values.std():.2f}")
    print(f"    Range: [{bp_values.min():.2f}, {bp_values.max():.2f}]")
    
    # How many drugs have breakpoints in [4.0, 5.5]?
    bp_in_range = thresh_df[(thresh_df['bp_NetITH'] >= 4.0) & (thresh_df['bp_NetITH'] <= 5.5)]
    print(f"    BP in [4.0, 5.5]: {len(bp_in_range)}/{len(thresh_df)}")
    
    # Slope direction after breakpoint
    positive_after = (thresh_df['slope_after'] > 0).sum()
    negative_after = (thresh_df['slope_after'] < 0).sum()
    print(f"    Slope after BP: {positive_after} pos, {negative_after} neg")

# ═══════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════
print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(22, 13))
fig.suptitle('NetITH Nonlinear Drug Sensitivity: Threshold Effects & Resistance Breakpoints',
             fontsize=14, fontweight='bold', y=0.99)

# Panel A: Top 4 drugs — piecewise fit
for panel_i, ax_idx in enumerate([(0,0), (0,1), (1,0), (1,1)]):
    ax = axes[ax_idx]
    if panel_i < min(4, len(thresh_df)):
        row = thresh_df.iloc[panel_i]
        drug = row['drug']
        
        ic50_vals = ic50_aligned[drug].dropna()
        common_idx = ic50_vals.index.intersection(common_cells)
        x = netith_aligned.loc[common_idx].values
        y = ic50_vals.loc[common_idx].values
        
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        
        bp = row['bp_NetITH']
        mask1 = x_sorted <= bp
        mask2 = x_sorted > bp
        
        ax.scatter(x, y, c='steelblue', alpha=0.3, s=12, edgecolors='none')
        
        # Piecewise lines
        lr1 = LinearRegression().fit(x_sorted[mask1].reshape(-1,1), y_sorted[mask1])
        lr2 = LinearRegression().fit(x_sorted[mask2].reshape(-1,1), y_sorted[mask2])
        
        x1_line = np.linspace(x_sorted[mask1].min(), bp, 50)
        ax.plot(x1_line, lr1.predict(x1_line.reshape(-1,1)), 'r-', linewidth=2)
        
        x2_line = np.linspace(bp, x_sorted[mask2].max(), 50)
        ax.plot(x2_line, lr2.predict(x2_line.reshape(-1,1)), 'orange', linewidth=2)
        
        ax.axvline(x=bp, color='darkred', linestyle='--', linewidth=1.5,
                   label=f'BP={bp:.2f}')
        ax.set_xlabel('NetITH', fontsize=10)
        ax.set_ylabel('LN_IC50', fontsize=10)
        ax.set_title(f'{drug}\nR²={row["r2_single"]:.3f}→{row["r2_piecewise"]:.3f}', fontsize=9)
        ax.legend(fontsize=7)

# Panel C: Breakpoint distribution histogram
ax = axes[0, 2]
if len(thresh_df) > 0:
    ax.hist(thresh_df['bp_NetITH'], bins=20, color='#d62728', alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax.axvline(x=np.median(thresh_df['bp_NetITH']), color='darkred',
               linestyle='--', linewidth=2, label=f'Median={np.median(thresh_df["bp_NetITH"]):.2f}')
    ax.set_xlabel('NetITH Breakpoint', fontsize=11)
    ax.set_ylabel('Number of Drugs', fontsize=11)
    ax.set_title(f'C: NetITH Resistance Threshold Distribution\n(n={len(thresh_df)} drugs)', fontsize=10)
    ax.legend(fontsize=8)

# Panel D: Slope change vs baseline slope
ax = axes[1, 2]
if len(thresh_df) > 0:
    ax.scatter(thresh_df['slope_before'], thresh_df['slope_delta'],
               c=thresh_df['delta_r2'], cmap='plasma', s=60, alpha=0.8,
               edgecolors='black', linewidth=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Slope Before BP', fontsize=11)
    ax.set_ylabel('Slope Change (After − Before)', fontsize=11)
    ax.set_title('D: Slope Changes at Breakpoint', fontsize=10)
    plt.colorbar(ax.collections[0], ax=ax, label='Δ R²', shrink=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = OUTPUT_DIR / "figures" / "netith_nonlinear_threshold.png"
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved: {fig_path}")

# ═══════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════
thresh_df.to_csv(OUTPUT_DIR / "netith_drug_thresholds.csv", index=False)
gam_df.to_csv(OUTPUT_DIR / "netith_drug_nonlinearity.csv", index=False)

print("\n" + "=" * 70)
print("NetITH NONLINEAR THRESHOLD ANALYSIS — COMPLETE")
print("=" * 70)
if len(thresh_df) > 0:
    print(f"""
  Drugs with threshold: {len(thresh_df)}
  Median breakpoint: {np.median(bp_values):.2f}
  Mean ΔR²: {thresh_df['delta_r2'].mean():.4f}
  
  Top nonlinear drugs: {gam_df.head(3)['drug'].values}
  
Output:
  {OUTPUT_DIR}/netith_drug_thresholds.csv
  {OUTPUT_DIR}/netith_drug_nonlinearity.csv
  {OUTPUT_DIR}/figures/netith_nonlinear_threshold.png
""")
