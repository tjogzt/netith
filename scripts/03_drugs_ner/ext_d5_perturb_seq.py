"""ext_d5_perturb_seq.py — Perturb-seq-equivalent causal perturbation landscape (Extended Data Fig. 4).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/gdsc/{rna_expr.csv, cell_annot.csv, ensg_symbol_map.csv}; results/gdsc/gdsc_netith_cell_lines.csv; data/gdsc_download/GDSC2_IC50_all.csv; data/CRISPRGeneEffect.csv; data/external/depmap_metadata.csv; results/focused_genes_collectri.txt (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/ext_d5_{perturb_effect_ranking, tf_perturb_mediation, combinatorial_screen, perturb_drug_signature}.csv
Pipeline: drug-ner stage — see repository README
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from pathlib import Path
from itertools import combinations
import warnings
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))
warnings.filterwarnings("ignore")

GDSC_DIR = f"{DATA_ROOT}/gdsc_download"
DATA_DIR = f"{DATA_ROOT}/gdsc"
DEPMAP_CRISPR = f"{DATA_ROOT}/CRISPRGeneEffect.csv"
RESULTS_DIR = Path(f"{ROOT}/results/depmap")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

print("=" * 70)
print("Track 5: Perturb-seq Causal Perturbation Analysis")
print("=" * 70)

# ═══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════
print("\n[1] Loading data...", flush=True)

# GDSC expression: rename CEL columns to cell line names and map the ENSG index to symbols
# GDSC expression
expr_raw = pd.read_csv(f"{DATA_DIR}/rna_expr.csv", index_col=0)
annot = pd.read_csv(f"{DATA_DIR}/cell_annot.csv", index_col=0)
gene_map = pd.read_csv(f"{DATA_DIR}/ensg_symbol_map.csv")
ensg_to_sym = dict(zip(gene_map['ensg'].astype(str), gene_map['symbol']))

cell_line_map = {}
for cel, row in annot.iterrows():
    cl = str(row.get('Characteristics.cell.line.', ''))
    if cl and cl != 'nan' and cl != 'NA':
        cell_line_map[cel] = cl

expr_named = expr_raw.copy()
expr_named.columns = [cell_line_map.get(c, c) for c in expr_raw.columns]
expr_named = expr_named.loc[:, ~expr_named.columns.duplicated()]
expr_named.index = expr_named.index.astype(str)

matched_idx = expr_named.index.isin(ensg_to_sym.keys())
expr_sym = expr_named.loc[matched_idx].copy()
expr_sym.index = [ensg_to_sym[g] for g in expr_sym.index]
expr_sym = expr_sym[~expr_sym.index.duplicated(keep='first')]

# NetITH
netith = pd.read_csv(
    f"{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv",
    index_col=0
)
gdsc_names = set(netith.index)

# Drug IC50
ic50_raw = pd.read_csv(f"{GDSC_DIR}/GDSC2_IC50_all.csv")
ic50_mat = ic50_raw.pivot_table(
    index='CELL_LINE_NAME', columns='DRUG_NAME',
    values='LN_IC50', aggfunc='mean'
)

# DepMap CRISPR
print("  Loading CRISPR data...", flush=True)
# Load DepMap CRISPR gene-effect scores; negative score = stronger dependency (essential)
crispr_raw = pd.read_csv(DEPMAP_CRISPR, index_col=0)
crispr_raw.columns = [c.split(' (')[0] for c in crispr_raw.columns]
crispr_raw.index = crispr_raw.index.str.strip()

# Map DepMap → GDSC (reusing proven mapping logic)
meta = pd.read_csv(f"{DATA_ROOT}/external/depmap_metadata.csv")
ach_to_stripped = {}
for _, row in meta.iterrows():
    aid = str(row.get('depmap_id', ''))
    sname = str(row.get('stripped_cell_line_name', ''))
    if aid and sname and sname != 'nan':
        ach_to_stripped[aid] = sname.replace(' ', '').replace('-', '').replace('.', '').upper()

gdsc_stripped_to_orig = {}
for n in gdsc_names:
    gdsc_stripped_to_orig[n.replace(' ', '').replace('-', '').replace('.', '').upper()] = n

mapper = {}
for dep_idx in crispr_raw.index:
    dep_stripped = dep_idx.strip().replace(' ', '').replace('-', '').upper()
    if dep_idx in ach_to_stripped:
        stripped_name = ach_to_stripped[dep_idx]
        if stripped_name in gdsc_stripped_to_orig:
            mapper[dep_idx] = gdsc_stripped_to_orig[stripped_name]
            continue
    if dep_stripped in gdsc_stripped_to_orig:
        mapper[dep_idx] = gdsc_stripped_to_orig[dep_stripped]

common_names = sorted(set(mapper.values()) & gdsc_names)
crispr_rev = {v: k for k, v in mapper.items() if v in common_names}
crispr = crispr_raw.loc[[crispr_rev[n] for n in common_names]]
crispr.index = [mapper.get(c, c) for c in crispr.index]

common_all = sorted(set(common_names) & set(ic50_mat.index))
netith_vals = netith.loc[common_all, 'NetITH'].values
print(f"  Common cell lines (CRISPR+NetITH+IC50): {len(common_all)}", flush=True)

# Get TF list from CollecTRI
try:
    with open(f"{ROOT}/results/focused_genes_collectri.txt") as f:
        focused_genes = set(line.strip() for line in f if line.strip())
except:
    focused_genes = set()

# ═══════════════════════════════════════════════════════════
# 2. GENOME-WIDE PERTURBATION EFFECT RANKING
# ═══════════════════════════════════════════════════════════
print(f"\n[2] Genome-wide perturbation effect ranking...", flush=True)
print(f"    Testing {len(crispr.columns)} genes on {len(common_all)} cell lines...", flush=True)

# Align the CRISPR matrix to the common cell-line order for fast per-gene analysis
# Efficient batch computation
crispr_aligned = pd.DataFrame(index=common_all)
for gene in crispr.columns:
    try:
        vals = crispr.loc[common_all, gene].values
        crispr_aligned[gene] = vals
    except:
        pass

perturb_results = []
for gene in crispr_aligned.columns:
    gene_vals = crispr_aligned[gene].values
    valid = ~(np.isnan(gene_vals) | np.isnan(netith_vals))
    if valid.sum() < 100:
        continue
    
    gv = gene_vals[valid]
    nv = netith_vals[valid]
    
    # Perturbation proxy: split cells into bottom/top 25% CRISPR dependency (essential vs non-essential)
    # Split: bottom 25% (most dependent/essential) vs top 25% (least dependent)
    n_total = len(gv)
    n_low = max(20, n_total // 4)
    sorted_idx = np.argsort(gv)  # ascending: most negative = most dependent
    low_dep = sorted_idx[:n_low]
    high_dep = sorted_idx[-n_low:]
    
    netith_low = nv[low_dep]
    netith_high = nv[high_dep]
    
    delta = np.mean(netith_low) - np.mean(netith_high)
    cohens_d = delta / (np.sqrt((np.var(netith_low) + np.var(netith_high)) / 2) + 1e-10)
    
    # Mann–Whitney U test (null: NetITH distributions equal in dependent vs non-dependent groups)
    try:
        u_stat, p_val = mannwhitneyu(netith_low, netith_high, alternative='two-sided')
    except:
        p_val = 1.0
    
    rho, p_rho = spearmanr(gv, nv)
    
    perturb_results.append({
        'gene': gene,
        'n_total': int(valid.sum()),
        'delta_netith': delta,
        'cohens_d': cohens_d,
        'p_mwu': p_val,
        'rho_crispr_netith': rho,
        'p_spearman': p_rho,
        'is_tf': gene in focused_genes
    })

perturb_df = pd.DataFrame(perturb_results)
perturb_df = perturb_df.sort_values('cohens_d', key=abs, ascending=False)
# Benjamini–Hochberg FDR control across all tested genes
perturb_df['q_fdr'] = stats.false_discovery_control(perturb_df['p_mwu'].values)

print(f"  Genes tested: {len(perturb_df)}", flush=True)
print(f"  Significant (FDR<0.05): {(perturb_df['q_fdr'] < 0.05).sum()}", flush=True)
print(f"  TFs tested: {perturb_df['is_tf'].sum()}", flush=True)

# Top perturbing genes
print(f"\n  Top 20 perturbation effect genes:")
for _, row in perturb_df.head(20).iterrows():
    tf_tag = " [TF]" if row['is_tf'] else ""
    print(f"    {row['gene']:<25} ΔNetITH={row['delta_netith']:+.4f}  "
          f"d={row['cohens_d']:+.3f}  p={row['p_mwu']:.2e}  FDR={row['q_fdr']:.4f}{tf_tag}")

# Write the genome-wide perturbation-effect ranking (ΔNetITH, Cohen's d, MWU p, FDR)
perturb_df.to_csv(RESULTS_DIR / "ext_d5_perturb_effect_ranking.csv", index=False)

# ═══════════════════════════════════════════════════════════
# 3. TF PERTURBATION → NetITH MEDIATION CHAIN
# ═══════════════════════════════════════════════════════════
print(f"\n[3] TF Perturbation → NetITH Mediation Chain...", flush=True)

# For top TFs: CRISPR_dependency → TF_expression → NetITH
tf_perturb = perturb_df[perturb_df['is_tf']].head(50)
tf_mediation_results = []

for _, row in tf_perturb.iterrows():
    gene = row['gene']
    if gene not in crispr_aligned.columns:
        continue
    if gene not in expr_sym.index:
        continue
    
    # Align CRISPR, TF expression, and NetITH
    gene_crispr = crispr_aligned[gene].reindex(common_all).values
    gene_expr = expr_sym.loc[gene, [c for c in common_all if c in expr_sym.columns]].values
    
    valid = ~(np.isnan(gene_crispr) | np.isnan(gene_expr) | np.isnan(netith_vals))
    if valid.sum() < 80:
        continue
    
    z = gene_crispr[valid]  # CRISPR (instrument)
    m = gene_expr[valid]    # TF expression (mediator)
    y = netith_vals[valid]  # NetITH (outcome)
    
    # Standardize
    zs = (z - np.mean(z)) / np.std(z)
    ms = (m - np.mean(m)) / np.std(m)
    ys = (y - np.mean(y)) / np.std(y)
    
    # Product-of-coefficients mediation chain: a = CRISPR→TF expr, c = total CRISPR→NetITH
    # Path coefficients
    a, _, _, _, _ = stats.linregress(zs, ms)  # CRISPR → TF expr
    c, _, _, _, _ = stats.linregress(zs, ys)  # CRISPR → NetITH (total)
    b, _, _, _, _ = stats.linregress(ms, ys)  # TF expr → NetITH (crude)
    
    # Mediation: CRISPR → TF expr → NetITH
    Xmat = np.column_stack([np.ones(len(zs)), zs, ms])
    beta = np.linalg.lstsq(Xmat, ys, rcond=None)[0]
    c_prime, b_path = beta[1], beta[2]
    indirect = a * b_path
    
    # Bootstrap (2,000 resamples) percentile CI for the indirect effect a*b'
    # Bootstrap CI for indirect effect
    n_boot = 2000
    boot_indirect = []
    for _ in range(n_boot):
        idx = np.random.choice(len(zs), len(zs), replace=True)
        a_b = np.cov(zs[idx], ms[idx])[0, 1] / (np.var(zs[idx]) + 1e-10)
        Xb = np.column_stack([np.ones(len(idx)), zs[idx], ms[idx]])
        beta_b = np.linalg.lstsq(Xb, ys[idx], rcond=None)[0]
        boot_indirect.append(a_b * beta_b[2])
    boot_indirect = np.array(boot_indirect)
    ci_95 = np.percentile(boot_indirect, [2.5, 97.5])
    prop_med = indirect / c if abs(c) > 1e-10 else 0
    
    tf_mediation_results.append({
        'tf': gene,
        'n': int(valid.sum()),
        'path_a_crispr_to_expr': a,
        'path_b_expr_to_netith': b_path,
        'total_c': c,
        'direct_c_prime': c_prime,
        'indirect_ab': indirect,
        'prop_mediated': prop_med,
        'ci95_low': ci_95[0],
        'ci95_high': ci_95[1]
    })

# Save TF perturbation→NetITH mediation chains, ranked by |indirect effect|
tf_med_df = pd.DataFrame(tf_mediation_results).sort_values('indirect_ab', key=abs, ascending=False)
tf_med_df.to_csv(RESULTS_DIR / "ext_d5_tf_perturb_mediation.csv", index=False)

print(f"  TF mediation chains tested: {len(tf_med_df)}")
print(f"\n  Top 10 TF → NetITH perturbation chains:")
for _, row in tf_med_df.head(10).iterrows():
    print(f"    {row['tf']:<20} CRISPR→TF(a)={row['path_a_crispr_to_expr']:+.3f}  "
          f"TF→NetITH(b)={row['path_b_expr_to_netith']:+.3f}  "
          f"indirect={row['indirect_ab']:+.4f}  %med={row['prop_mediated']:.3f}")

# ═══════════════════════════════════════════════════════════
# 4. COMBINATORIAL PERTURBATION SCREEN
# ═══════════════════════════════════════════════════════════
print(f"\n[4] Combinatorial TF Perturbation Screen...", flush=True)

# Test pairs of top TFs for synergistic perturbation effects
top_tf_genes = perturb_df[perturb_df['is_tf']].head(30)['gene'].tolist()
available_tfs = [g for g in top_tf_genes if g in crispr_aligned.columns]
print(f"  Screening {len(available_tfs)} TFs ({len(list(combinations(available_tfs, 2)))} pairs)...")

combo_results = []
for tf_a, tf_b in combinations(available_tfs[:20], 2):  # limit to 20 TFs = 190 pairs
    crispr_a = crispr_aligned[tf_a].reindex(common_all).values
    crispr_b = crispr_aligned[tf_b].reindex(common_all).values
    
    valid = ~(np.isnan(crispr_a) | np.isnan(crispr_b) | np.isnan(netith_vals))
    if valid.sum() < 100:
        continue
    
    a_vals = crispr_a[valid]
    b_vals = crispr_b[valid]
    n_vals = netith_vals[valid]
    
    # Synergy test: NetITH ~ A + B + A×B interaction; permutation null shuffles B
    # Quantify synergistic effect: NetITH ~ CRISPR_A + CRISPR_B + CRISPR_A*CRISPR_B
    a_s = (a_vals - np.mean(a_vals)) / np.std(a_vals)
    b_s = (b_vals - np.mean(b_vals)) / np.std(b_vals)
    n_s = (n_vals - np.mean(n_vals)) / np.std(n_vals)
    
    interaction = a_s * b_s
    Xmat = np.column_stack([np.ones(len(a_s)), a_s, b_s, interaction])
    beta = np.linalg.lstsq(Xmat, n_s, rcond=None)[0]
    
    # Interaction significance via permutation
    n_perm = 1000
    perm_interactions = []
    for _ in range(n_perm):
        b_perm = np.random.permutation(b_s)
        X_perm = np.column_stack([np.ones(len(a_s)), a_s, b_perm, a_s * b_perm])
        beta_perm = np.linalg.lstsq(X_perm, n_s, rcond=None)[0]
        perm_interactions.append(beta_perm[3])
    perm_interactions = np.array(perm_interactions)
    p_interaction = (np.sum(np.abs(perm_interactions) >= np.abs(beta[3])) + 1) / (n_perm + 1)
    
    combo_results.append({
        'tf_a': tf_a, 'tf_b': tf_b,
        'n': int(valid.sum()),
        'beta_a': beta[1], 'beta_b': beta[2],
        'beta_interaction': beta[3],
        'p_interaction': p_interaction
    })

# Save the TF-pair combinatorial screen, ranked by interaction p-value
combo_df = pd.DataFrame(combo_results).sort_values('p_interaction')
combo_df.to_csv(RESULTS_DIR / "ext_d5_combinatorial_screen.csv", index=False)

sig_combos = combo_df[combo_df['p_interaction'] < 0.05]
print(f"  Significant interactions (p<0.05): {len(sig_combos)}/{len(combo_df)}")
if len(sig_combos) > 0:
    print(f"\n  Top 10 synergistic TF pairs:")
    for _, row in sig_combos.head(10).iterrows():
        print(f"    {row['tf_a']:<20} × {row['tf_b']:<20}  "
              f"β_int={row['beta_interaction']:+.4f}  p={row['p_interaction']:.4f}")

# ═══════════════════════════════════════════════════════════
# 5. PERTURB-SEQ INFORMED DRUG SENSITIVITY
# ═══════════════════════════════════════════════════════════
print(f"\n[5] Perturb-seq Informed Drug Sensitivity...", flush=True)

# Build a "Perturb-seq signature": top genes whose perturbation changes NetITH
top_perturb_genes = perturb_df.head(100)['gene'].tolist()
perturb_genes_in_crispr = [g for g in top_perturb_genes if g in crispr_aligned.columns]

# For each cell line, compute a Perturb-NetITH score:
# Weighted sum of CRISPR dependencies for top perturbing genes
# Perturb-seq signature: weight each cell line's CRISPR scores by the gene's NetITH perturbation effect (Cohen's d)
perturb_weights = {}
for _, row in perturb_df.head(100).iterrows():
    if row['gene'] in crispr_aligned.columns:
        perturb_weights[row['gene']] = row['cohens_d']

perturb_netith_score = np.zeros(len(common_all))
for gene, w in perturb_weights.items():
    gv = crispr_aligned[gene].reindex(common_all).values
    gv_clean = np.nan_to_num(gv, nan=0)
    perturb_netith_score += w * gv_clean

perturb_netith_score = (perturb_netith_score - np.mean(perturb_netith_score)) / np.std(perturb_netith_score)

# Correlate Perturb-NetITH score with actual NetITH
# Spearman test (null: no association) of the Perturb-NetITH signature against actual NetITH
rho_perturb, p_perturb = spearmanr(perturb_netith_score, netith_vals)
print(f"  Perturb-NetITH vs actual NetITH: ρ={rho_perturb:.4f} (p={p_perturb:.2e})")

# Drug sensitivity correlation
drug_perturb_corrs = []
for drug in sorted(ic50_mat.columns):
    ic50_col = ic50_mat.loc[common_all, drug]
    ic50_v = ic50_col.values.astype(float)
    valid = ~(np.isnan(ic50_v) | np.isnan(perturb_netith_score))
    if valid.sum() < 50:
        continue
    rho, p = spearmanr(perturb_netith_score[valid], ic50_v[valid])
    drug_perturb_corrs.append({
        'drug': drug,
        'rho_perturb_ic50': rho,
        'p': p,
        'n': int(valid.sum())
    })

drug_perturb_df = pd.DataFrame(drug_perturb_corrs).sort_values('rho_perturb_ic50', key=abs, ascending=False)
# Write drug-level Spearman correlations of the Perturb-NetITH signature vs IC50
drug_perturb_df.to_csv(RESULTS_DIR / "ext_d5_perturb_drug_signature.csv", index=False)

print(f"  Drugs tested: {len(drug_perturb_df)}")
print(f"\n  Top 15 drugs by Perturb-NetITH association:")
for _, row in drug_perturb_df.head(15).iterrows():
    sig = "*" if row['p'] < 0.05 else ""
    print(f"    {row['drug']:<35} ρ={row['rho_perturb_ic50']:+.4f}  "
          f"p={row['p']:.4f}  n={row['n']} {sig}")

# Compare Perturb-NetITH vs actual NetITH for drug prediction
# For each drug, compare ρ(NetITH, IC50) vs ρ(Perturb-NetITH, IC50)
drug_compare = []
for drug in sorted(ic50_mat.columns):
    ic50_col = ic50_mat.loc[common_all, drug]
    ic50_v = ic50_col.values.astype(float)
    
    valid_n = ~(np.isnan(ic50_v) | np.isnan(netith_vals))
    valid_p = ~(np.isnan(ic50_v) | np.isnan(perturb_netith_score))
    
    if valid_n.sum() < 50 or valid_p.sum() < 50:
        continue
    
    rho_netith, _ = spearmanr(netith_vals[valid_n], ic50_v[valid_n])
    rho_perturb, _ = spearmanr(perturb_netith_score[valid_p], ic50_v[valid_p])
    
    drug_compare.append({
        'drug': drug,
        'rho_netith': rho_netith,
        'rho_perturb': rho_perturb,
        'delta': rho_perturb - rho_netith
    })

drug_comp_df = pd.DataFrame(drug_compare)
print(f"\n  NetITH vs Perturb-NetITH drug prediction comparison:")
print(f"    Mean ρ(NetITH, IC50)        = {drug_comp_df['rho_netith'].mean():.4f}")
print(f"    Mean ρ(Perturb-NetITH, IC50) = {drug_comp_df['rho_perturb'].mean():.4f}")
print(f"    Mean Δ = {drug_comp_df['delta'].mean():+.4f}")
print(f"    Perturb-NetITH better in {(drug_comp_df['delta'] > 0).sum()}/{len(drug_comp_df)} drugs")

# ═══════════════════════════════════════════════════════════
# 6. LITERATURE CROSS-REFERENCE
# ═══════════════════════════════════════════════════════════
print(f"\n[6] Cross-reference with published Perturb-seq studies...", flush=True)

# Known Perturb-seq essential TFs from literature
# Published Perturb-seq TF lists (Replogle 2022, Dixit 2016, Adamson 2016, Norman 2019) for benchmarking
KNOWN_PERTURB_TFS = {
    'Replogle_2022_K562': ['GATA1', 'KLF1', 'SPI1', 'CEBPB', 'RUNX1', 'MYC', 'TP53'],
    'Dixit_2016_K562': ['GATA1', 'CEBPA', 'SPI1', 'RUNX1', 'GFI1B', 'NFE2'],
    'Adamson_2016_U2OS': ['TP53', 'CDKN1A', 'RB1', 'E2F1', 'MYC'],
    'Norman_2019_K562': ['BCL11A', 'NFE2', 'GATA1', 'KLF1', 'ZBTB7A']
}

print(f"  Checking overlap with known Perturb-seq TFs...")
all_known = set()
for study, tfs in KNOWN_PERTURB_TFS.items():
    overlap = [t for t in tfs if t in perturb_df['gene'].values]
    all_known.update(tfs)
    if overlap:
        overlap_df = perturb_df[perturb_df['gene'].isin(overlap)]
        mean_rank = overlap_df.index.min() if len(overlap_df) > 0 else np.nan
        print(f"    {study:<25}: {len(overlap)}/{len(tfs)} TFs found  "
              f"(best rank: {mean_rank})")
    else:
        print(f"    {study:<25}: 0/{len(tfs)} TFs found")

# Rank of known Perturb-seq TFs in our analysis
known_ranks = []
for tf in all_known:
    if tf in perturb_df['gene'].values:
        rank = perturb_df[perturb_df['gene'] == tf].index[0]
        known_ranks.append({'tf': tf, 'rank': rank, 'delta': perturb_df.loc[rank, 'delta_netith']})

if known_ranks:
    known_ranks_df = pd.DataFrame(known_ranks).sort_values('rank')
    print(f"\n  Known Perturb-seq TF ranks in our analysis:")
    for _, row in known_ranks_df.iterrows():
        print(f"    {row['tf']:<20} rank={row['rank']+1}  ΔNetITH={row['delta']:+.4f}")

# ═══════════════════════════════════════════════════════════
# 7. SUMMARY
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY — Direction 5: Perturb-seq Analysis")
print(f"{'='*70}")

n_sig_fdr = (perturb_df['q_fdr'] < 0.05).sum()
n_tf_sig = ((perturb_df['q_fdr'] < 0.05) & perturb_df['is_tf']).sum()
print(f"\n  A. Perturbation Effect Ranking:")
print(f"    Total genes tested: {len(perturb_df)}")
print(f"    Significant (FDR<0.05): {n_sig_fdr}")
print(f"    Significant TFs: {n_tf_sig}")

print(f"\n  B. TF Perturbation → NetITH Mediation:")
print(f"    TF mediation chains: {len(tf_med_df)}")
if len(tf_med_df) > 0:
    top_med = tf_med_df.iloc[0]
    print(f"    Top mediator: {top_med['tf']} "
          f"(indirect={top_med['indirect_ab']:+.4f}, "
          f"%med={top_med['prop_mediated']:.3f})")

print(f"\n  C. Combinatorial Perturbations:")
print(f"    TF pairs screened: {len(combo_df)}")
print(f"    Significant interactions: {len(sig_combos)}")

print(f"\n  D. Perturb-seq Drug Signature:")
print(f"    Perturb-NetITH vs actual NetITH: ρ={rho_perturb:.4f}")
print(f"    Drugs with Perturb-netITH association: {len(drug_perturb_df)}")

print(f"\n  E. Literature Overlap:")
print(f"    Known Perturb-seq TFs found: {len(known_ranks)}/{len(all_known)}")

print(f"\nDone. All results in {RESULTS_DIR}/")
