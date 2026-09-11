"""ext_d6_phospho_netith.py — kinase-informed von Neumann entropy (Phospho-NetITH) of the CollecTRI GRN (Extended Data Fig. 5).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/gdsc/{rna_expr.csv, cell_annot.csv, ensg_symbol_map.csv}; results/gdsc/gdsc_netith_cell_lines.csv; data/gdsc_download/GDSC2_IC50_all.csv; /tmp/collectri_net.pkl; results/focused_genes_collectri.txt (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/ext_d6_{phospho_netith, kinase_pathway_netith, drug_comparison}.csv
Pipeline: drug-ner stage — see repository README
"""
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from scipy.linalg import eigh
from pathlib import Path
import pickle
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
print("Track 6: Phospho-NetITH — Kinase-Informed Network Entropy")
print("=" * 70)

# ═══════════════════════════════════════════════════════
# 0. CORE KINASE-SUBSTRATE NETWORK (Literature-curated)
# ═══════════════════════════════════════════════════════

# Literature-curated kinase→substrate core network (kinase, substrate, pathway label)
KINASE_SUBSTRATE_CORE = [
    # RTK signaling
    ("EGFR", "GRB2", "RTK"), ("EGFR", "SHC1", "RTK"), ("EGFR", "STAT3", "RTK"),
    ("ERBB2", "PIK3CA", "RTK"), ("FGFR1", "FRS2", "RTK"), ("FGFR1", "PLCG1", "RTK"),
    ("IGF1R", "IRS1", "RTK"), ("MET", "GAB1", "RTK"), ("KIT", "GRB2", "RTK"),
    ("PDGFRA", "PIK3CA", "RTK"), ("PDGFRB", "GRB2", "RTK"), ("FLT3", "GRB2", "RTK"),
    # MAPK cascade
    ("MAP3K1", "MAP2K1", "MAPK"), ("RAF1", "MAP2K1", "MAPK"),
    ("BRAF", "MAP2K1", "MAPK"), ("MAP2K1", "MAPK1", "MAPK"),
    ("MAP2K1", "MAPK3", "MAPK"), ("MAP2K2", "MAPK1", "MAPK"),
    ("MAP2K4", "MAPK8", "MAPK"), ("MAP2K4", "MAPK9", "MAPK"),
    ("MAP2K6", "MAPK14", "MAPK"), ("MAP2K7", "MAPK8", "MAPK"),
    # PI3K/AKT/mTOR
    ("PIK3CA", "AKT1", "PI3K"), ("AKT1", "MTOR", "PI3K"),
    ("AKT1", "GSK3B", "PI3K"), ("AKT1", "MDM2", "PI3K"),
    ("AKT1", "FOXO1", "PI3K"), ("AKT1", "FOXO3", "PI3K"),
    ("AKT1", "BAD", "PI3K"), ("MTOR", "EIF4EBP1", "PI3K"),
    ("MTOR", "RPS6KB1", "PI3K"), ("MTOR", "ULK1", "PI3K"),
    ("GSK3B", "CTNNB1", "PI3K"), ("GSK3B", "MYC", "PI3K"),
    ("GSK3B", "CCND1", "PI3K"),
    # JAK/STAT
    ("JAK1", "STAT1", "JAKSTAT"), ("JAK1", "STAT3", "JAKSTAT"),
    ("JAK2", "STAT3", "JAKSTAT"), ("JAK2", "STAT5A", "JAKSTAT"),
    ("JAK2", "STAT5B", "JAKSTAT"), ("TYK2", "STAT1", "JAKSTAT"),
    # Cell cycle / CDK
    ("CDK1", "RB1", "CellCycle"), ("CDK2", "RB1", "CellCycle"),
    ("CDK2", "E2F1", "CellCycle"), ("CDK4", "RB1", "CellCycle"),
    ("CDK6", "RB1", "CellCycle"), ("CDK7", "CDK1", "CellCycle"),
    # DNA damage
    ("ATM", "TP53", "DDR"), ("ATM", "CHEK2", "DDR"),
    ("ATR", "CHEK1", "DDR"), ("ATR", "TP53", "DDR"),
    ("CHEK1", "CDC25C", "DDR"), ("CHEK2", "TP53", "DDR"),
    # NF-kB
    ("IKBKB", "NFKBIA", "NFkB"), ("IKBKB", "RELA", "NFkB"),
    ("CHUK", "NFKBIA", "NFkB"),
    # Stress
    ("MAPK14", "ATF2", "Stress"), ("MAPK14", "TP53", "Stress"),
    # TGF-beta
    ("TGFBR1", "SMAD2", "TGFb"), ("TGFBR1", "SMAD3", "TGFb"),
    ("BMPR1A", "SMAD1", "TGFb"),
    # SRC family
    ("SRC", "PTK2", "SRC"), ("SRC", "CTNNB1", "SRC"),
    ("SRC", "STAT3", "SRC"),
    # Other
    ("ABL1", "CRKL", "Other"), ("PRKCA", "GSK3B", "PKC"),
    ("PRKCA", "RAF1", "PKC"), ("PRKACA", "CREB1", "PKA"),
    ("PRKAA1", "MTOR", "AMPK"), ("STK11", "PRKAA1", "AMPK"),
    ("AURKA", "TP53", "Mitotic"), ("PLK1", "CDC25C", "Mitotic"),
    ("PLK1", "WEE1", "Mitotic"),
]

ks_df = pd.DataFrame(KINASE_SUBSTRATE_CORE, columns=['kinase', 'substrate', 'pathway'])
tf_to_kinases = {}
for _, row in ks_df.iterrows():
    sub = row['substrate']
    kin = row['kinase']
    if sub not in tf_to_kinases:
        tf_to_kinases[sub] = []
    tf_to_kinases[sub].append(kin)

print(f"  Core kinase-substrate interactions: {len(ks_df)}")
print(f"  Unique kinases: {ks_df['kinase'].nunique()}, substrates: {ks_df['substrate'].nunique()}")

# ═══════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════
print(f"\n[1] Loading data...", flush=True)

# Expression: rename CEL columns to cell line names and map the ENSG index to symbols
# Expression
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

print(f"  Genes: {len(expr_sym)}, Cell lines: {len(expr_sym.columns)}")

# Existing NetITH
netith = pd.read_csv(
    f"{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv",
    index_col=0
)
print(f"  Existing NetITH: {len(netith)} cell lines")

# Drug IC50
ic50_raw = pd.read_csv(f"{GDSC_DIR}/GDSC2_IC50_all.csv")
ic50_mat = ic50_raw.pivot_table(
    index='CELL_LINE_NAME', columns='DRUG_NAME',
    values='LN_IC50', aggfunc='mean'
)

# CollecTRI TF→target network (source/target/weight edges), precomputed pickle
# CollecTRI
with open('/tmp/collectri_net.pkl', 'rb') as f:
    net = pickle.load(f)
print(f"  CollecTRI edges: {len(net)}")

# ═══════════════════════════════════════════════════════
# 2. PREPARE GENE UNIVERSE (using focused 239-gene set)
# ═══════════════════════════════════════════════════════
print(f"\n[2] Preparing gene universe (focused 239-gene set)...", flush=True)

# Gene universe: focused CollecTRI genes plus all kinase/substrate genes present in expression
# Use focused genes for computational efficiency
focused_file = f"{ROOT}/results/focused_genes_collectri.txt"
with open(focused_file) as f:
    focused = set(line.strip() for line in f if line.strip())

# Add kinase and substrate genes to ensure phospho-signaling coverage
all_ks_genes = set(ks_df['kinase']) | set(ks_df['substrate'])
ks_in_expr = all_ks_genes & set(expr_sym.index)
expanded_focused = focused | ks_in_expr

genes_in_expr = sorted(expanded_focused & set(expr_sym.index))
print(f"  Original focused: {len(focused)}, KS genes added: {len(ks_in_expr - focused)}")
print(f"  Final gene set: {len(genes_in_expr)}")

gene_to_idx = {g: i for i, g in enumerate(genes_in_expr)}
n_genes = len(genes_in_expr)

# Keep only CollecTRI edges whose TF and target both lie in the gene universe
# Filter edges
edges = []
for _, row in net.iterrows():
    tf = row['source']; target = row['target']
    if tf in gene_to_idx and target in gene_to_idx:
        edges.append({
            'tf_idx': gene_to_idx[tf],
            'target_idx': gene_to_idx[target],
            'weight': float(row.get('weight', 1.0)),
        })
print(f"  Valid CollecTRI edges: {len(edges)}")

# Cell lines
common_cells = sorted(set(expr_sym.columns) & set(netith.index) & set(ic50_mat.index))
expr_sub = expr_sym[common_cells]
n_cells = len(common_cells)
print(f"  Cell lines: {n_cells}")

# Per-cell-line z-scored expression (rows = cells, cols = genes), clipped to [-3, 3]
# Expression matrix
expr_vals = expr_sub.loc[genes_in_expr].values.T  # (n_cells, n_genes)
expr_mean = expr_vals.mean(axis=0)
expr_std = expr_vals.std(axis=0) + 1e-10
expr_z = np.clip((expr_vals - expr_mean) / expr_std, -3, 3)

# Build kinase weight matrix: for each gene, its "kinase score" = mean |z| of kinases
kinase_gene_indices = {}
for gene in genes_in_expr:
    if gene in tf_to_kinases:
        kin_genes = [k for k in tf_to_kinases[gene] if k in gene_to_idx]
        if kin_genes:
            kinase_gene_indices[gene] = [gene_to_idx[k] for k in kin_genes]

print(f"  Genes with known kinase regulators: {len(kinase_gene_indices)}")

# ═══════════════════════════════════════════════════════
# 3. COMPUTE PHOSPHO-NetITH
# ═══════════════════════════════════════════════════════
print(f"\n[3] Computing Phospho-NetITH (kinase-weighted GRN entropy)...", flush=True)

alpha = 0.5  # kinase weight contribution
phospho_netith_vals = np.full(n_cells, np.nan)

# Kinase score per gene and cell line = mean |z| of the kinases phosphorylating that gene
# Compute per-gene kinase score for each cell line
# kinase_score[c, g] = mean |z| of kinases that phosphorylate gene g
kinase_scores = np.zeros((n_cells, n_genes))
for gene, kin_idxs in kinase_gene_indices.items():
    g_idx = gene_to_idx[gene]
    kinase_scores[:, g_idx] = np.mean(np.abs(expr_z[:, kin_idxs]), axis=1)

# Normalize kinase scores
ks_mean = kinase_scores.mean(axis=0)
ks_std = kinase_scores.std(axis=0) + 1e-10
kinase_scores = np.clip((kinase_scores - ks_mean) / ks_std, -3, 3)

# Phospho-NetITH per cell line: weighted adjacency, graph Laplacian, von Neumann entropy of its eigenvalue distribution
for i in range(n_cells):
    if (i + 1) % 200 == 0:
        print(f"    Cell line {i+1}/{n_cells}")
    
    A = np.zeros((n_genes, n_genes))
    for e in edges:
        tf_a = abs(expr_z[i, e['tf_idx']])
        tgt_a = abs(expr_z[i, e['target_idx']])
        # Kinase enhancement: edge weight boosted by TF's kinase regulators
        tf_ks = kinase_scores[i, e['tf_idx']]
        kinase_factor = 1.0 + alpha * max(0, tf_ks)  # only positive (activated) kinase effect
        A[e['tf_idx'], e['target_idx']] = e['weight'] * tf_a * tgt_a * kinase_factor
    
    deg = A.sum(axis=1)
    trace = deg.sum()
    
    if trace < 1e-10:
        phospho_netith_vals[i] = 0.0
        continue
    
    L = np.diag(deg) - A
    try:
        eigvals = eigh((L + L.T) / 2, eigvals_only=True)
        eigvals = np.clip(eigvals, 0, None)
        rho = eigvals / (trace + 1e-10)
        rho = np.clip(rho, 1e-12, 1.0)
        phospho_netith_vals[i] = -np.sum(rho * np.log2(rho))
    except:
        phospho_netith_vals[i] = np.nan

valid = ~np.isnan(phospho_netith_vals)
print(f"  Valid: {valid.sum()}/{n_cells}")
print(f"  Phospho-NetITH range: [{phospho_netith_vals[valid].min():.3f}, {phospho_netith_vals[valid].max():.3f}]")

# Spearman test (null: no association) between Phospho-NetITH and the original NetITH
# Compare with original
original_vals = netith.loc[common_cells, 'NetITH'].values
rho_po, p_po = spearmanr(phospho_netith_vals[valid], original_vals[valid])
print(f"  ρ(Phospho-NetITH, Original NetITH) = {rho_po:.4f} (p={p_po:.2e})")

# Save per-cell-line Phospho-NetITH together with the original NetITH
# Save
phospho_df = pd.DataFrame({
    'cell_line': common_cells,
    'phospho_netith': phospho_netith_vals,
    'original_netith': original_vals
}).set_index('cell_line')
phospho_df.to_csv(RESULTS_DIR / "ext_d6_phospho_netith.csv")

# ═══════════════════════════════════════════════════════
# 4. PATHWAY-LEVEL DECOMPOSITION
# ═══════════════════════════════════════════════════════
print(f"\n[4] Pathway-level NetITH decomposition...", flush=True)

# Pathway decomposition: reweight the GRN using only the kinases of one signaling pathway at a time
# For each pathway, compute NetITH using only kinases in that pathway
pathway_kinases = {}
for _, row in ks_df.iterrows():
    pw = row['pathway']
    if pw not in pathway_kinases:
        pathway_kinases[pw] = set()
    pathway_kinases[pw].add(row['kinase'])

pathway_netith = {}
for pw, kinases in sorted(pathway_kinases.items()):
    kin_in_net = [k for k in kinases if k in gene_to_idx]
    if len(kin_in_net) < 2:
        continue
    
    # Build pathway-specific kinase score matrix
    kin_idxs = [gene_to_idx[k] for k in kin_in_net]
    pw_kinase_scores = np.mean(np.abs(expr_z[:, kin_idxs]), axis=1)
    pw_ks = (pw_kinase_scores - pw_kinase_scores.mean()) / (pw_kinase_scores.std() + 1e-10)
    pw_ks = np.clip(pw_ks, -3, 3)
    
    # Compute pathway-specific Phospho-NetITH
    pw_entropies = np.full(n_cells, np.nan)
    for i in range(n_cells):
        A = np.zeros((n_genes, n_genes))
        for e in edges:
            tf_a = abs(expr_z[i, e['tf_idx']])
            tgt_a = abs(expr_z[i, e['target_idx']])
            pw_factor = 1.0 + alpha * max(0, pw_ks[i])
            A[e['tf_idx'], e['target_idx']] = e['weight'] * tf_a * tgt_a * pw_factor
        
        deg = A.sum(axis=1)
        trace = deg.sum()
        if trace < 1e-10:
            pw_entropies[i] = 0.0; continue
        
        L = np.diag(deg) - A
        try:
            eigvals = eigh((L + L.T) / 2, eigvals_only=True)
            eigvals = np.clip(eigvals, 0, None)
            rho = eigvals / (trace + 1e-10)
            rho = np.clip(rho, 1e-12, 1.0)
            pw_entropies[i] = -np.sum(rho * np.log2(rho))
        except:
            pw_entropies[i] = np.nan
    
    pw_valid = ~np.isnan(pw_entropies)
    # Spearman test (null: no association) of pathway-specific entropy vs original NetITH
    if pw_valid.sum() > 50:
        rho_pw, p_pw = spearmanr(pw_entropies[pw_valid], original_vals[pw_valid])
        pathway_netith[pw] = {
            'n_kinases': len(kin_in_net),
            'mean_phospho_netith': np.mean(pw_entropies[pw_valid]),
            'rho_with_original': rho_pw,
            'p_value': p_pw
        }

# Save pathway-level Phospho-NetITH decomposition
if pathway_netith:
    pw_df = pd.DataFrame(pathway_netith).T.sort_values('rho_with_original', key=abs, ascending=False)
    pw_df.to_csv(RESULTS_DIR / "ext_d6_kinase_pathway_netith.csv")
    print(f"  Pathways analyzed: {len(pathway_netith)}")
    for pw, info in sorted(pathway_netith.items(), key=lambda x: abs(x[1]['rho_with_original']), reverse=True):
        print(f"    {pw:<15}: {info['n_kinases']} kinases, "
              f"ρ(original)={info['rho_with_original']:+.4f} (p={info['p_value']:.4f})")
else:
    print(f"  No pathways with sufficient kinase coverage")

# ═══════════════════════════════════════════════════════
# 5. DRUG SENSITIVITY: Phospho-NetITH vs Original
# ═══════════════════════════════════════════════════════
print(f"\n[5] Drug Sensitivity: Phospho-NetITH vs Original NetITH...", flush=True)

# Kinase-only NetITH: restrict the GRN to edges whose TF is a known kinase
# Also compute a "kinase-only" NetITH: use only edges where TF is a known kinase
kinase_tfs = set(ks_df['kinase'])
kinase_tf_edges = [e for e in edges if genes_in_expr[e['tf_idx']] in kinase_tfs]
print(f"  Kinase-TF edges: {len(kinase_tf_edges)}/{len(edges)}")

# Compute kinase-only NetITH
kinase_only_netith = np.full(n_cells, np.nan)
for i in range(n_cells):
    A = np.zeros((n_genes, n_genes))
    for e in kinase_tf_edges:
        tf_a = abs(expr_z[i, e['tf_idx']])
        tgt_a = abs(expr_z[i, e['target_idx']])
        A[e['tf_idx'], e['target_idx']] = e['weight'] * tf_a * tgt_a
    
    deg = A.sum(axis=1); trace = deg.sum()
    if trace < 1e-10:
        kinase_only_netith[i] = 0.0; continue
    
    L = np.diag(deg) - A
    try:
        eigvals = eigh((L + L.T) / 2, eigvals_only=True)
        eigvals = np.clip(eigvals, 0, None)
        rho = eigvals / (trace + 1e-10)
        rho = np.clip(rho, 1e-12, 1.0)
        kinase_only_netith[i] = -np.sum(rho * np.log2(rho))
    except:
        kinase_only_netith[i] = np.nan

ko_valid = ~np.isnan(kinase_only_netith)
rho_ko, p_ko = spearmanr(kinase_only_netith[ko_valid], original_vals[ko_valid])
print(f"  Kinase-only NetITH: {ko_valid.sum()} valid, ρ(original)={rho_ko:.4f}")

# Drug sensitivity comparison
common_all = sorted(set(common_cells) & set(ic50_mat.index))
idx_map = {c: i for i, c in enumerate(common_cells)}

drug_comparison = []
for drug in sorted(ic50_mat.columns):
    ic50_col = ic50_mat.loc[common_all, drug]
    ic50_v = ic50_col.values.astype(float)
    mask = ~np.isnan(ic50_v)
    if mask.sum() < 50:
        continue
    
    # Align indices
    cell_indices = [idx_map[c] for c in common_all if c in idx_map]
    cell_indices = [i for i in cell_indices if mask[list(common_all).index(common_cells[i])] 
                   if common_cells[i] in common_all]
    # Simplified: compute for common subset
    common_drug_cells = sorted(set(common_all) & set(common_cells))
    sub_idx = [common_cells.index(c) for c in common_drug_cells]
    
    orig_sub = original_vals[sub_idx]
    phos_sub = phospho_netith_vals[sub_idx]
    ko_sub = kinase_only_netith[sub_idx]
    ic50_sub = ic50_mat.loc[common_drug_cells, drug].values.astype(float)
    
    vmask = ~(np.isnan(orig_sub) | np.isnan(phos_sub) | np.isnan(ic50_sub) | np.isnan(ko_sub))
    if vmask.sum() < 50:
        continue
    
    # Spearman test per drug (null: no association) of each NetITH variant vs IC50
    r_orig, p_orig = spearmanr(orig_sub[vmask], ic50_sub[vmask])
    r_phos, p_phos = spearmanr(phos_sub[vmask], ic50_sub[vmask])
    r_ko, p_ko = spearmanr(ko_sub[vmask], ic50_sub[vmask])
    
    # Combined score
    combined = phos_sub + ko_sub - orig_sub  # residual kinase signal
    r_comb, p_comb = spearmanr(combined[vmask], ic50_sub[vmask])
    
    drug_comparison.append({
        'drug': drug, 'n': int(vmask.sum()),
        'rho_original': r_orig, 'p_original': p_orig,
        'rho_phospho': r_phos, 'p_phospho': p_phos,
        'rho_kinase_only': r_ko, 'p_kinase_only': p_ko,
        'rho_combined': r_comb, 'p_combined': p_comb,
        'delta_phospho': r_phos - r_orig if not np.isnan(r_orig) and not np.isnan(r_phos) else np.nan,
        'delta_combined': r_comb - r_orig if not np.isnan(r_orig) and not np.isnan(r_comb) else np.nan,
    })

# Save drug-wise comparison of NetITH variants for drug-sensitivity prediction
drug_comp_df = pd.DataFrame(drug_comparison).sort_values('delta_phospho', key=abs, ascending=False)
drug_comp_df.to_csv(RESULTS_DIR / "ext_d6_drug_comparison.csv", index=False)

mean_orig = drug_comp_df['rho_original'].dropna().mean()
mean_phos = drug_comp_df['rho_phospho'].dropna().mean()
mean_ko = drug_comp_df['rho_kinase_only'].dropna().mean()
mean_comb = drug_comp_df['rho_combined'].dropna().mean()

print(f"  Drugs tested: {len(drug_comp_df)}")
print(f"\n  Mean drug correlation |ρ|:")
print(f"    Original NetITH : {mean_orig:+.4f}")
print(f"    Phospho-NetITH  : {mean_phos:+.4f}  (Δ={mean_phos-mean_orig:+.4f})")
print(f"    Kinase-Only     : {mean_ko:+.4f}  (Δ={mean_ko-mean_orig:+.4f})")
print(f"    Combined        : {mean_comb:+.4f}  (Δ={mean_comb-mean_orig:+.4f})")

# Top drugs where kinase information helps
print(f"\n  Top 10 drugs where phospho information improves prediction:")
best_delta = drug_comp_df.dropna(subset=['delta_phospho']).nlargest(10, 'delta_phospho')
for _, row in best_delta.iterrows():
    print(f"    {row['drug']:<35} orig={row['rho_original']:+.4f} → "
          f"phos={row['rho_phospho']:+.4f}  Δ={row['delta_phospho']:+.4f}")

# ═══════════════════════════════════════════════════════
# 6. SUMMARY
# ═══════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SUMMARY — Direction 6: Phospho-NetITH")
print(f"{'='*70}")

print(f"\n  A. Phospho-NetITH vs Original:")
print(f"    ρ = {rho_po:.4f} (p={p_po:.2e})")
print(f"    Phospho-NetITH range: [{phospho_netith_vals[valid].min():.3f}, {phospho_netith_vals[valid].max():.3f}]")

print(f"\n  B. Kinase-Only NetITH:")
print(f"    ρ(original) = {rho_ko:.4f}")
print(f"    Using {len(kinase_tf_edges)} edges with kinase TFs")

print(f"\n  C. Pathway Decomposition:")
if pathway_netith:
    for pw, info in sorted(pathway_netith.items(), key=lambda x: abs(x[1]['rho_with_original']), reverse=True)[:5]:
        print(f"    {pw:<15}: {info['n_kinases']} kinases, ρ={info['rho_with_original']:+.4f}")
else:
    print(f"    No pathways with sufficient coverage")

print(f"\n  D. Drug Prediction:")
print(f"    Original    : {mean_orig:+.4f}")
print(f"    Phospho     : {mean_phos:+.4f}")
print(f"    Combined    : {mean_comb:+.4f}")

print(f"\nDone. All results in {RESULTS_DIR}/")
