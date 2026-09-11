"""run_gse131907_validation_v3.py — GSE131907 external validation of per-cell NetITH (shell-assisted streaming).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/geo/GSE131907/{GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz, GSE131907_Lung_Cancer_cell_annotation.txt.gz}; results/focused_genes_collectri.txt; /tmp/collectri_net.pkl (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/gse131907/{cell_entropy_results.csv, patient_netith.csv, cell_entropies.npy}; results/gse131907/figures/gse131907_validation.png
Pipeline: single-cell stage — see repository README
"""
import numpy as np
import pandas as pd
import gzip
import pickle
import os
import sys
import subprocess
import warnings
from pathlib import Path
from typing import Set, List
import scipy.linalg
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = f"{DATA_ROOT}/geo/GSE131907"
EXPR_PATH = f"{DATA_DIR}/GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt.gz"
ANNO_PATH = f"{DATA_DIR}/GSE131907_Lung_Cancer_cell_annotation.txt.gz"
OUTPUT_DIR = Path(f"{ROOT}/results/gse131907")
N_CELLS_STRATUM = 150
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use focused gene set (same as main pipeline for consistency)
FOCUSED_GENES_FILE = f"{ROOT}/results/focused_genes_collectri.txt"
COLLECTRI_NET_FILE = "/tmp/collectri_net.pkl"


def stratified_cell_sample(anno: pd.DataFrame, n_per_stratum: int) -> Set[str]:
    """Sample cells stratified by Sample_Origin x Cell_type."""
    rng = np.random.RandomState(SEED)
    sampled = set()
    strata_counts = {}
    for (origin, ct), group in anno.groupby(['Sample_Origin', 'Cell_type']):
        n = min(len(group), n_per_stratum)
        if n > 0:
            idx = rng.choice(group.index, size=n, replace=False)
            sampled.update(idx.tolist())
            strata_counts[(origin, ct)] = n
    print(f"Sampled {len(sampled)} cells from {len(anno)} ({len(strata_counts)} strata)")
    return sampled


def fast_extract_expression(
    expr_path: str,
    sampled_cells: Set[str],
) -> pd.DataFrame:
    """Streaming: gunzip -c piped to Python, line-by-line with gene filter."""
    
    # ── Load focused gene set ──
    with open(FOCUSED_GENES_FILE) as f:
        target_genes = set(line.strip() for line in f if line.strip())
    print(f"  Target genes: {len(target_genes)}")
    
    # ── Open pipe, read header ──
    print("Opening pipe: gunzip -c ...")
    proc = subprocess.Popen(
        ['gunzip', '-c', expr_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1024*1024,  # 1MB buffer
    )
    
    # Read header line
    header_line = proc.stdout.readline()
    header = header_line.strip().split('\t')
    print(f"  Header: {len(header)} columns")
    
    # Map cell barcodes to column indices
    col_to_idx = {name: i for i, name in enumerate(header)}
    keep_cells = [c for c in sampled_cells if c in col_to_idx]
    keep_indices = [col_to_idx[c] for c in keep_cells]
    print(f"  Matched {len(keep_cells)}/{len(sampled_cells)} cells")
    
    # ── Stream through gene rows ──
    print("Streaming gene rows (filtering focused genes)...")
    gene_rows = {}
    line_count = 0
    matched = 0
    
    for line in proc.stdout:
        line_count += 1
        if line_count % 1000000 == 0:
            print(f"    Scanned {line_count/1e6:.1f}M lines, matched {matched} genes...")
        
        gene = line.split('\t', 1)[0]
        if gene in target_genes:
            parts = line.strip().split('\t')
            values = []
            for ci in keep_indices:
                if ci < len(parts):
                    try:
                        values.append(float(parts[ci]))
                    except ValueError:
                        values.append(0.0)
                else:
                    values.append(0.0)
            gene_rows[gene] = values
            matched += 1
    
    proc.wait()
    print(f"  Scanned {line_count} lines, extracted {len(gene_rows)}/{len(target_genes)} genes")
    
    # ── Build DataFrame: cells x genes ──
    gene_list = sorted(gene_rows.keys())
    if not gene_list:
        print("ERROR: No genes extracted!")
        proc.terminate()
        return None
    
    expr_arr = np.column_stack([gene_rows[g] for g in gene_list])
    expr_df = pd.DataFrame(expr_arr, index=keep_cells, columns=gene_list)
    print(f"  Expression matrix: {expr_df.shape}")
    print(f"  Value range: [{expr_df.values.min():.2f}, {expr_df.values.max():.2f}]")
    
    return expr_df


def compute_entropy(expr_df: pd.DataFrame, net: pd.DataFrame) -> np.ndarray:
    """Per-cell von Neumann entropy via CollecTRI GRN."""
    print(f"\nComputing entropy for {len(expr_df)} cells...")
    
    genes = list(expr_df.columns)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    
    # Build edge list from CollecTRI
    edges = []
    for _, row in net.iterrows():
        tf = row['source']
        target = row['target']
        if tf in gene_to_idx and target in gene_to_idx:
            edges.append({
                'tf_idx': gene_to_idx[tf],
                'target_idx': gene_to_idx[target],
                'weight': float(row.get('weight', 1.0)),
            })
    
    n_genes = len(genes)
    n_cells = len(expr_df)
    print(f"  Genes: {n_genes}, Edges: {len(edges)}")
    
    if len(edges) == 0:
        print("  WARNING: No edges! Using correlation fallback.")
        corr = np.corrcoef(expr_df.values.T)
        np.fill_diagonal(corr, 0)
        top_n = min(1000, n_genes * 5)
        triu_idx = np.triu_indices(n_genes, k=1)
        corr_abs = np.abs(corr)
        top_edges = np.argsort(corr_abs[triu_idx])[::-1][:top_n]
        for ei in top_edges:
            i, j = triu_idx[0][ei], triu_idx[1][ei]
            edges.append({'tf_idx': i, 'target_idx': j, 'weight': abs(corr[i, j])})
    
    # Z-score normalize
    expr_vals = expr_df.values.astype(np.float32)
    expr_mean = expr_vals.mean(axis=0)
    expr_std = expr_vals.std(axis=0) + 1e-10
    expr_z = np.clip((expr_vals - expr_mean) / expr_std, -3, 3)
    
    entropies = np.full(n_cells, np.nan)
    
    for i in range(n_cells):
        if (i + 1) % 1000 == 0:
            print(f"    Cell {i+1}/{n_cells}")
        
        A = np.zeros((n_genes, n_genes))
        for e in edges:
            tf_act = abs(float(expr_z[i, e['tf_idx']]))
            tgt_act = abs(float(expr_z[i, e['target_idx']]))
            A[e['tf_idx'], e['target_idx']] = e['weight'] * tf_act * tgt_act
        
        A = A + A.T  # FIX(eigvalsh-audit 2026-08-16): symmetrize directed TF->target adjacency before Laplacian (eigvalsh silently used one triangle)
        degrees = A.sum(axis=1)
        L = np.diag(degrees) - A
        trace = degrees.sum()
        
        if trace < 1e-10:
            entropies[i] = 0.0
            continue
        
        try:
            eigs = scipy.linalg.eigvalsh(L)
            eigs = np.clip(eigs, 0, None)
            rho = eigs / (trace + 1e-10)
            rho = np.clip(rho, 1e-12, 1.0)
            entropies[i] = -np.sum(rho * np.log2(rho))
        except Exception:
            entropies[i] = np.nan
    
    valid = ~np.isnan(entropies)
    print(f"  Valid: {valid.sum()}/{n_cells}")
    if valid.sum() > 0:
        print(f"  Entropy range: [{entropies[valid].min():.3f}, {entropies[valid].max():.3f}]")
    return entropies


def analyze_and_plot(entropies, expr_df, anno_full, output_dir):
    """Analyze and generate figures."""
    from scipy.stats import mannwhitneyu
    
    valid = ~np.isnan(entropies)
    cells = expr_df.index[valid]
    ent_valid = entropies[valid]
    
    anno = anno_full.loc[anno_full.index.isin(cells)].copy()
    cell_to_ent = dict(zip(cells, ent_valid))
    anno['vn_entropy'] = [cell_to_ent[c] for c in anno.index]
    
    print(f"\n{'='*60}")
    print("[Analysis]  Cells: {len(anno)}")
    print(f"{'='*60}")
    
    # Tumor vs Normal
    tumor_mask = anno['Sample_Origin'].isin(['tLung', 'tL/B'])
    normal_mask = anno['Sample_Origin'].isin(['nLung'])
    if tumor_mask.sum() > 10 and normal_mask.sum() > 10:
        t = anno.loc[tumor_mask, 'vn_entropy']
        n_vals = anno.loc[normal_mask, 'vn_entropy']
        stat, p = mannwhitneyu(t, n_vals)
        d = (t.mean() - n_vals.mean()) / anno['vn_entropy'].std()
        print(f"\n  Tumor vs Normal: d={d:.3f}, p={p:.4f}")
        print(f"    Tumor: {t.mean():.3f} +/- {t.std():.3f} (n={len(t)})")
        print(f"    Normal: {n_vals.mean():.3f} +/- {n_vals.std():.3f} (n={len(n_vals)})")
    
    # By origin
    print(f"\n  By Sample_Origin:")
    for origin in sorted(anno['Sample_Origin'].unique()):
        v = anno[anno['Sample_Origin'] == origin]['vn_entropy']
        print(f"    {origin}: {v.mean():.3f}+/-{v.std():.3f} (n={len(v)})")
    
    # Epithelial: tumor vs normal vs met
    epi = anno[anno['Cell_type'] == 'Epithelial cells']
    for label, origins in [('Normal', ['nLung']), ('Tumor', ['tLung', 'tL/B']),
                            ('Met', ['mLN', 'mBrain'])]:
        v = epi[epi['Sample_Origin'].isin(origins)]['vn_entropy']
        if len(v) > 0:
            print(f"    Epi-{label}: {v.mean():.3f}+/-{v.std():.3f} (n={len(v)})")
    
    # By cell type (top 8)
    print(f"\n  By Cell_type:")
    for ct, cnt in anno['Cell_type'].value_counts().head(8).items():
        v = anno[anno['Cell_type'] == ct]['vn_entropy']
        print(f"    {ct}: {v.mean():.3f}+/-{v.std():.3f} (n={len(v)})")
    
    # ── Plots ──
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # A: By origin
    ax = axes[0, 0]
    order = [o for o in ['nLung', 'tLung', 'tL/B', 'nLN', 'mLN', 'mBrain', 'PE']
             if o in anno['Sample_Origin'].unique()]
    colors = ['#2ecc71' if 'n' in o else '#e74c3c' if 't' in o else '#f39c12' for o in order]
    for i, (o, c) in enumerate(zip(order, colors)):
        v = anno[anno['Sample_Origin'] == o]['vn_entropy']
        bp = ax.boxplot([v], positions=[i], patch_artist=True, widths=0.5, showfliers=False)
        bp['boxes'][0].set_facecolor(c)
        bp['boxes'][0].set_alpha(0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30)
    ax.set_ylabel('von Neumann Entropy (bits)')
    ax.set_title('Per-Cell Entropy by Tissue Origin')
    
    # B: By cell type
    ax = axes[0, 1]
    ct_order = anno.groupby('Cell_type')['vn_entropy'].median().sort_values().index.tolist()
    for i, ct in enumerate(ct_order[:8]):
        v = anno[anno['Cell_type'] == ct]['vn_entropy']
        bp = ax.boxplot([v], positions=[i], patch_artist=True, widths=0.5, showfliers=False)
        bp['boxes'][0].set_facecolor(sns.color_palette("Set2")[i % 8])
    ax.set_xticks(range(min(8, len(ct_order))))
    ax.set_xticklabels(ct_order[:8], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('von Neumann Entropy')
    ax.set_title('Per-Cell Entropy by Cell Type')
    
    # C: Epithelial density
    ax = axes[1, 0]
    for label, origins, c in [('Normal', ['nLung'], '#2ecc71'),
                               ('Tumor', ['tLung', 'tL/B'], '#e74c3c'),
                               ('Metastasis', ['mLN', 'mBrain'], '#f39c12')]:
        v = epi[epi['Sample_Origin'].isin(origins)]['vn_entropy']
        if len(v) > 5:
            ax.hist(v, bins=25, alpha=0.4, color=c, label=f'{label} (n={len(v)})', density=True)
    ax.set_xlabel('von Neumann Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Epithelial Cells: Entropy Distribution')
    ax.legend(fontsize=8)
    
    # D: Patient NetITH
    ax = axes[1, 1]
    p_stats = []
    for patient in anno['Sample'].unique():
        p_data = anno[anno['Sample'] == patient]
        if len(p_data) < 10:
            continue
        hist, _ = np.histogram(p_data['vn_entropy'], bins=min(12, len(p_data) // 3), density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        p_stats.append({
            'patient': patient,
            'origin': p_data['Sample_Origin'].mode().iloc[0],
            'netith': -np.sum(hist * np.log2(hist + 1e-10)),
            'n': len(p_data)
        })
    p_df = pd.DataFrame(p_stats)
    for i, o in enumerate(p_df['origin'].unique()):
        v = p_df[p_df['origin'] == o]['netith']
        if len(v) > 0:
            bp = ax.boxplot([v], positions=[i], patch_artist=True, widths=0.5, showfliers=False)
            c = '#2ecc71' if 'n' in o else '#e74c3c'
            bp['boxes'][0].set_facecolor(c)
            bp['boxes'][0].set_alpha(0.6)
    ax.set_xticks(range(len(p_df['origin'].unique())))
    ax.set_xticklabels(p_df['origin'].unique(), rotation=30)
    ax.set_ylabel('NetITH')
    ax.set_title('Patient NetITH')
    
    plt.suptitle('GSE131907: NetEntropy External Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_dir = Path(output_dir) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / 'gse131907_validation.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: gse131907_validation.png")
    
    # Save
    anno.to_csv(os.path.join(output_dir, 'cell_entropy_results.csv'))
    p_df.to_csv(os.path.join(output_dir, 'patient_netith.csv'), index=False)
    np.save(os.path.join(output_dir, 'cell_entropies.npy'), entropies)
    
    return anno, p_df


def main():
    print("=" * 60)
    print("  GSE131907 Lung Cancer Validation (v3 shell-assisted)")
    print("=" * 60)
    
    # Load annotation
    print("\n[1/5] Loading annotation...")
    with gzip.open(ANNO_PATH, 'rt') as f:
        anno = pd.read_csv(f, sep='\t', index_col=0)
    print(f"  {len(anno)} cells, {anno['Sample'].nunique()} samples")
    
    # Sample cells
    print("\n[2/5] Sampling cells...")
    sampled = stratified_cell_sample(anno, N_CELLS_STRATUM)
    
    # Load expression (fast shell-assisted)
    print("\n[3/5] Extracting expression (shell-assisted)...")
    expr_df = fast_extract_expression(EXPR_PATH, sampled)
    
    if expr_df is None or expr_df.shape[1] < 10:
        print("ERROR: Too few genes. Aborting.")
        return
    
    # Load CollecTRI network
    print("\n[4/5] Loading CollecTRI network...")
    with open(COLLECTRI_NET_FILE, 'rb') as f:
        net = pickle.load(f)
    print(f"  {len(net)} edges")
    
    # Compute entropy
    print("\n[5/5] Computing entropy...")
    entropies = compute_entropy(expr_df, net)
    
    # Analyze
    analyze_and_plot(entropies, expr_df, anno, str(OUTPUT_DIR))
    
    print(f"\nDone! Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
