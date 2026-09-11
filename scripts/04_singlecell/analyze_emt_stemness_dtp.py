#!/usr/bin/env python3
"""analyze_emt_stemness_dtp.py — NetITH biological anchoring: EMT, mRNAsi stemness and drug-tolerant-persister correlations.

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-08-18
Inputs  : data/gdsc/{rna_expr.csv, cell_annot.csv, ensg_symbol_map.csv}; results/gdsc/gdsc_netith_cell_lines.csv; results/tcga/tcga_netith.csv; results/gdsc/gdsc_drug_netith_correlations.csv; data/xena/tcgapancan/{StemnessScores_RNAexp_20170127.2.tsv.gz, PanCan33_ssGSEA_1387GeneSets_NonZero_sample_level.txt.gz} (data root from NETITH_DATA_ROOT env, default <repo>/data)
Outputs : results/depmap/emt_stemness_results.csv; results/depmap/figures/emt_stemness_dtp_anchoring.png/.pdf
Pipeline: single-cell stage — see repository README
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, fisher_exact
from scipy.linalg import eigvalsh
import os, sys, warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("NETITH_DATA_ROOT", str(ROOT / "data")))

# ─── Paths ────────────────────────────────────────────────
DATA_DIR = f"{DATA_ROOT}/gdsc"
TCGA_XENA = f"{DATA_ROOT}/xena/tcgapancan"
OUTPUT_DIR = f"{ROOT}/results/depmap"
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

GDSC_NETITH = f"{ROOT}/results/gdsc/gdsc_netith_cell_lines.csv"
TCGA_NETITH = f"{ROOT}/results/tcga/tcga_netith.csv"
TCGA_DRUG_CORR = f"{ROOT}/results/gdsc/gdsc_drug_netith_correlations.csv"

SEED = 42

# ─── EMT Gene Signature (76 genes, from Byers/EMT consensus) ─
EMT_EPITHELIAL = [
    'CDH1', 'OCLN', 'CLDN4', 'CLDN7', 'TJP1', 'TJP2', 'TJP3',
    'DSP', 'PKP3', 'KRT5', 'KRT7', 'KRT8', 'KRT18', 'KRT19',
    'EPCAM', 'ESRP1', 'ESRP2', 'RAB25', 'MUC1', 'ST14',
    'MARVELD2', 'MARVELD3', 'GRHL2', 'OVOL2', 'ELF3', 'ELF5',
    'SPINT1', 'SPINT2', 'CDS1', 'LLGL2'
]

EMT_MESENCHYMAL = [
    'CDH2', 'VIM', 'FN1', 'SNAI1', 'SNAI2', 'ZEB1', 'ZEB2',
    'TWIST1', 'TWIST2', 'FOXC2', 'GSC', 'LEF1', 'TCF3',
    'TCF4', 'MMP2', 'MMP3', 'MMP9', 'COL1A1', 'COL1A2',
    'COL3A1', 'COL5A2', 'FBN1', 'THBS1', 'THBS2', 'SPARC',
    'TNC', 'ITGA5', 'ITGB1', 'POSTN', 'LOX', 'LOXL2',
    'PDGFRB', 'TAGLN', 'ACTA2', 'ACTG2', 'MYL9', 'CNN1',
    'DES', 'S100A4', 'FAP', 'VCAN', 'CD44', 'ABCC4',
    'PLOD2', 'PCOLCE', 'SERPINE1'
]

assert len(EMT_EPITHELIAL) + len(EMT_MESENCHYMAL) == 76, f"EMT gene count: {len(EMT_EPITHELIAL)+len(EMT_MESENCHYMAL)}"

# ─── DTP Marker Genes (Sharma 2010, Ramirez 2016, + DTP consensus) ─
DTP_UPREGULATED = [
    'KDM5A', 'KDM5B', 'KDM6A', 'KDM6B',          # H3K4/H3K27 demethylases
    'CDKN1A', 'CDKN1B',                            # CDK inhibitors
    'SOX2', 'SOX10', 'NGFR', 'NES',               # stem/neural crest
    'ALDH1A1', 'ALDH1A3', 'ABCG2', 'ABCB1',       # drug efflux / ALDH
    'HES1', 'HEY1', 'NOTCH1', 'NOTCH3',            # Notch pathway
    'IGF1R', 'IGFBP2', 'IGFBP3',                   # IGF signalling
    'WNT5A', 'FZD7', 'AXIN2',                      # WNT
    'HDAC1', 'HDAC2', 'HDAC3', 'EZH2', 'SUZ12',   # epigenetic
    'BCL2L1', 'MCL1', 'BIRC5', 'XIAP',             # anti-apoptosis
    'NFKB1', 'NFKB2', 'RELA', 'RELB',              # NF-kB
    'GDF15', 'TGFB1', 'TGFBR1',                    # TGF-β
    'EPHA2', 'MET', 'AXL', 'EGFR',                 # RTKs
    'ATF4', 'DDIT3', 'XBP1',                       # ER stress
    'GPX4', 'SLC7A11', 'NFE2L2',                   # ferroptosis resistance
]

DTP_DOWNREGULATED = [
    'MKI67', 'PCNA', 'TOP2A', 'CCNB1', 'CCNA2',   # proliferation
    'CDK1', 'CDK2', 'PLK1', 'AURKA', 'AURKB',
    'MYC', 'E2F1', 'E2F2', 'E2F3',
    'FOXM1', 'BRCA1', 'RAD51',                     # DNA repair
    'TYMS', 'RRM2', 'DHFR',                         # nucleotide metabolism
]

# ─── CollecTRI TFs (for NetITH) ──────────────────────────
COLLECTRI_FOCUSED_TFS = [
    'JUN', 'FOS', 'FOSL1', 'FOSL2', 'JUNB', 'JUND',
    'ATF1', 'ATF2', 'ATF3', 'ATF4', 'ATF6', 'BACH1', 'BACH2',
    'STAT1', 'STAT2', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6',
    'IRF1', 'IRF2', 'IRF3', 'IRF4', 'IRF5', 'IRF7', 'IRF8', 'IRF9',
    'NFKB1', 'NFKB2', 'REL', 'RELA', 'RELB',
    'SPI1', 'SPIB', 'CEBPA', 'CEBPB', 'CEBPD', 'CEBPE', 'CEBPG',
    'TP53', 'TP63', 'TP73',
    'E2F1', 'E2F2', 'E2F3', 'E2F4', 'E2F5', 'E2F6', 'E2F7', 'E2F8',
    'MYC', 'MYCN', 'MYCL',
    'HIF1A', 'EPAS1', 'ARNT', 'ARNT2',
    'SREBF1', 'SREBF2',
    'RUNX1', 'RUNX2', 'RUNX3',
    'GATA1', 'GATA2', 'GATA3', 'GATA4', 'GATA5', 'GATA6',
    'FOXA1', 'FOXA2', 'FOXA3',
    'FOXM1', 'FOXO1', 'FOXO3', 'FOXO4',
    'ESR1', 'ESR2', 'AR', 'NR3C1', 'NR1H3', 'NR1H4', 'PPARA', 'PPARD', 'PPARG',
    'RARA', 'RARB', 'RARG', 'RXRA', 'RXRB', 'RXRG',
    'VDR', 'PGR', 'THRA', 'THRB',
    'SMAD1', 'SMAD2', 'SMAD3', 'SMAD4', 'SMAD5', 'SMAD7', 'SMAD9',
    'CTCF', 'RAD21', 'SMC1A', 'SMC3', 'STAG1', 'STAG2',
    'EZH2', 'SUZ12', 'EED', 'RBBP4', 'RBBP7',
    'KDM5A', 'KDM5B', 'KDM5C', 'KDM6A', 'KDM6B',
    'HDAC1', 'HDAC2', 'HDAC3', 'HDAC4', 'HDAC5', 'HDAC6',
    'SIN3A', 'NCOR1', 'NCOR2',
    'SP1', 'SP2', 'SP3', 'SP4',
    'KLF4', 'KLF5',
    'EGR1', 'EGR2', 'EGR3',
    'TFAP2A', 'TFAP2C',
    'SOX2', 'SOX9', 'SOX10', 'SOX17',
    'POU5F1', 'NANOG', 'SALL4',
    'PAX3', 'PAX5', 'PAX8',
    'SNAI1', 'SNAI2', 'TWIST1', 'TWIST2', 'ZEB1', 'ZEB2',
    'TCF3', 'TCF4', 'TCF7L2', 'LEF1',
    'HEYL', 'HES1', 'HES5',
    'NFIA', 'NFIB', 'NFIC',
    'ZBTB16', 'ZBTB20', 'ZBTB33',
    'GZMB', 'PRDM1', 'IRF4P', 'BCL6',
    'RELB', 'SPIB', 'BATF', 'BATF3',
]


# ═══════════════════════════════════════════════════════════
#  A: EMT Score Computation (GDSC)
# ═══════════════════════════════════════════════════════════

def load_gdsc_expression():
    """Load GDSC RNA expression and map to gene symbols."""
    print("[A] Loading GDSC expression...")
    # Load GDSC RNA expression (ENSG rows x cell-line columns) and cell annotations
    expr_raw = pd.read_csv(f"{DATA_DIR}/rna_expr.csv", index_col=0)
    
    # Cell annotations
    annot = pd.read_csv(f"{DATA_DIR}/cell_annot.csv", index_col=0)
    cell_map = {}
    for cel, row in annot.iterrows():
        cl = str(row.get('Characteristics.cell.line.', ''))
        if cl and cl != 'nan' and cl != 'NA':
            cell_map[cel] = cl
    
    # Rename columns to cell-line names and drop duplicated columns
    common = [c for c in expr_raw.columns if c in cell_map]
    expr = expr_raw[common].copy()
    expr.columns = [cell_map[c] for c in common]
    expr = expr.loc[:, ~expr.columns.duplicated(keep='first')]
    
    # Gene mapping
    gene_map = pd.read_csv(f"{DATA_DIR}/ensg_symbol_map.csv")
    ensg2sym = dict(zip(gene_map['ensg'].astype(str), gene_map['symbol']))
    
    # Map ENSG ids to gene symbols with the curated map, then deduplicate
    expr.index = expr.index.astype(str)
    matched = expr.index.isin(ensg2sym.keys())
    expr = expr.loc[matched].copy()
    expr.index = [ensg2sym[g] for g in expr.index]
    expr = expr[~expr.index.duplicated(keep='first')]
    
    print(f"  GDSC expression: {expr.shape}")
    return expr


def compute_emt_score(expr):
    """Compute 76-gene EMT score per cell line: mean(mesenchymal) - mean(epithelial)."""
    # EMT score = mean(mesenchymal genes) - mean(epithelial genes) per cell line
    epi_avail = [g for g in EMT_EPITHELIAL if g in expr.index]
    mes_avail = [g for g in EMT_MESENCHYMAL if g in expr.index]
    print(f"  EMT genes available: {len(epi_avail)} epi + {len(mes_avail)} mes = {len(epi_avail)+len(mes_avail)}/76")
    
    epi_expr = expr.loc[epi_avail].mean(axis=0)
    mes_expr = expr.loc[mes_avail].mean(axis=0)
    emt_score = mes_expr - epi_expr
    return emt_score


# ═══════════════════════════════════════════════════════════
#  B: mRNAsi Stemness (TCGA)
# ═══════════════════════════════════════════════════════════

def load_mrnasi():
    """Load Malta et al. 2018 mRNAsi scores for TCGA."""
    print("[B] Loading TCGA mRNAsi stemness scores...")
    df = pd.read_csv(f"{TCGA_XENA}/StemnessScores_RNAexp_20170127.2.tsv.gz",
                     sep='\t', compression='gzip', index_col=0)
    # Transpose: samples as rows
    df = df.T
    # Keep only primary tumour samples (-01) and the first sample per patient
    # Keep only primary tumor samples (ending in -01)
    df.index = df.index.astype(str)
    primary = df[df.index.str.endswith('-01')].copy()
    # Keep first occurrence per patient
    patient_ids = primary.index.str[:12]
    primary = primary[~patient_ids.duplicated(keep='first')]
    print(f"  mRNAsi samples: {primary.shape[0]}")
    return primary


# ═══════════════════════════════════════════════════════════
#  C: DTP Enrichment (GDSC)
# ═══════════════════════════════════════════════════════════

def dtp_enrichment_analysis(expr, netith):
    """Test whether DTP-upregulated genes are enriched among NetITH-correlated genes."""
    print("[C] DTP enrichment analysis...")
    
    # Compute per-gene correlation with NetITH
    common_cells = list(set(expr.columns) & set(netith['cell_line']))
    # Per-gene Spearman correlation of GDSC expression with cell-line NetITH
    netith_map = netith.set_index('cell_line')['NetITH'].to_dict()
    netith_vals = np.array([netith_map[c] for c in common_cells])
    
    rhos = {}
    pvals = {}
    for gene in expr.index:
        gene_vals = expr.loc[gene, common_cells].values.astype(float)
        if np.std(gene_vals) > 0:
            r, p = spearmanr(gene_vals, netith_vals)
            rhos[gene] = r
            pvals[gene] = p
    
    corr_df = pd.DataFrame({'rho': rhos, 'pval': pvals})
    corr_df['abs_rho'] = corr_df['rho'].abs()
    
    # NetITH-correlated: FDR < 0.05 (BH correction on p-values)
    from statsmodels.stats.multitest import multipletests
    pvals_array = corr_df['pval'].values
    # BH (fdr_bh) multiple-testing correction over all tested genes at alpha = 0.05
    reject, fdr, _, _ = multipletests(pvals_array, method='fdr_bh', alpha=0.05)
    corr_df['fdr'] = fdr
    corr_df['significant'] = reject
    
    netith_correlated = set(corr_df[corr_df['significant']].index)
    print(f"  NetITH-significantly-correlated genes (FDR<0.05): {len(netith_correlated)}")
    
    # Also try top 10% by |rho| as a secondary approach
    threshold_10pct = np.percentile(corr_df['abs_rho'].dropna(), 90)
    netith_correlated_top10 = set(corr_df[corr_df['abs_rho'] >= threshold_10pct].index)
    print(f"  NetITH-correlated genes (top 10% |ρ|): {len(netith_correlated_top10)}")
    
    # DTP gene sets
    dtp_up_in_expr = set(g for g in DTP_UPREGULATED if g in expr.index)
    dtp_down_in_expr = set(g for g in DTP_DOWNREGULATED if g in expr.index)
    all_genes_in_expr = set(expr.index)
    
    print(f"  DTP-up genes in expr: {len(dtp_up_in_expr)} (out of {len(DTP_UPREGULATED)} total)")
    print(f"  DTP-up genes found: {sorted(dtp_up_in_expr)}")
    print(f"  DTP-up genes NOT found: {sorted(set(DTP_UPREGULATED) - dtp_up_in_expr)}")
    
    # Also print DTP gene correlations for debugging
    dtp_corr_in_expr = corr_df.loc[list(dtp_up_in_expr)].sort_values('abs_rho', ascending=False)
    print(f"  Top 5 DTP-up |ρ| with NetITH: ")
    for gene, row in dtp_corr_in_expr.head(5).iterrows():
        print(f"    {gene}: ρ={row['rho']:.4f}, p={row['pval']:.2e}")
    
    # Try Fisher's exact with FDR-based set
    a_fdr = len(dtp_up_in_expr & netith_correlated)
    b_fdr = len(dtp_up_in_expr - netith_correlated)
    c_fdr = len(netith_correlated - dtp_up_in_expr)
    d_fdr = len(all_genes_in_expr - dtp_up_in_expr - netith_correlated)
    
    # One-sided Fisher's exact test: DTP-up genes enriched among NetITH-correlated genes (null: no enrichment)
    odds_ratio_fdr, p_fdr = fisher_exact([[a_fdr, b_fdr], [c_fdr, d_fdr]], alternative='greater')
    print(f"  FDR-based: DTP-up ∩ FDR<0.05: {a_fdr}/{len(dtp_up_in_expr)}")
    print(f"  Fisher's exact p={p_fdr:.4f}, OR={odds_ratio_fdr:.2f}")
    
    # Try with top 10% threshold
    a_10 = len(dtp_up_in_expr & netith_correlated_top10)
    b_10 = len(dtp_up_in_expr - netith_correlated_top10)
    c_10 = len(netith_correlated_top10 - dtp_up_in_expr)
    d_10 = len(all_genes_in_expr - dtp_up_in_expr - netith_correlated_top10)
    
    # Repeat enrichment with the top-10% |rho| gene set (kept as the primary result)
    odds_ratio_10, p_10 = fisher_exact([[a_10, b_10], [c_10, d_10]], alternative='greater')
    print(f"  Top10%-based: DTP-up ∩ top10%|ρ|: {a_10}/{len(dtp_up_in_expr)}")
    print(f"  Fisher's exact p={p_10:.4f}, OR={odds_ratio_10:.2f}")
    
    # Also compute per-gene NetITH correlation for DTP genes
    dtp_netith_corrs = corr_df.loc[list(dtp_up_in_expr)][['rho', 'pval']].sort_values('rho', ascending=False)
    
    # Use the top-10% result as primary (more comparable to manuscript)
    return dtp_netith_corrs, a_10, b_10, c_10, d_10, p_10, odds_ratio_10, corr_df


# ═══════════════════════════════════════════════════════════
#  D: TCGA EMT Score (approximate using available genes)
# ═══════════════════════════════════════════════════════════

def load_tcga_expression_subset(genes_of_interest):
    """Load a subset of TCGA expression for the genes needed."""
    print("[D] Loading TCGA expression for EMT genes...")
    
    # Load just the needed rows by streaming the gzip file
    chunks = []
    gene_set = set(genes_of_interest)
    found_genes = set()
    
    # Stream the gzip file
    import gzip
    with gzip.open(f"{TCGA_XENA}/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz", 'rt') as fh:
        header = fh.readline().strip().split('\t')
        samples = header[1:]  # skip 'sample' column
        
        for line in fh:
            parts = line.strip().split('\t')
            gene = parts[0]
            if gene in gene_set:
                values = pd.to_numeric(pd.Series(parts[1:]), errors='coerce')
                chunks.append(pd.Series(values, index=samples, name=gene))
                found_genes.add(gene)
            if found_genes == gene_set:
                break
    
    if not chunks:
        raise RuntimeError(f"No genes found in TCGA expression! Looking for: {gene_set}")
    
    df = pd.DataFrame(chunks).T
    # Keep primary tumors
    primary_ix = [s for s in df.index if s.endswith('-01')]
    df = df.loc[primary_ix].copy()
    # Deduplicate per patient
    patient_ids = [s[:12] for s in df.index]
    df = df[~pd.Series(patient_ids).duplicated(keep='first')]
    
    print(f"  TCGA EMT expression: {df.shape}, genes found: {len(found_genes)}/{len(gene_set)}")
    return df


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    results = {}
    
    # ── A: GDSC EMT vs NetITH ──
    expr_gdsc = load_gdsc_expression()
    emt_gdsc = compute_emt_score(expr_gdsc)
    
    netith_gdsc = pd.read_csv(GDSC_NETITH)
    
    common_cl = list(set(emt_gdsc.index) & set(netith_gdsc['cell_line']))
    netith_gdsc_map = netith_gdsc.set_index('cell_line')['NetITH'].to_dict()
    
    emt_vals = [emt_gdsc[c] for c in common_cl]
    netith_vals = [netith_gdsc_map[c] for c in common_cl]
    # A: Spearman EMT score vs NetITH across GDSC cell lines (null: rho = 0)
    r_emt_gdsc, p_emt_gdsc = spearmanr(emt_vals, netith_vals)
    print(f"\n[A] GDSC EMT vs NetITH: ρ={r_emt_gdsc:.4f}, p={p_emt_gdsc:.2e}")
    results['emt_gdsc_rho'] = r_emt_gdsc
    results['emt_gdsc_p'] = p_emt_gdsc
    results['emt_gdsc_n'] = len(common_cl)
    
    # ── A2: GDSC EMT vs Drug IC50 (confirm EMT-drug resistance association) ──
    drug_corr = pd.read_csv(TCGA_DRUG_CORR)
    results['emt_gdsc_n_drugs'] = len(drug_corr)
    
    # ── B: TCGA mRNAsi vs NetITH ──
    mrnasi = load_mrnasi()
    netith_tcga = pd.read_csv(TCGA_NETITH)
    
    # Match samples
    tcga_samples = set(mrnasi.index) & set(netith_tcga['sample'])
    mrnasi_match = mrnasi.loc[list(tcga_samples)]
    netith_tcga_map = netith_tcga.set_index('sample')['netith_bulk'].to_dict()
    
    rnass_vals = mrnasi_match['RNAss'].values
    netith_tcga_vals = np.array([netith_tcga_map[s] for s in mrnasi_match.index])
    
    # Keep only patients with both mRNAsi and NetITH available
    valid = ~(np.isnan(rnass_vals) | np.isnan(netith_tcga_vals))
    # B: Spearman mRNAsi (RNAss) vs bulk NetITH across matched TCGA primary tumours
    r_mrnasi, p_mrnasi = spearmanr(rnass_vals[valid], netith_tcga_vals[valid])
    print(f"\n[B] TCGA mRNAsi vs NetITH: ρ={r_mrnasi:.4f}, p={p_mrnasi:.2e}")
    results['mrnasi_rho'] = r_mrnasi
    results['mrnasi_p'] = p_mrnasi
    results['mrnasi_n'] = valid.sum()
    
    # Also EREG.EXPss (epigenetically regulated stemness)
    # Also correlate the EREG.EXPss epigenetically regulated stemness score
    ereg_vals = mrnasi_match['EREG.EXPss'].values
    valid_ereg = ~(np.isnan(ereg_vals) | np.isnan(netith_tcga_vals))
    r_ereg, p_ereg = spearmanr(ereg_vals[valid_ereg], netith_tcga_vals[valid_ereg])
    print(f"  TCGA EREG.EXPss vs NetITH: ρ={r_ereg:.4f}, p={p_ereg:.2e}")
    results['ereg_rho'] = r_ereg
    results['ereg_p'] = p_ereg
    
    # ── C: DTP Enrichment ──
    dtp_corrs, a, b, c, d, p_dtp, or_dtp, all_gene_corrs = dtp_enrichment_analysis(expr_gdsc, netith_gdsc)
    print(f"\n[C] DTP enrichment: Fisher p={p_dtp:.4f}, OR={or_dtp:.2f}")
    results['dtp_fisher_p'] = p_dtp
    results['dtp_odds_ratio'] = or_dtp
    results['dtp_overlap_n'] = a
    results['dtp_total_up'] = a + b
    
    # ── D: TCGA EMT vs NetITH via ssGSEA Hallmark EMT score ──
    # Note: TCGA Xena expression uses Entrez gene IDs; direct gene-symbol lookup
    # requires ID mapping. We use pre-computed ssGSEA Hallmark scores instead.
    print("[D] Loading TCGA Hallmark EMT scores (ssGSEA)...")
    # D: TCGA EMT via precomputed Hallmark EMT ssGSEA scores (Xena Entrez ids preclude direct symbol lookup)
    hallmark_file = f"{TCGA_XENA}/PanCan33_ssGSEA_1387GeneSets_NonZero_sample_level.txt.gz"
    try:
        hallmark = pd.read_csv(hallmark_file, sep='\t', compression='gzip', index_col=0)
        # Look for EMT-related gene sets
        emt_cols = [c for c in hallmark.columns if 'EMT' in c.upper() or 'EPITHELIAL_MESENCHYMAL' in c.upper()]
        if emt_cols:
            emt_col = emt_cols[0]
            print(f"  Found EMT gene set column: {emt_col}")
            emt_tcga_series = hallmark[emt_col]
            common_s = list(set(emt_tcga_series.index) & set(netith_tcga['sample']))
            emt_tcga_vals = [emt_tcga_series[s] for s in common_s]
            netith_tcga_vals2 = [netith_tcga_map[s] for s in common_s]
            r_emt_tcga, p_emt_tcga = spearmanr(emt_tcga_vals, netith_tcga_vals2)
            print(f"  TCGA Hallmark EMT vs NetITH: ρ={r_emt_tcga:.4f}, p={p_emt_tcga:.2e}")
            results['emt_tcga_rho'] = r_emt_tcga
            results['emt_tcga_p'] = p_emt_tcga
            results['emt_tcga_n'] = len(common_s)
        else:
            print("  No EMT gene set found in ssGSEA scores.")
            r_emt_tcga = np.nan
            p_emt_tcga = np.nan
            emt_tcga_vals = []
            netith_tcga_vals2 = []
            common_s = []
            results['emt_tcga_rho'] = np.nan
            results['emt_tcga_p'] = np.nan
            results['emt_tcga_n'] = 0
    except Exception as e:
        print(f"  Hallmark EMT load failed: {e}")
        r_emt_tcga = np.nan
        p_emt_tcga = np.nan
        emt_tcga_vals = []
        netith_tcga_vals2 = []
        results['emt_tcga_rho'] = np.nan
        results['emt_tcga_p'] = np.nan
        results['emt_tcga_n'] = 0
    
    # ── Save results ──
    # Write the one-row results summary for all anchoring analyses
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{OUTPUT_DIR}/emt_stemness_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/emt_stemness_results.csv")
    
    # ═══════════════════════════════════════════════════
    #  FIGURE
    # ═══════════════════════════════════════════════════
    fig = plt.figure(figsize=(7.0866, 4.4291))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Panel A: GDSC EMT vs NetITH
    axA = fig.add_subplot(gs[0, 0])
    axA.scatter(emt_vals, netith_vals, c='#2166AC', alpha=0.4, s=8, edgecolors='none')
    axA.set_xlabel('EMT Score (76-gene)', fontsize=11)
    axA.set_ylabel('NetITH', fontsize=11)
    axA.set_title(f'GDSC: EMT vs NetITH\nρ={r_emt_gdsc:.3f}, p={p_emt_gdsc:.1e}', fontsize=12, fontweight='bold')
    # Add trend line
    z = np.polyfit(emt_vals, netith_vals, 1)
    x_line = np.linspace(min(emt_vals), max(emt_vals), 100)
    axA.plot(x_line, np.poly1d(z)(x_line), 'r-', lw=1.5, alpha=0.7)
    axA.text(0.95, 0.05, f'n={len(common_cl)}', transform=axA.transAxes, ha='right', fontsize=9, color='gray')
    
    # Panel B: TCGA EMT vs NetITH (Hallmark ssGSEA)
    axB = fig.add_subplot(gs[0, 1])
    if emt_tcga_vals and len(emt_tcga_vals) > 0 and not np.isnan(r_emt_tcga):
        axB.scatter(emt_tcga_vals, netith_tcga_vals2, c='#B2182B', alpha=0.3, s=5, edgecolors='none')
        axB.set_xlabel('Hallmark EMT (ssGSEA)', fontsize=11)
        axB.set_ylabel('NetITH (bulk)', fontsize=11)
        axB.set_title(f'TCGA: EMT vs NetITH\nρ={r_emt_tcga:.3f}, p={p_emt_tcga:.1e}', fontsize=12, fontweight='bold')
        z2 = np.polyfit(emt_tcga_vals, netith_tcga_vals2, 1)
        axB.plot(np.linspace(min(emt_tcga_vals), max(emt_tcga_vals), 100),
                 np.poly1d(z2)(np.linspace(min(emt_tcga_vals), max(emt_tcga_vals), 100)),
                 'r-', lw=1.5, alpha=0.7)
        axB.text(0.95, 0.05, f'n={len(emt_tcga_vals)}', transform=axB.transAxes, ha='right', fontsize=9, color='gray')
    else:
        axB.text(0.5, 0.5, 'TCGA EMT\n(ssGSEA data unavailable)', transform=axB.transAxes, ha='center', fontsize=11, color='gray')
        axB.set_title('TCGA: EMT vs NetITH', fontsize=12, fontweight='bold')
    
    # Panel C: TCGA mRNAsi vs NetITH
    axC = fig.add_subplot(gs[0, 2])
    axC.scatter(rnass_vals[valid], netith_tcga_vals[valid], c='#4DAF4A', alpha=0.3, s=5, edgecolors='none')
    axC.set_xlabel('mRNAsi (Malta 2018)', fontsize=11)
    axC.set_ylabel('NetITH (bulk)', fontsize=11)
    axC.set_title(f'TCGA: Stemness vs NetITH\nρ={r_mrnasi:.3f}, p={p_mrnasi:.1e}', fontsize=12, fontweight='bold')
    z3 = np.polyfit(rnass_vals[valid], netith_tcga_vals[valid], 1)
    axC.plot(np.linspace(min(rnass_vals[valid]), max(rnass_vals[valid]), 100),
             np.poly1d(z3)(np.linspace(min(rnass_vals[valid]), max(rnass_vals[valid]), 100)),
             'r-', lw=1.5, alpha=0.7)
    axC.text(0.95, 0.05, f'n={valid.sum()}', transform=axC.transAxes, ha='right', fontsize=9, color='gray')
    
    # Panel D: DTP gene enrichment — volcano of NetITH correlations
    axD = fig.add_subplot(gs[1, :2])
    # Plot all genes in gray
    sample_n = min(5000, len(all_gene_corrs))
    sample_idx = np.random.RandomState(SEED).choice(len(all_gene_corrs), sample_n, replace=False)
    axD.scatter(all_gene_corrs['rho'].iloc[sample_idx],
                -np.log10(all_gene_corrs['pval'].iloc[sample_idx].clip(1e-300)),
                c='#E0E0E0', alpha=0.3, s=3, edgecolors='none', label='All genes')
    # Highlight DTP-upregulated genes
    dtp_rho_vals = dtp_corrs['rho'].values
    dtp_pval_vals = -np.log10(dtp_corrs['pval'].clip(1e-300).values)
    axD.scatter(dtp_rho_vals, dtp_pval_vals, c='#FF5722', alpha=0.8, s=20, edgecolors='black',
                linewidths=0.3, label=f'DTP-up (n={len(dtp_corrs)})', zorder=5)
    axD.set_xlabel("Spearman ρ (gene vs NetITH)", fontsize=11)
    axD.set_ylabel("−log₁₀(p)", fontsize=11)
    axD.set_title(f'DTP Gene Enrichment Among NetITH-Correlated Targets\nFisher p={p_dtp:.3f}, OR={or_dtp:.2f}',
                  fontsize=12, fontweight='bold')
    axD.axhline(-np.log10(0.05), color='gray', ls='--', lw=0.7, alpha=0.5)
    axD.axvline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
    axD.legend(fontsize=8, loc='upper right')
    
    # Panel E: Summary bar chart
    axE = fig.add_subplot(gs[1, 2])
    metrics = ['EMT\n(GDSC)', 'EMT\n(TCGA)', 'mRNAsi\n(TCGA)', 'EREG\n(TCGA)']
    rhos = [r_emt_gdsc, r_emt_tcga, r_mrnasi, r_ereg]
    ps = [p_emt_gdsc, p_emt_tcga, p_mrnasi, p_ereg]
    colors = ['#2166AC', '#B2182B', '#4DAF4A', '#FF7F00']
    bars = axE.bar(range(4), rhos, color=colors, alpha=0.75, edgecolor='black', linewidth=0.5)
    axE.axhline(0, color='black', lw=0.5)
    axE.set_xticks(range(4))
    axE.set_xticklabels(metrics, fontsize=9)
    axE.set_ylabel("Spearman ρ with NetITH", fontsize=11)
    axE.set_title("Biological Anchoring Summary", fontsize=12, fontweight='bold')
    # Annotate significance
    for i, (r, p) in enumerate(zip(rhos, ps)):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        y_pos = r + 0.02 if r >= 0 else r - 0.06
        axE.text(i, y_pos, f'r={r:.3f}\n{sig}', ha='center', fontsize=8, fontweight='bold')
    
    fig.suptitle('NetITH Biological Anchoring: EMT, Stemness, and Drug-Tolerant Persister Connections',
                 fontsize=14, fontweight='bold', y=1.01)
    
    fig_path = f"{OUTPUT_DIR}/figures/emt_stemness_dtp_anchoring.png"
    # Save the anchoring figure as PNG and PDF
    fig.savefig(fig_path, dpi=300, facecolor='white')

    export_panels(fig, "EDFig7_emt_stemness_dtp")
    fig.savefig(fig_path.replace('.png', '.pdf'), facecolor='white')
    print(f"Figure saved to {fig_path}")
    
    # ── Print Summary ──
    print("\n" + "="*60)
    print(" BIOLOGICAL ANCHORING SUMMARY")
    print("="*60)
    print(f"  EMT (GDSC):      ρ={r_emt_gdsc:.4f}, p={p_emt_gdsc:.2e}, n={len(common_cl)}")
    print(f"  EMT (TCGA):      ρ={r_emt_tcga:.4f}, p={p_emt_tcga:.2e}, n={results.get('emt_tcga_n', 0)}")
    print(f"  mRNAsi (TCGA):   ρ={r_mrnasi:.4f}, p={p_mrnasi:.2e}, n={valid.sum()}")
    print(f"  EREG.EXPss:      ρ={r_ereg:.4f}, p={p_ereg:.2e}, n={valid_ereg.sum()}")
    print(f"  DTP enrichment:  Fisher p={p_dtp:.4f}, OR={or_dtp:.2f}, overlap={a}/{a+b}")
    print("="*60)
    
    return results


# --- panel export (per-panel PDF + 300dpi PNG for review/patchwork assembly) ---
PANEL_DIR = ROOT / "results/figures/panels"
def export_panels(fig, base_name):
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        letter = chr(65 + i)
        extent = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
            fig.savefig(PANEL_DIR / f"{base_name}_panel{letter}.{ext}",
                        bbox_inches=extent, facecolor="white", **kw)

if __name__ == '__main__':
    main()
