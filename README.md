# NetITH — a spectral-entropy descriptor of transcription-factor networks

**NetITH: a spectral-entropy descriptor of transcription-factor networks with context-dependent, partly transferable clinical associations**

Hu T, Mo Q, Chen P, Xu C, Wang Y, Sun Q, Zhu T
Department of Obstetrics and Gynecology, National Clinical Research Center for Obstetrics and Gynecology, Tongji Hospital, Tongji Medical College, Huazhong University of Science and Technology, Wuhan, China

This repository is the complete, runnable replication package for the NetITH manuscript (target: *Briefings in Bioinformatics*; submission in preparation). All manuscript figures, extended-data figures, supplementary figures and tables can be reproduced end-to-end from the scripts and data below (~3 h on a standard workstation).

## What NetITH measures

NetITH is the von Neumann entropy of the non-normalised Laplacian **L = D − A** of a sample-specific transcription-factor–target graph built on the curated signed CollecTRI network (edge weight *w*·|*z*(TF)|·|*z*(target)|, *w* ∈ {−1,+1}, z-scored expression clipped to [−3, 3]). It quantifies how "mixed" the regulatory wiring state of a sample is, and is reported as a reference-cohort-normalised score (bulk), a per-cell construction, a patient-level Shannon aggregate (NetITH_E), a co-expression spatial variant, and a phosphoproteomic variant.

## Key findings

- **Drug sensitivity (GDSC, 1,013 cell lines, 286 compounds):** higher NetITH associates with higher ln(IC50) for all 286 drugs (median Spearman ρ = 0.232; 276/286 FDR < 0.05; 100% directional consistency). The association is ln(IC50)-specific — it vanishes on the AUC readout (median ρ = 0.02) and does not replicate in the PRISM screen (median ρ = 0.002), a three-factor attribution locating most of the cross-screen collapse in the readout.
- **Construct controls pass:** a degree-preserving rewired null (50 draws) gives median ρ = 0.130 versus 0.232 (empirical p = 1/51); a signed-permutation null drops ρ to 0.016 (signs carry ≈58% of the signal); an amplitude-matched construction retains 0.158 (wiring carries ≈two-thirds). NetITH is orthogonal to five published tumour-purity estimates (|ρ| ≤ 0.093) and to a depth proxy.
- **Clinical associations are context-dependent:** TCGA pan-cancer survival is non-significant after RE pooling (HR 0.827, p = 0.121); GSE25066 neoadjuvant pCR OR = 0.86 per SD (p = 0.29); IMvigor210 anti-PD-L1 OR = 0.88 (p = 0.36). Small single-cell/spatial cohorts show partly transferable, resolution-dependent effects (descriptive).
- **Boundary results, reported honestly:** incremental prediction over lineage + expression features is small (ΔR² = 0.003); a scalar expression-MAD baseline matches NetITH's raw strength (ρ = 0.274); regulatory coefficients learned in GDSC attenuate in TCGA (JUN R² 0.28 → 0.126) — cell-line→tumour transfer is partial.
- **A four-test reporting protocol** (topology null · scalar baseline · purity/depth confounding · fixed-network transfer), calibrated on a seven-descriptor census (signaling entropy, differential network entropy, TF-activity variance, expression Shannon entropy, expression MAD, CytoTRACE proxy): NetITH is the only panel member whose topology-null pass is robust (empirical p = 1/51), the least purity-confounded, and the only one with 286/286 directional consistency.

## Repository layout

```
├── code/R/            R analysis chain (00–16) + run_all.R + figures/ (Fig 1–4, EDFig 1–10, Fig S1–S6)
├── scripts/
│   ├── 00_data/       Preprocessing (raw → per-project caches)
│   ├── 01_core/       Core pipeline (GDSC drugs, TCGA survival, multi-omics, meta-analysis, ...)
│   ├── 02_controls/   Construct controls, null models, descriptor census, robustness
│   ├── 03_drugs_ner/  CRISPR causality, TF mediation, perturbation, DrugComb, NER workflow
│   ├── 04_singlecell/ scRNA-seq analyses (GSE131907, EMT/stemness, pseudotime)
│   ├── 05_replications/ Independent cohort replications (GSE120575, GSE72056, GSE123139, GSE115978, Xue2022)
│   ├── 06_spatial/    Visium / Xenium spatial transcriptomics
│   ├── 07_clinical/   Neoadjuvant, nomogram, DCA, panel optimisation
│   └── 08_tests/      Consistency checks and regression tests
├── src/               Shared library (network_entropy, grn_inference, benchmarking, visualization)
├── tests/             Unit tests (von Neumann entropy, index invariance)
├── data/              Small version-stable inputs (CollecTRI snapshot, dataset registry, processed tables)
├── results/figures/   All 20 manuscript figures (vector PDF + PNG preview)
├── results/tables/    Supplementary Tables 1–26
├── docs/PIPELINE.md   Full pipeline order and execution guide
└── environment.yml    Locked conda environment (Python 3.12)
```

## Quickstart

```bash
# 1. Environment
conda env create -f environment.yml && conda activate netentropy

# 2. Data — two options:
#    (a) external data disk (recommended): export NETITH_DATA_ROOT=/path/to/data
#        with the same layout as data/README.md describes (GDSC2, Xena TCGA, DepMap 26Q1, GEO, ...)
#    (b) repository-local: place the downloaded files under data/ (default fallback)

# 3. Full R chain + replications + figures (~3 h):
Rscript code/R/run_all.R

# 4. Independent Python pipeline (numeric cross-validation of the R chain):
#    run each stage in order — see docs/PIPELINE.md

# 5. Census (descriptor panel): run benchmark_teschendorff_entropy.py before run_descriptor_census.py --stage gdsc

# 6. Tests:
python3 -m pytest tests/ -q
python3 scripts/08_tests/test_index_invariance.py
```

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `NETITH_DATA_ROOT` | Python pipeline data directory | `<repo>/data` |
| `NETITH_DATA_DISK` | External data-disk mount root (R side) | `/Volumes/tjogzt4T` |
| `NETITH_PYTHON` | Python interpreter used by `run_all.R` | `/opt/anaconda3/bin/python` |
| `NETITH_COLLECTRI_PKL` | CollecTRI pickle cache location | `/tmp/collectri_net.pkl` (auto-built from `data/collectri_network.csv` when absent) |
| `NETITH_GDSC_EXPR` / `_MAP` / `_ANNOT` | GDSC expression cache overrides (rna_expr.csv / ensg_symbol_map.csv / cell_annot.csv) | `<repo>/data/gdsc/*` |
| `NETITH_GDSC_DIR` | GDSC cache directory (Teschendorff benchmark) | `<repo>/data/gdsc` |
| `NETITH_GSE25066_MATRIX` / `NETITH_GPL96_PLATFORM` | GSE25066 series matrix / GPL96 platform (census stage) | `<repo>/data/external/gse25066/*` |
| `NETITH_COLLECTRI_CSV` | CollecTRI network CSV (census stage) | `<repo>/data/collectri_network.csv` |

## Numerical consistency

The R layer is the authoritative implementation for figures and manuscript statistics; the Python pipeline is an independent cross-validation reference. Agreement: NetITH values ρ = 0.999999 (GDSC median 5.6987), drug associations ρ = 0.999993 (median 0.232, 276 FDR). Random seeds are disclosed in the manuscript Online Methods: 42 (Python), 49 (R figures); core entropy values are deterministic and seed-independent.

## Citation

Hu T, Mo Q, Chen P, Xu C, Wang Y, Sun Q, Zhu T. NetITH: a spectral-entropy descriptor of transcription-factor networks with context-dependent, partly transferable clinical associations. *Briefings in Bioinformatics* (submission in preparation).

## License

MIT — see LICENSE.
