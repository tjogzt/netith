# NetITH R Analysis Layer

## Purpose

The `code/R/` layer implements the NetITH analysis pipeline in R (authoritative
implementation for all manuscript figures and core statistics). A Python
pipeline (`scripts/`, repo root) serves as the independent numerical
cross-validation reference; the two implementations agree to ρ ≥ 0.999999 on
core entropy values and ρ = 0.999993 on drug associations (see
`results/RERUN_VERIFICATION_REPORT.md`).

## Layout (three-tier: data/ · code/ · results/)

- `data/` — raw and intermediate inputs (CollecTRI network, GDSC expression,
  cached results); large raw files live on an external data disk mounted at
  `$NETITH_DATA_DISK` (default `/Volumes/tjogzt4T`), never in the repository.
- `code/R/` — this layer: numbered analysis scripts executed in order.
- `results/` — all analysis outputs, including `results/figures/r/` for
  R-generated figures.

## Script inventory (execution order)

| File | Content | Status |
|---|---|---|
| `00_global_config.R` | Global config: Chinese-palette colours, 180-mm canvas constants, `theme_pub`, `save_fig`/`save_panels`, shared helpers, `SEED = 49L` | ✓ |
| `01_data_import.R` | Data import: CollecTRI, GDSC expression (ENSG→symbol, CEL→cell line), cached results | ✓ |
| `02_netith_core.R` | **NetITH core computation** (von Neumann entropy, L = D − A) | ✓ |
| `03_gdsc_drug_sensitivity.R` | GDSC2 drug associations: 286 compounds, Spearman + BH-FDR | ✓ |
| `04_tcga_validation.R` | TCGA survival validation (29 cancer types, Cox PH) | ✓ |
| `05_focused_mad_baseline.R` | Focused 239-gene expression-MAD scalar baseline | ✓ |
| `06_gse25066_fixed_network.R` | GSE25066 neoadjuvant pCR, fixed-network construction | ✓ |
| `07_tcga_jun_fixed_network.R` | JUN edge-weight contribution (GDSC + TCGA) | ✓ |
| `08_signed_permutation_null.R` | Signed-edge permutation null (sign channel attribution) | ✓ |
| `09_auc_sensitivity.R` | AUC-readout sensitivity of the GDSC drug association | ✓ |
| `10_zclip_sensitivity.R` | Z-clipping robustness of the entropy construction | ✓ |
| `11_amplitude_equalized_null.R` | Amplitude-matched null (wiring vs. amplitude channels) | ✓ |
| `12_prism_root_cause.R` | PRISM cross-screen attribution (readout factor decomposition) | ✓ |
| `13_gse120575_os_cox.R` | GSE120575 OS Cox companion (reads the CSV written by the merged `11_gse120575_replication.py --os` stage) | ✓ |
| `14_xue2022_hcc_convert.R` | Xue2022 HCC Seurat RDS → expression cache conversion | ✓ |
| `15_recompute_abs_weight.R` | \|w\| vs. signed-w NetITH recomputation (GDSC, 286 drugs) | ✓ |
| `16_recompute_tcga_meta.R` | TCGA meta-analysis recomputation | ✓ |
| `run_all.R` | Master runner: full chain (R + Python stages + figures) with per-step logging to `results/pipeline_run_log.csv`; failures do not abort remaining steps | ✓ |
| `figures/fig01_definition.R` | Fig. 1: definition, distributions, tumour-vs-normal, null contrast (4 panels) | ✓ |
| `figures/fig02_drug_response.R` | Fig. 2: survival forest, GDSC drugs, GSE25066, IMvigor210 (4 panels) | ✓ |
| `figures/fig03_robustness.R` | Fig. 3: robustness, construct validity, cross-platform drivers | ✓ |
| `figures/fig04_ner_hypothesis.R` | Fig. 4: NER workflow as hypothesis generation | ✓ |
| `figures/figED1–figED10_*.R` | Extended Data Figures 1–10 | ✓ |
| `figures/figS1–figS6_*.R` | Supplementary Figures 1–6 (incl. descriptor census, Fig. S6) | ✓ |

## Figure outputs

- Combined figures: `results/figures/r/<name>.pdf` (vector) + `.png` (300 dpi)
- Single panels: `results/figures/r/panels/<name>_panel<letter>.pdf/.png`
  (for inspection and patchwork reassembly)

## Numerical consistency

Every statistic plotted in a figure is read from cached CSV/JSON under
`results/` (single-source rule); figure scripts never recompute manuscript
numbers ad hoc. Cross-implementation agreement (R vs. Python) is recorded in
`results/RERUN_VERIFICATION_REPORT.md` and `results/REVISION_CHECKLIST.md`.

## Random-seed policy (R = 49, Python = 42)

The R implementation and the Python pipeline intentionally use different
global seeds: R uses 49 (`00_global_config.R`, `SEED <- 49L`), Python uses 42
(module-level `SEED = 42`). This difference only affects stages involving
random subsampling or jittered plotting (e.g. the 30,000-cell Xenium stratified
subsample); it does NOT affect core entropy values — the von Neumann entropy is
a deterministic function of the expression matrix and edge set — hence the
ρ = 0.999999 agreement. Seed disclosure in the manuscript (Online Methods,
Statistical Analysis): "Random seeds: 42 (Python pipeline), 49 (R figures)."
