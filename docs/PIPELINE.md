# NetITH pipeline — execution guide

## Pipeline order (dependency-respecting)

```
[0] scripts/00_data/01_preprocess_data.py     raw → project caches (CollecTRI, GDSC expr, TCGA, DepMap)
[1] code/R/run_all.R                          R chain 00→16 + replication scripts + all figures
      code/R/01_data_import.R                 caches → in-memory tables
      code/R/02_netith_core.R                 NetITH computation (von Neumann entropy)
      code/R/03_gdsc_drug_sensitivity.R       GDSC2 286-drug Spearman + BH-FDR
      code/R/04_tcga_validation.R             TCGA survival (29 cancer types)
      code/R/05_focused_mad_baseline.R        scalar MAD baseline (focused 239 genes)
      code/R/06_gse25066_fixed_network.R      neoadjuvant pCR, fixed construction
      code/R/07_tcga_jun_fixed_network.R      JUN edge-weight contribution
      code/R/08_signed_permutation_null.R     signed-edge permutation null
      code/R/09_auc_sensitivity.R             AUC-readout sensitivity
      code/R/10_zclip_sensitivity.R           z-clipping robustness
      code/R/11_amplitude_equalized_null.R    amplitude-matched null
      code/R/12_prism_root_cause.R            PRISM cross-screen attribution
      code/R/13_gse120575_os_cox.R            GSE120575 OS Cox (companion)
      code/R/14_xue2022_hcc_convert.R         Xue2022 RDS conversion (conditional)
      code/R/15_recompute_abs_weight.R        |w| vs signed-w recomputation
      code/R/16_recompute_tcga_meta.R         TCGA meta-analysis recomputation
[2] scripts/01_core/                          independent Python cross-validation of the core chain
      run_gdsc_drug_sensitivity.py            (R 03 counterpart)
      run_tcga_validation.py                  (R 04 counterpart)
      run_meta_analysis.py · run_module_conservation.py · run_gdsc_mutations.py
      run_gdsc_multiomics.py · run_tcga_methylation.py · run_tcga_survival_netith_vs_tmb.py
      run_imvigor210_validation.py · run_largescale_netith.py
      run_netith_threshold.py · run_crossplatform.py · run_crossplatform_calibration.py
      run_p08_drug_robustness.py · run_p1_robustness.py
[3] scripts/02_controls/                      construct controls + descriptor census
      run_control_random_graph_null.py · run_expression_permutation.py
      run_control_depth.py · run_control_mad_purity.py
      run_ith_benchmark.py                    ITH metric benchmark (NetITH vs 5 metrics)
      benchmark_teschendorff_entropy.py       signaling-entropy reference implementation
      run_descriptor_census.py                seven-descriptor × four-test census (stages: gdsc/null/tcga/gse)
      assemble_descriptor_census.py           census assembly (empirical p from draw arrays)
      run_control_netith_census_rewire_null.py   NetITH rewire null (follow-up A)
      run_null_activity_500.py                500-draw activity null (stabilised Test-1 verdict)
      run_tf_activity_baseline.py · run_tf_subset_50pct.py · run_depmap_26q1_differential.py
      run_prism_validation.py · run_mediation_bootstrap.py · run_nomogram_bootstrap.py
[4] scripts/03_drugs_ner/                     CRISPR causality, TF mediation, NER workflow
      run_crispr_dependency_netith.py · run_crispr_causality.py · run_depmap_crispr.py
      run_tf_mediation.py · run_jun_mediation.py · run_tf_combinatorial.py
      run_resistance_rewiring.py · run_drugcomb_cellline_validation.py
      run_drugcomb_degree_permutation.py · complete_perturbation_analysis.py
      ext_d1_causal_validation.py · ext_d5_perturb_seq.py · ext_d6_phospho_netith.py
[5] scripts/04_singlecell/                    scRNA-seq analyses
      run_gse131907_validation_v3.py · run_gse131907_tf_decomp.py · run_gse131907_pseudotime.py
      run_network_rewiring.py · run_coexpr_formulation_validation.py · analyze_emt_stemness_dtp.py
[6] scripts/05_replications/                  independent cohort replications (shared replication_common.py)
      01_scrnaseq_hardening.py · 02_spatial_autocorrelation.py
      03_gse120575_replication.py (--os stage) · 04_gse72056_replication.py
      05_gse123139_replication.py (--sensitivity stage) · 06_gse115978_replication.py
      07_xue2022_hcc_replication.py
[7] scripts/06_spatial/                       spatial transcriptomics
      run_spatial_visium_netith.py · run_xenium_spatial_netith.py
[8] scripts/07_clinical/                      clinical models
      run_neoadjuvant_validation.py · run_chemo_subset_analysis.py
      run_nomogram.py · run_dca_analysis.py · run_clinical_panel_optimization.py
[9] code/R/figures/                           all 20 figures (run via run_all.R or individually)
[10] scripts/08_tests/                        consistency + regression tests
```

## Dependency notes

- `results/focused_genes_collectri.txt` (the focused 239-gene CollecTRI set, SI-M1) is a
  small version-stable input shipped with this repository.
- `scripts/01_core/run_gdsc_drug_sensitivity.py` must run before
  `scripts/02_controls/run_descriptor_census.py --stage gdsc` (it writes
  `results/gdsc/gdsc_netith_cell_lines.csv`), and
  `scripts/02_controls/benchmark_teschendorff_entropy.py` must run before the same
  census stage (it writes `results/gdsc/teschendorff/signaling_entropy_gdsc.csv`).
- The CollecTRI pickle (`NETITH_COLLECTRI_PKL`) is auto-built from the shipped
  `data/collectri_network.csv` when absent.
- GDSC expression caches default to `<repo>/data/gdsc/` and can be redirected via
  `NETITH_GDSC_EXPR` / `NETITH_GDSC_MAP` / `NETITH_GDSC_ANNOT` / `NETITH_GDSC_DIR`
  (use this to reuse caches kept on an external disk).

## Runtime and resources

- Full chain: ~3 h on a standard workstation (external data disk recommended; tens of GB of public data).
- Null-model stages (census rewire null, 500-draw activity null) are the slowest single steps (~60 min and ~27 min respectively); every stage caches its outputs and skips already-computed results.
- `run_all.R` logs every step to `results/pipeline_run_log.csv` and does not abort the remaining steps on failure.

## Outputs → manuscript mapping

| Output | Manuscript item |
|---|---|
| `results/figures/r/Fig1–Fig4_*.pdf` | Main Figures 1–4 |
| `results/figures/r/EDFig1–EDFig10_*.pdf` | Extended Data Figures 1–10 |
| `results/figures/r/FigS1–FigS6_*.pdf` | Supplementary Figures 1–6 |
| `results/submission/tables/Supplementary_Table_01…26` | Supplementary Tables 1–26 (shipped under `results/tables/`) |
| `results/control/census/*` | Descriptor census (SI-N20, Fig. S6) |

Every figure reads its statistics from cached JSON/CSV single sources (no ad-hoc recomputation inside figure scripts); per-figure commands are listed in the repository README.
