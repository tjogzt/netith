# Data

Only small, version-stable inputs are committed to this repository:

| File | Content | Checksum |
|---|---|---|
| `collectri_network.csv` | CollecTRI signed TF→target network snapshot (42,990 interactions; 982 TFs in our copy) | see `results/submission/SHA256_MANIFEST.txt` |
| `dataset_registry.json` | Registry of all datasets with versions and download metadata | — |

All other data are public from their source consortia and **must be downloaded**
before running the pipeline (sizes in the tens of GB):

- **GDSC2**: expression + IC50 from the GDSC portal (`GDSC2_IC50_all.csv`);
- **TCGA pan-cancer**: Xena TOIL RSEM TPM matrix and MC3 v0.2.8 non-silent mutations;
- **DepMap 26Q1**: Chronos gene-effect matrix (18,530 genes × 1,208 cell lines;
  SHA-256 e610a4cefb13a82b5b256b47eb08b63ff14843f8dbd0fb164bc0a32688e5b89e);
- **DrugComb v1.4**: Zenodo DOI 10.5281/zenodo.11102665 (`summary_table_v1.4.csv`);
- **GSE25066 / GSE131907**: GEO;
- **IMvigor210**: NCT02108652 companion RNA-seq;
- **10x Visium (BRCA, OV) and Xenium (breast IDC)**: public 10x sections.

The exact file names, accessions, and download dates are recorded in
`results/submission/ENVIRONMENT.txt` and in the manuscript's Online Methods
(Data Provenance). Processed NetITH matrices are deposited on Zenodo
(DOI at submission) and mirrored in `results/`.

## Data layout convention

Scripts locate external data via `NETITH_DATA_ROOT` (environment variable,
default `<repo>/data`). With the default, place datasets directly under
`data/` exactly as named in the table above (e.g. `data/gdsc_download/GDSC2_IC50_all.csv`,
`data/xena/...`, `data/geo/GSE131907/...`). If your data live elsewhere, set
`NETITH_DATA_ROOT=/path/to/data/root` — the layout below that root must match
the paths used in the scripts (see `scripts/02_controls/run_tf_activity_baseline.py`
for the simplest example).
