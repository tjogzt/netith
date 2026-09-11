#!/bin/bash
# ─────────────────────────────────────────────────────────────
# NetEntropy: Data Download Script
# Downloads scRNA-seq data from TISCH2, TCGA, and motif databases
# ─────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
RAW_DIR="${DATA_DIR}/raw"
EXTERNAL_DIR="${DATA_DIR}/external"

echo "========================================"
echo "NetEntropy Data Download"
echo "Project: ${PROJECT_DIR}"
echo "========================================"

mkdir -p "${RAW_DIR}" "${EXTERNAL_DIR}/cistarget" "${EXTERNAL_DIR}/tcga"

# ─── 1. TISCH2 scRNA-seq data ────────────────────────────────
echo ""
echo "[1/4] TISCH2 scRNA-seq data..."
echo "  TISCH2 provides pre-processed pan-cancer scRNA-seq data."
echo "  Manual download required from: http://tisch.comp-genomics.org/"
echo "  → Navigate to 'Download' → Select cancer types → Download h5ad files"
echo "  → Place downloaded files in: ${RAW_DIR}/"
echo ""
echo "  Recommended cancer types:"
echo "    - LUAD (Lung Adenocarcinoma)"
echo "    - SKCM (Skin Cutaneous Melanoma)"
echo "    - GBM (Glioblastoma Multiforme)"
echo "    - BRCA (Breast Invasive Carcinoma)"
echo "    - COAD (Colon Adenocarcinoma)"

# ─── 2. TCGA data (via cBioPortal or GDC) ────────────────────
echo ""
echo "[2/4] TCGA data..."
echo "  Bulk RNA-seq and clinical data from TCGA."
echo "  Options:"
echo "    a) cBioPortal: https://www.cbioportal.org/"
echo "       → Query by cancer type → Download 'mRNA Expression' and 'Clinical Data'"
echo "    b) GDC Data Portal: https://portal.gdc.cancer.gov/"
echo "       → Use GDC Data Transfer Tool for batch download"
echo "    c) UCSC Xena: https://xenabrowser.net/"
echo "       → Pre-processed TCGA data, convenient for analysis"
echo ""
echo "  Place downloaded files in: ${EXTERNAL_DIR}/tcga/"

# ─── 3. SCENIC motif database ────────────────────────────────
echo ""
echo "[3/4] SCENIC motif databases..."
echo "  Downloading cisTarget motif rankings for hg38..."

CISTARGET_DIR="${EXTERNAL_DIR}/cistarget"

# Human motif rankings (hg38) - feather format for pySCENIC
MOTIF_URL="https://resources.aertslab.org/cistarget/motif2tf/motifs-v10nr_clust-hg38.feat"

if [ ! -f "${CISTARGET_DIR}/motifs-v10nr_clust-hg38.feat" ]; then
    echo "  Downloading motif annotations..."
    wget -P "${CISTARGET_DIR}" "${MOTIF_URL}" || {
        echo "  WARNING: wget failed. Please download manually from:"
        echo "  https://resources.aertslab.org/cistarget/"
    }
fi

# TF list
TF_LIST_URL="https://resources.aertslab.org/cistarget/tf_lists/allTFs_hg38.txt"
if [ ! -f "${CISTARGET_DIR}/allTFs_hg38.txt" ]; then
    echo "  Downloading TF list..."
    wget -P "${CISTARGET_DIR}" "${TF_LIST_URL}" || true
fi

echo "  cisTarget database ready."

# ─── 4. Drug target databases ─────────────────────────────────
echo ""
echo "[4/4] Supporting databases..."
echo "  DrugBank: https://go.drugbank.com/releases/latest (registration required)"
echo "  DGIdb: https://www.dgidb.org/downloads"
echo "  DepMap: https://depmap.org/portal/download/"

# ─── Summary ──────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Download Summary"
echo "========================================"
echo ""
echo "Required files:"
echo "  ✅ cisTarget motif database: ${CISTARGET_DIR}/"
echo "  ⬜ TISCH2 h5ad files: ${RAW_DIR}/ (manual download)"
echo "  ⬜ TCGA data: ${EXTERNAL_DIR}/tcga/ (manual download)"
echo "  ⬜ Drug databases: ${EXTERNAL_DIR}/ (optional)"
echo ""
echo "After downloading all data, run preprocessing:"
echo "  python scripts/01_preprocess_data.py --cancer LUAD"
echo ""
