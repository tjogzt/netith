"""
verify_r_figures.py — verify all R-generated figures (width, font size, DPI).

Project : NetITH — spectral-entropy descriptor of transcription-factor networks
Author  : Tao Zhu, Tongji Hospital, Tongji Medical College, HUST
Created : 2026-09-06
Inputs  : results/figures/r/** (PDF/PNG outputs of the figure scripts)
Outputs : (none — console pass/fail table; exit 0 iff every figure passes)
Pipeline: replication stage — see repository README
"""
import fitz, os, sys
from collections import Counter
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
figdir = os.path.join(ROOT, "results", "figures", "r")
panels = os.path.join(figdir, "panels")
ok = total = 0
print(f"{'figure':52s} {'width':>7} {'min_font':>8} {'dpi':>5}")
for f in sorted(os.listdir(figdir)):
    if not f.endswith(".pdf") or f.startswith("._"): continue
    name = f[:-4]
    total += 1
    doc = fitz.open(os.path.join(figdir, f))
    w = doc[0].rect.width / 72 * 25.4
    sizes = Counter()
    for b in doc[0].get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for sp in l["spans"]: sizes[round(sp["size"], 1)] += len(sp["text"])
    doc.close()
    min_s = min(sizes) if sizes else 99
    png = os.path.join(figdir, name + ".png")
    dpi = 0
    if os.path.exists(png):
        im = Image.open(png); dpi = round(im.info.get("dpi", (0, 0))[0])
    # double-column (170-183 mm) or single-column (75-95 mm) are both valid
    w_ok = (w <= 183 and w >= 170) or (w <= 95 and w >= 75)
    f_ok = min_s >= 8
    d_ok = dpi >= 299 or (dpi == 0 and not os.path.exists(png))
    pass_ = w_ok and f_ok and d_ok
    ok += pass_
    print(f"{name:52s} {w:7.1f} {min_s:8.1f} {dpi:5d} {'OK' if pass_ else 'FAIL'}")
# panel files
n_panels = len([f for f in os.listdir(panels) if f.endswith('.pdf') and not f.startswith('._')]) if os.path.isdir(panels) else 0
print(f"\n{total} composite figures, {ok} pass; panel files: {n_panels}")
sys.exit(0 if ok == total else 1)
