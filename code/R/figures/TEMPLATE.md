# R 图脚本开发模板(必须遵守)

## 目标
把 NetITH 课题的 matplotlib 图用 R/ggplot2 重写,输出与论文数字完全一致、符合 SCI 规范的图。

## 每个脚本的固定文件头(复制,改 NAME)
```r
.d <- getwd()
while (!file.exists(file.path(.d, "data", "collectri_network.csv")) && nchar(.d) > 1) .d <- dirname(.d)
PROJECT_ROOT <- .d
CODE_DIR <- file.path(PROJECT_ROOT, "code", "R")
source(file.path(CODE_DIR, "00_global_config.R"))
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(patchwork) })
NAME <- "FigX_..."   # 输出文件名(不含扩展名)
```

## 可用 API(00_global_config.R 提供)
- 路径:RESULTS_DIR(项目 results/ 目录)
- 颜色:COL_NETITH=#4DBBD5, COL_GDSC=#E64B35, COL_TCGA=#00A087, COL_SIG=#3C5488,
  COL_NER=#F39B7F, COL_NS=#8491B4, COL_GREY=#E0E0E0, NPG_COLORS(10 色)
- 主题:theme_pub() —— 7pt 基础字号、0.3pt 轴线、无边框
- 保存:save_fig(composite, NAME, height_mm, width_mm=180) → results/figures/r/{NAME}.pdf/.png
  save_panels(panel_list, NAME) → results/figures/r/panels/{NAME}_panel{A,B,...}.pdf/.png
- 工具:spearman_test(x,y) 返回 c(rho,p,n);fmt_p(p);set.seed(49) 已设置

## 硬性规范(违反即返工)
1. **尺寸**:组合图宽度固定 180mm;高度 = Python 原图 figsize 高/7.0866×180mm(如 5.8360in → 148.2mm)。
2. **字号**:base_size=7(theme_pub 默认);panel 标题 8pt bold;任何文字不低于 7pt(物理字号)。
3. **配色**:只用 NPG_COLORS / COL_* 常量,与原图语义一致。
4. **图内文字全英文**;注释英文。
5. **数字一致性**:图中标注的所有统计量(rho、p、HR、OR、n、median、R²、百分比)必须与
   results/ 下的缓存 CSV/JSON 及论文数值一致,禁止凭空计算替代;脚本运行后打印关键数字。
6. **数据读取**:用 data.table::fread;CSV 若第一列是未命名索引列,先 netith[,1:=NULL] 或显式处理。
7. 组合图用 patchwork 拼接;pA、pB... 单独变量,最后 panels <- list(pA,pB,...) 传 save_panels。
8. 脚本运行:Rscript code/R/figures/figXX_xxx.R;输出后必须用 PyMuPDF 验证:
   python3 -c "import fitz; d=fitz.open('results/figures/r/NAME.pdf'); print(d[0].rect.width/72*25.4)" → ~180mm
   且所有 span size >= 7(用 Counter 检查)。
9. 原始 Python 图代码:scripts/07_figures/generate_manuscript_figures_v2.py(Fig1-4,FigS1,EDFig10)、
   scripts/archive/generate_manuscript_figures.py(EDFig1-5)、scripts/02_controls/benchmark_teschendorff_entropy.py(EDFig6)、
   scripts/04_singlecell/analyze_emt_stemness_dtp.py(EDFig7)、scripts/06_clinical/run_chemo_subset_analysis.py(EDFig8)、
   scripts/07_figures/generate_figS11_neoadjuvant.py(EDFig9)、scripts/archive/generate_figS14_sample_flow.py(FigS2)、
   scripts/07_figures/generate_tf_subset_figure.py(FigS3)
10. 若某 panel 是示意图(纯文字/箭头),用 ggplot annotate 实现,保持简洁。

## 完成标准
- 脚本 + PDF/PNG + panels 全部生成
- 180mm、字号≥7pt 验证通过
- 关键数字与缓存 CSV/JSON 交叉核对一致(脚本里打印)
- 输出最终报告:文件清单 + 每个 panel 的内容 + 验证结果
