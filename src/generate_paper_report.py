# -*- coding: utf-8 -*-
"""
Scientific Paper Summary Report Generator:
Consolidates evaluation outputs across Argo, WOCE, and custom sources
into publication-ready Markdown tables and LaTeX snippets.
"""
from pathlib import Path
import pandas as pd
from . import config


def generate_comparative_markdown_report(summary_csv_files: list, output_md: Path):
    """Combines individual source benchmark summaries into a unified report."""
    dfs = []
    for f in summary_csv_files:
        if Path(f).exists():
            dfs.append(pd.read_csv(f))

    if not dfs:
        print("[WARN] No summary CSVs found to compile report.")
        return

    full_summary = pd.concat(dfs, ignore_index=True)

    md_content = [
        "# Independent Validation & Cross-Domain Performance Report",
        "",
        "## 1. Summary of Experimental Results (Test Split 20%)",
        "",
        "| Evaluation Metric | " + " | ".join([f"**{s.upper()}**" for s in full_summary["source"]]) + " |",
        "|---| " + " | ".join(["---" for _ in full_summary["source"]]) + " |",
        "| **Test Samples** | " + " | ".join([f"{int(v):,}" for v in full_summary["test_samples"]]) + " |",
        "| **Ground Truth Anomalies** | " + " | ".join([f"{int(v):,}" for v in full_summary["anomalies_ground_truth"]]) + " |",
        "| **Calibrated Threshold $\\tau^*$** | " + " | ".join([f"${v:.4f}$" for v in full_summary["calibrated_threshold"]]) + " |",
        "| **ROC-AUC (95% CI)** | " + " | ".join([f"**${v:.4f}$** [${l:.4f} - {h:.4f}$]" for v, l, h in zip(full_summary["roc_auc"], full_summary["roc_auc_ci_low"], full_summary["roc_auc_ci_high"])]) + " |",
        "| **PR-AUC (95% CI)** | " + " | ".join([f"**${v:.4f}$** [${l:.4f} - {h:.4f}$]" for v, l, h in zip(full_summary["pr_auc"], full_summary["pr_auc_ci_low"], full_summary["pr_auc_ci_high"])]) + " |",
        "| **F1-Score** | " + " | ".join([f"**{v * 100:.2f}%**" for v in full_summary["f1_score"]]) + " |",
        "| **Recall (Sensitivity)** | " + " | ".join([f"**{v * 100:.2f}%**" for v in full_summary["recall"]]) + " |",
        "| **Precision** | " + " | ".join([f"**{v * 100:.2f}%**" for v in full_summary["precision"]]) + " |",
        "| **Specificity** | " + " | ".join([f"**{v * 100:.2f}%**" for v in full_summary["specificity"]]) + " |",
        "| **Matthews Corr. (MCC)** | " + " | ".join([f"**${v:.4f}$**" for v in full_summary["mcc"]]) + " |",
        "| **Confusion (TP / FP / TN / FN)** | " + " | ".join([f"{int(tp)} / {int(fp)} / {int(tn)} / {int(fn)}" for tp, fp, tn, fn in zip(full_summary["tp"], full_summary["fp"], full_summary["tn"], full_summary["fn"])]) + " |",
        "",
        "## 2. Methodology Highlights",
        "- **Zero Profile-Leakage**: Train/Val/Test partitioning performed strictly by `profile_id`/`station_id` (60:20:20).",
        "- **Semi-Supervised Fitting**: Model fit exclusively on validated clean observations ($QC \\le 2$).",
        "- **Physics-Embedded Domain Constraints**: Incorporates Mackenzie acoustic formulation, UNESCO density, and vertical gradients.",
        "- **Statistical Rigor**: All primary metrics reported with non-parametric Bootstrap 95% Confidence Intervals (1,000 iterations)."
    ]

    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    print(f"[SUCCESS] Exported comparative report -> {output_md}")


if __name__ == "__main__":
    summaries = list(config.OUTPUTS_DIR.glob("*/benchmark_metrics_summary.csv"))
    out_file = config.REPORTS_DIR / "final_comparative_paper_report.md"
    generate_comparative_markdown_report(summaries, out_file)
