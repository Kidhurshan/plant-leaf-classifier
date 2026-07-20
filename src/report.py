"""Report-asset generation: the centrepiece model-comparison table.

Writes ``summary_table.md`` + ``summary_table.csv`` and renders the table figure,
from a ``per_model`` metrics dict. Shared by ``scripts/evaluate_all.py`` and
``notebooks/05_compare_and_ensemble.ipynb`` so the notebook stays a thin driver.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

from src import viz
from src.utils import format_param_count, human_time

SUMMARY_COLUMNS = ["Model", "Accuracy", "Macro-F1", "Macro-P", "Macro-R",
                   "Acc+TTA", "F1+TTA", "Params", "Train time"]


def _fmt_row(key: str, m: Dict) -> Dict[str, str]:
    def num(x, spec="{:.4f}"):
        return "-" if (x is None or (isinstance(x, float) and math.isnan(x))) else spec.format(x)

    tt = m.get("train_time_s")
    tt_str = "-" if (tt is None or math.isnan(tt)) else human_time(tt)
    return {
        "Model": viz.display_name(key),
        "Accuracy": num(m["accuracy"]),
        "Macro-F1": num(m["macro_f1"]),
        "Macro-P": num(m["macro_precision"]),
        "Macro-R": num(m["macro_recall"]),
        "Acc+TTA": num(m.get("acc_tta")),
        "F1+TTA": num(m.get("f1_tta")),
        "Params": format_param_count(m["params"]),
        "Train time": tt_str,
    }


def build_summary_rows(per_model: Dict[str, Dict], order: List[str]) -> List[Dict[str, str]]:
    return [_fmt_row(k, per_model[k]) for k in order if k in per_model]


def write_summary_table(
    per_model: Dict[str, Dict], order: List[str],
    metrics_dir: str | Path, figures_dir: str | Path,
) -> Dict[str, Path]:
    """Write CSV + Markdown + a table figure. Returns the three paths."""
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = build_summary_rows(per_model, order)
    cols = SUMMARY_COLUMNS

    csv_path = metrics_dir / "summary_table.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    lines = ["# Model comparison — held-out test set", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    present = [k for k in order if k in per_model]
    best = max(present, key=lambda k: per_model[k]["macro_f1"])
    lines += ["", f"**Best by macro-F1:** {viz.display_name(best)} "
                  f"({per_model[best]['macro_f1']:.4f}).", ""]
    md_path = metrics_dir / "summary_table.md"
    md_path.write_text("\n".join(lines))

    png_path = figures_dir / "summary_table.png"
    viz.plot_comparison_table(rows, cols, out_path=png_path,
                              title="Model comparison (test set)")
    return {"csv": csv_path, "md": md_path, "png": png_path}
