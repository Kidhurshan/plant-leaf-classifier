"""ALL plotting for the project, in one consistent visual style.

Every figure uses a colourblind-safe palette (the data-viz reference palette),
readable fonts, titles + axis labels on everything, and is saved at 200 DPI PNG
so it can be pasted straight into the report. Functions save to ``out_path`` and
return the Matplotlib ``Figure`` so notebooks display it inline.

Categorical colour is assigned in a fixed order, never cycled. Models use slots
1-3 (blue/green/magenta), which pass the all-pairs CVD gate; species use the full
8-hue order (validated on the adjacent pairlist) with a legend as the required
secondary encoding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Palette + style (data-viz reference instance, light surface)                #
# --------------------------------------------------------------------------- #
SPECIES_PALETTE = [
    "#2a78d6",  # 1 blue
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
# Models use the all-pairs-safe first slots; ensemble gets slot 4.
MODEL_ORDER_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#4a3aa7"]

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"

MODEL_DISPLAY = {
    "efficientnetv2s": "EfficientNetV2-S",
    "swin_small": "Swin-Small",
    "cbam_convnext": "CBAM-ConvNeXt",
    "ensemble": "Ensemble",
}


def display_name(key: str) -> str:
    return MODEL_DISPLAY.get(key, key)


def model_color(index: int) -> str:
    return MODEL_ORDER_COLORS[index % len(MODEL_ORDER_COLORS)]


def set_style() -> None:
    """Apply the consistent project-wide Matplotlib style."""
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "figure.dpi": 110,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.edgecolor": SECONDARY,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": SECONDARY,
        "ytick.color": SECONDARY,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.autolayout": False,
    })


set_style()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _save(fig, out_path: Optional[str | Path]) -> Optional[Path]:
    if out_path is None:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def to_hwc_uint8(img) -> np.ndarray:
    """Convert a CHW/HWC torch tensor or numpy array to an HWC uint8 image.

    Handles float [0,1] and uint8; does NOT undo ImageNet normalisation (pass a
    denormalised image for that -- see :func:`src.augment.denormalize`).
    """
    try:
        import torch
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().float().numpy()
    except Exception:  # noqa: BLE001
        img = np.asarray(img)
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):   # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.0 + 1e-6:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


# --------------------------------------------------------------------------- #
# 1. Class distribution                                                       #
# --------------------------------------------------------------------------- #
def plot_class_distribution(class_counts: Dict[str, int],
                            out_path=None, title="Class distribution (EgyPLI)"):
    names = list(class_counts.keys())
    counts = [class_counts[n] for n in names]
    colors = [SPECIES_PALETTE[i % len(SPECIES_PALETTE)] for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, counts, color=colors, width=0.72,
                  edgecolor=SURFACE, linewidth=1.5)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.01,
                str(c), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_title(title)
    ax.set_xlabel("Species")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, max(counts) * 1.12)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# 2. Sample image grid (one per species)                                      #
# --------------------------------------------------------------------------- #
def plot_sample_grid(images, labels, class_names: Sequence[str],
                     out_path=None, per_class=1,
                     title="Sample images per species"):
    labels = np.asarray(labels)
    n_classes = len(class_names)
    cols = n_classes
    rows = per_class
    fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.9 * rows + 0.4))
    axes = np.atleast_2d(axes)
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        for r in range(rows):
            ax = axes[r, c]
            ax.axis("off")
            if r < len(idx):
                ax.imshow(to_hwc_uint8(images[idx[r]]))
            if r == 0:
                ax.set_title(class_names[c], fontsize=10,
                             color=SPECIES_PALETTE[c % len(SPECIES_PALETTE)])
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# 3. Augmentation preview (before vs after)                                   #
# --------------------------------------------------------------------------- #
def plot_augmentation_preview(before, after_list, out_path=None,
                              title="Augmentation preview (same image)"):
    n_after = len(after_list)
    fig, axes = plt.subplots(1, n_after + 1, figsize=(1.9 * (n_after + 1), 2.4))
    axes[0].imshow(to_hwc_uint8(before))
    axes[0].set_title("original", fontsize=10, color=INK)
    axes[0].axis("off")
    for i, img in enumerate(after_list):
        axes[i + 1].imshow(to_hwc_uint8(img))
        axes[i + 1].set_title(f"aug {i + 1}", fontsize=10, color=SECONDARY)
        axes[i + 1].axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# 4. Per-model training curves                                                #
# --------------------------------------------------------------------------- #
def plot_training_curves(history: List[dict], model_key: str, out_path=None):
    epochs = list(range(1, len(history) + 1))
    tr_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]
    color = model_color(0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(epochs, tr_loss, "-o", color=color, label="train", ms=4)
    axes[0].plot(epochs, val_loss, "--s", color="#eb6834", label="val", ms=4)
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_acc, "-o", color=color, ms=4)
    axes[1].set_title("Validation accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1.02)

    axes[2].plot(epochs, val_f1, "-o", color="#4a3aa7", ms=4)
    axes[2].set_title("Validation macro-F1"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1"); axes[2].set_ylim(0, 1.02)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Training curves — {display_name(model_key)}",
                 fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# 5. Combined training curves (all models)                                    #
# --------------------------------------------------------------------------- #
def plot_combined_curves(histories: Dict[str, List[dict]], out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (key, hist) in enumerate(histories.items()):
        epochs = list(range(1, len(hist) + 1))
        c = model_color(i)
        axes[0].plot(epochs, [h["val_loss"] for h in hist], "-o",
                     color=c, ms=3, label=display_name(key))
        axes[1].plot(epochs, [h["val_macro_f1"] for h in hist], "-o",
                     color=c, ms=3, label=display_name(key))
    axes[0].set_title("Validation loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[1].set_title("Validation macro-F1"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1"); axes[1].set_ylim(0, 1.02); axes[1].legend()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model comparison — training curves",
                 fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# 6. Confusion matrix                                                         #
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm, class_names: Sequence[str], out_path=None,
                          title="Confusion matrix", normalize=False):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cm_disp = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
        fmt = "{:.2f}"
    else:
        cm_disp = cm
        fmt = "{:.0f}"
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(0.9 * n + 2.5, 0.9 * n + 2))
    im = ax.imshow(cm_disp, cmap="Blues", vmin=0,
                   vmax=cm_disp.max() if cm_disp.max() > 0 else 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    thresh = cm_disp.max() / 2 if cm_disp.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, fmt.format(cm_disp[i, j]), ha="center", va="center",
                    fontsize=8,
                    color="white" if cm_disp[i, j] > thresh else INK)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# 7. Per-class F1 grouped bar                                                 #
# --------------------------------------------------------------------------- #
def plot_per_class_f1(per_model_f1: Dict[str, Sequence[float]],
                      class_names: Sequence[str], out_path=None,
                      title="Per-class F1 by model"):
    models = list(per_model_f1.keys())
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(1.3 * n_classes + 2, 4.8))
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * width, per_model_f1[m], width,
               label=display_name(m), color=model_color(i),
               edgecolor=SURFACE, linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("F1 score"); ax.set_ylim(0, 1.05)
    ax.set_title(title); ax.legend()
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# 8. Model comparison table                                                   #
# --------------------------------------------------------------------------- #
def plot_comparison_table(rows: List[dict], columns: List[str], out_path=None,
                          title="Model comparison"):
    fig, ax = plt.subplots(figsize=(1.7 * len(columns) + 1, 0.6 * len(rows) + 1.4))
    ax.axis("off")
    cell_text = [[str(r.get(c, "")) for c in columns] for r in rows]
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for j in range(len(columns)):  # header styling
        cell = table[0, j]
        cell.set_facecolor("#2a78d6")
        cell.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):  # zebra striping
        for j in range(len(columns)):
            table[i, j].set_facecolor("#f4f7fb" if i % 2 else "#ffffff")
    ax.set_title(title, fontweight="bold", color=INK, pad=12)
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# 9. TTA gain chart                                                           #
# --------------------------------------------------------------------------- #
def plot_tta_gain(tta: Dict[str, Dict[str, float]], metric="macro_f1",
                  out_path=None, title="Test-time augmentation gain"):
    models = list(tta.keys())
    x = np.arange(len(models))
    width = 0.36
    base = [tta[m]["without"] for m in models]
    with_tta = [tta[m]["with"] for m in models]
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 4.5))
    ax.bar(x - width / 2, base, width, label="without TTA", color="#898781",
           edgecolor=SURFACE, linewidth=1.0)
    ax.bar(x + width / 2, with_tta, width, label="with TTA", color="#2a78d6",
           edgecolor=SURFACE, linewidth=1.0)
    for xi, b, w in zip(x, base, with_tta):
        ax.text(xi + width / 2, w + 0.005, f"+{(w - b):.3f}", ha="center",
                va="bottom", fontsize=8, color="#006300")
    ax.set_xticks(x); ax.set_xticklabels([display_name(m) for m in models])
    ax.set_ylabel(metric.replace("_", " ").title()); ax.set_ylim(0, 1.05)
    ax.set_title(title); ax.legend()
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# 10. Grad-CAM grid                                                           #
# --------------------------------------------------------------------------- #
def plot_gradcam_grid(items: List[dict], out_path=None,
                      title="Grad-CAM (correct vs incorrect)", ncols=4):
    """``items``: list of dicts with keys ``overlay`` (HWC image), ``true``,
    ``pred``, ``correct`` (bool)."""
    n = len(items)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for k, ax in enumerate(axes):
        ax.axis("off")
        if k >= n:
            continue
        it = items[k]
        ax.imshow(to_hwc_uint8(it["overlay"]))
        ok = it.get("correct", it["true"] == it["pred"])
        col = "#006300" if ok else "#d03b3b"
        ax.set_title(f"T:{it['true']}\nP:{it['pred']}", fontsize=9, color=col)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(2.5)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# Evaluator demo: per-image, all-models comparison panel                      #
# --------------------------------------------------------------------------- #
def plot_model_comparison_panel(items, out_path=None,
                                title="Per-image model comparison (Grad-CAM evidence)"):
    """One row per image: the input, then each model's Grad-CAM with its
    prediction, confidence and a correct/incorrect border.

    ``items`` come from :meth:`src.inference.ModelComparer.compare`.
    """
    if not items:
        raise ValueError("No items to plot.")
    keys = list(items[0]["models"].keys())
    nrows, ncols = len(items), 1 + len(keys)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(2.9 * ncols, 3.3 * nrows), squeeze=False)
    for r, it in enumerate(items):
        ax = axes[r][0]
        ax.imshow(to_hwc_uint8(it["image"]))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        truth = it.get("true")
        ax.set_title(f"INPUT\ntrue: {truth if truth else 'unknown'}",
                     fontsize=9, color=INK, fontweight="bold")
        for sp in ax.spines.values():
            sp.set_edgecolor(SECONDARY); sp.set_linewidth(1.2)
        for c, k in enumerate(keys, start=1):
            m = it["models"][k]
            a = axes[r][c]
            a.imshow(to_hwc_uint8(m.get("gradcam", it["image"])))
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            ok = m.get("correct")
            col = MUTED if ok is None else ("#006300" if ok else "#d03b3b")
            mark = "" if ok is None else ("  ✓" if ok else "  ✗")
            a.set_title(f"{display_name(k)}\n{m['pred']} "
                        f"{m['confidence'] * 100:.1f}%{mark}",
                        fontsize=9, color=col)
            for sp in a.spines.values():
                sp.set_visible(True); sp.set_edgecolor(col); sp.set_linewidth(2.5)
    fig.suptitle(title, fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.03 / max(nrows, 1) - 0.01))
    _save(fig, out_path)
    return fig


def plot_contact_sheet(images, labels=None, ncols=8, out_path=None,
                       title="Available images"):
    """Numbered thumbnail sheet so an evaluator can see and pick images.

    ``images`` may be file paths **or** image arrays/tensors (e.g. rows of the
    cached uint8 tensor). ``labels`` gives the caption under each thumbnail
    (defaults to the position index).
    """
    from pathlib import Path as _P

    n = len(images)
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(1.6 * ncols, 1.85 * nrows), squeeze=False)
    flat = axes.ravel()
    for i, ax in enumerate(flat):
        ax.axis("off")
        if i >= n:
            continue
        img = images[i]
        if isinstance(img, (str, _P)):
            from PIL import Image
            with Image.open(img) as im:
                ax.imshow(im.convert("RGB").resize((128, 128)))
        else:
            ax.imshow(to_hwc_uint8(img))
        cap = labels[i] if labels is not None else i
        ax.set_title(f"[{cap}]", fontsize=8, color=SECONDARY)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_path)
    return fig


# --------------------------------------------------------------------------- #
# 11. t-SNE of penultimate features                                          #
# --------------------------------------------------------------------------- #
def plot_tsne(features, labels, class_names: Sequence[str], out_path=None,
              title="t-SNE of penultimate features", seed=42):
    from sklearn.manifold import TSNE

    features = np.asarray(features)
    labels = np.asarray(labels)
    perplexity = float(min(30, max(5, (len(features) - 1) / 3)))
    emb = TSNE(n_components=2, init="pca", perplexity=perplexity,
               random_state=seed).fit_transform(features)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for c, name in enumerate(class_names):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=16, alpha=0.8,
                   color=SPECIES_PALETTE[c % len(SPECIES_PALETTE)],
                   label=name, edgecolors=SURFACE, linewidths=0.3)
    ax.set_title(title); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=1.4, loc="best", ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)
    return fig
