"""Gradio web demo: an evaluator uploads leaf images from any browser and sees
all three models judge them side by side, with Grad-CAM evidence.

Why Gradio: ``google.colab.files.upload()`` is a browser-frontend widget and does
NOT work when the notebook is driven from the VS Code Colab extension (the kernel
is remote but there is no Colab browser frontend to talk to). Gradio launches its
own web page with a public ``*.gradio.live`` share link, so the upload happens in
a normal browser tab and the VS Code frontend is bypassed entirely.

Flow (one image at a time, so each image gets its own label):
  upload image  ->  optionally pick the true species  ->  Classify
  ->  panel of INPUT + 3 model Grad-CAMs (with prediction, confidence, +/-)
  ->  a running SESSION SCOREBOARD accumulates per-model accuracy over every
      LABELLED upload (Unknown images are predicted but not scored).

The heavy lifting is done by :class:`src.inference.ModelComparer`; this module is
only the UI layer.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from src import viz


# --------------------------------------------------------------------------- #
# Markdown builders                                                           #
# --------------------------------------------------------------------------- #
def _detail_md(item: dict) -> str:
    truth = item.get("true") or "unknown"
    agree = "ALL MODELS AGREE" if item["agreement"] else "MODELS DISAGREE"
    lines = [
        f"**Image:** `{os.path.basename(item['path'])}` &nbsp;|&nbsp; "
        f"**true species:** {truth} &nbsp;|&nbsp; **{agree}**",
        "",
        "| Model | Prediction | Confidence | Top-3 |",
        "|---|---|---|---|",
    ]
    for key, m in item["models"].items():
        mark = "" if m["correct"] is None else (" ✅" if m["correct"] else " ❌")
        top3 = ", ".join(f"{n} {p * 100:.1f}%" for n, p in m["topk"])
        lines.append(
            f"| {viz.display_name(key)}{mark} | **{m['pred']}** | "
            f"{m['confidence'] * 100:.1f}% | {top3} |"
        )
    return "\n".join(lines)


def _scoreboard_md(items: List[dict], model_keys) -> str:
    if not items:
        return "### Session scoreboard\n_No images classified yet._"
    labelled = [it for it in items if it.get("true")]
    agree = sum(1 for it in items if it["agreement"]) / len(items)
    lines = [
        "### Session scoreboard",
        f"Images classified this session: **{len(items)}** "
        f"(labelled: **{len(labelled)}**)  &nbsp;|&nbsp; "
        f"all-model agreement: **{agree * 100:.0f}%**",
    ]
    if labelled:
        lines += ["", "| Model | Correct | Accuracy | Mean confidence |",
                  "|---|---|---|---|"]
        n = len(labelled)
        for key in model_keys:
            n_ok = sum(1 for it in labelled if it["models"][key]["correct"])
            mconf = float(np.mean([it["models"][key]["confidence"]
                                   for it in labelled]))
            lines.append(
                f"| {viz.display_name(key)} | {n_ok}/{n} | "
                f"{n_ok / n * 100:.1f}% | {mconf * 100:.1f}% |"
            )
    else:
        lines.append("\n_Pick a true species when uploading to score accuracy._")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# App                                                                        #
# --------------------------------------------------------------------------- #
def build_demo(comparer):
    """Build (but do not launch) the Gradio app around a ``ModelComparer``."""
    import gradio as gr
    import matplotlib
    matplotlib.use("Agg")            # server-side rendering, no display needed
    import matplotlib.pyplot as plt

    model_keys = list(comparer.models)
    choices = ["Unknown"] + list(comparer.class_names)

    intro = (
        "# 🌿 Leaf Species Classifier — 3-Model Evaluator Demo\n"
        "Upload any leaf image. All three trained models classify it and show a "
        "**Grad-CAM** heat map of the pixels each used. Optionally set the true "
        "species to score **accuracy** across everything you upload.\n\n"
        f"**Species:** {', '.join(comparer.class_names)}"
    )

    def classify(img_path, label, items):
        plt.close("all")             # keep only the latest figure in memory
        items = items or []
        if not img_path:
            return None, "⚠️ Please upload an image first.", \
                _scoreboard_md(items, model_keys), items
        results = comparer.compare([img_path], top_k=3, with_gradcam=True)
        item = results[0]
        if label and label != "Unknown":
            item["true"] = label
            for m in item["models"].values():
                m["correct"] = (m["pred"] == label)
        fig = viz.plot_model_comparison_panel(
            [item], title="Prediction + Grad-CAM (all three models)")
        items = items + [item]
        return fig, _detail_md(item), _scoreboard_md(items, model_keys), items

    def reset():
        return None, "", _scoreboard_md([], model_keys), []

    with gr.Blocks(title="Leaf Classifier — Evaluator Demo") as demo:
        gr.Markdown(intro)
        state = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="filepath", label="Upload a leaf image",
                                  sources=["upload", "clipboard"], height=260)
                label_in = gr.Dropdown(choices, value="Unknown",
                                       label="True species (optional)")
                with gr.Row():
                    go = gr.Button("Classify", variant="primary")
                    clr = gr.Button("Reset scoreboard")
            with gr.Column(scale=2):
                board = gr.Markdown(_scoreboard_md([], model_keys))
        panel = gr.Plot(label="Per-model comparison")
        detail = gr.Markdown()

        go.click(classify, [img_in, label_in, state],
                 [panel, detail, board, state])
        clr.click(reset, None, [panel, detail, board, state])
    return demo


def launch_demo(comparer, share: bool = True, **kwargs):
    """Build and launch the demo. On Colab, ``share=True`` prints a public
    ``*.gradio.live`` URL the evaluator opens in any browser."""
    demo = build_demo(comparer)
    return demo.launch(share=share, **kwargs)
