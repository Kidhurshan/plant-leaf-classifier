# Claude Code Prompt for EC8010/EC8020 Design Task 4

Open your empty project folder in VS Code, start Claude Code, and paste everything inside the block below as your first message.

---

Build a complete, reproducible plant leaf species classification project. I am working under a hard 30 hour deadline, so the code must run correctly the first time. Prefer proven, simple approaches over clever ones. Do not leave placeholders, TODOs, or stub functions.

## 1. Context and execution environment

This is an academic design task. I must train and compare three deep learning models on the EgyPLI dataset, present visual results, and demonstrate the best model live while an evaluator picks 5 random images. Accuracy is the graded criterion. Training hardware and training time are not graded.

I write code locally in VS Code and execute it on a remote Google Colab runtime through the official Colab VS Code extension. This is the single most important architectural constraint:

- The notebook file is local, but the kernel is remote.
- The remote runtime cannot see any of my local files.
- Therefore all Python modules must be pushed to GitHub and cloned onto the runtime by the notebook itself.

Design for this from the very first file. Target GPU is A100 40GB, with L4 as fallback. Use bfloat16 mixed precision where supported and float16 otherwise, detected at runtime.

## 2. Architectural rules

**All logic lives in `src/*.py`. Notebooks are thin drivers.**

Notebooks must not contain training loops, model definitions, or data pipeline code. A notebook cell should be a few lines that call into `src/` and then display something. There are two reasons, and I want the code to honour both:

1. All three models must be trained by identical shared code so that the only variable is the backbone. If the training loop were duplicated per notebook it would drift and the comparison would be invalid.
2. `.py` files produce readable Git diffs. Notebooks do not.

Notebooks are still cell-by-cell and visual. Each one should walk through its stage step by step, printing and plotting as it goes.

## 3. Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── config.py          # YAML loading, dataclass config, validation
│   ├── utils.py           # seeding, device/AMP detection, timing, logging, paths
│   ├── data.py            # inspection, caching, stratified splits, GPU tensor dataset
│   ├── augment.py         # GPU batch augmentation
│   ├── cbam.py            # CBAM module
│   ├── models.py          # three model builders, shared head
│   ├── losses.py          # focal loss, CE with label smoothing
│   ├── engine.py          # two-phase train loop, early stopping, checkpointing
│   ├── evaluate.py        # metrics, classification report, confusion matrix
│   ├── tta.py             # test-time augmentation
│   ├── ensemble.py        # confidence-weighted soft voting
│   ├── gradcam.py         # Grad-CAM overlays
│   └── viz.py             # ALL plotting functions, consistent style
├── scripts/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train.py           # CLI: --model, --smoke, --config
│   ├── evaluate_all.py
│   └── predict.py         # CLI for the live demo
├── notebooks/
│   ├── 01_setup_and_data.ipynb
│   ├── 02_train_efficientnetv2s.ipynb
│   ├── 03_train_swin_small.ipynb
│   ├── 04_train_cbam_convnext.ipynb
│   ├── 05_compare_and_ensemble.ipynb
│   └── 06_live_demo.ipynb
└── results/
    ├── figures/
    ├── metrics/
    └── gradcam/
```

Every plotting function goes in `src/viz.py` with one consistent visual style, so every figure in my report looks like it belongs to the same project. Use a colourblind-safe palette, readable font sizes, titles and axis labels on everything, and save at 200 DPI.

## 4. Git setup, and do this first

Before writing any other code, create `.gitignore` and make the first commit. I have lost work to this before.

`.gitignore` must exclude at minimum:

```
data/
*.pt
*.pth
*.zip
checkpoints/
runs/
__pycache__/
*.pyc
.ipynb_checkpoints/
kaggle.json
.env
.DS_Store
```

Critical: `kaggle.json` is an API credential and must never be committed. If you generate any code that reads it, read it from an environment variable or from Colab's `userdata` secrets, with a documented manual fallback.

Do commit `results/figures/` and `results/metrics/`, since they are small and I need them for my report. Do not commit model checkpoints or the cached dataset tensor.

Notebooks should be committed with their outputs intact, because I need the visuals visible during the demo and viva. Do not add nbstripout.

In the README, document this exact workflow:

1. Local: `git init`, add `.gitignore`, first commit, create the GitHub repo, push.
2. In each notebook, cell 1 clones the repo if absent or pulls if present, then installs requirements.
3. When I change `src/` locally: commit, push, then run the notebook's sync cell to `git pull`. Autoreload picks up the change without a kernel restart.

Write a short `sync` helper used at the top of every training notebook that runs `git pull`, prints the current commit hash, and confirms which version of the code is executing. Every results file must record the commit hash it was produced from, so my numbers are traceable.

Suggest a commit message convention and milestone tags in the README, for example `v0.1-smoke-passing`, `v0.2-data-cached`, `v1.0-all-models-trained`. My commit history serves as the logbook proof of work my course requires, so encourage frequent, descriptive commits.

Work on `main`. Do not create feature branches.

## 5. Dataset

Kaggle slug: `mahmoudshaheen1134/plant-leaf-image-dataset` (EgyPLI, roughly 63 MB, 3,588 images, 8 species: apple, berry, fig, guava, orange, plum, persimmon, tomato, both healthy and diseased leaves, images around 256x256).

Do NOT assume the folder layout. `scripts/inspect_data.py` must download the dataset and print the real directory tree, per-folder file counts, image dimensions and formats, and any corrupt files. The loader must adapt to what is actually there, including nested healthy and diseased subfolders that need merging into one species label. Print the final class-to-count mapping, then assert 8 classes and warn loudly rather than crash if the total is not 3,588.

## 6. Performance approach

The dataset is small, so remove the data loading bottleneck entirely. In `prepare_data.py`, decode every image once, resize to 256x256, and cache the whole dataset as one uint8 tensor (roughly 705 MB) plus an integer label array. At training time, load that tensor to GPU memory once and index it directly. Perform all augmentation on GPU in batch. Do not build a per-image DataLoader that decodes JPEGs each epoch.

Stratified 70/15/15 train/validation/test split with a fixed seed, written to `results/metrics/splits.csv` so all three models see identical data. The test set is used exactly once, at final evaluation.

## 7. Models

Three models, all built through `timm` with `pretrained=True` and `num_classes=0`, then given my own shared head:

1. `tf_efficientnetv2_s.in21k_ft_in1k`
2. `swin_small_patch4_window7_224.ms_in22k_ft_in1k`
3. `convnext_tiny.fb_in22k_ft_in1k` wrapped with CBAM, my proposed model

Before creating a model, verify the checkpoint name exists via `timm.list_models(pretrained=True)`, and fall back to a documented alternative with a clear printed message if not. All three train at 224x224. Swin is fixed at 224 by its window-7 design, so never attempt another size for it.

**Shared head**, identical across all three so the comparison is controlled: global average pooling, LayerNorm, Dropout(0.3), Linear to 8 classes.

**CBAM** in `cbam.py`, implemented properly: a channel attention branch using both average and max pooling through a shared MLP with reduction ratio 16, then a spatial attention branch concatenating channel-wise average and max maps into a 7x7 convolution, each applied as a multiplicative residual gate. Insert CBAM after the later ConvNeXt stages using `features_only=True` or forward hooks, whichever is cleaner. Initialise it close to identity so pretrained features are not destroyed on the first step. Include a shape assertion test for all three models.

## 8. Training

**Two phases.** Phase 1 freezes the backbone and trains only the head for about 5 epochs at a higher learning rate. Phase 2 unfreezes everything and fine-tunes for up to 35 epochs at a low learning rate with cosine schedule and short warmup. Use layer-wise learning rate decay if it is straightforward.

**Loss.** Focal loss with gamma 2.0 and alpha from inverse class frequency, to address the under-represented tomato and persimmon classes. Make it config-selectable with cross entropy plus label smoothing 0.1 as the alternative.

**Augmentation** on GPU: random resized crop to 224, horizontal and vertical flips, rotation to 30 degrees, colour jitter on brightness, contrast and saturation for outdoor lighting, and random erasing at probability 0.25 for occlusion. Mixup and CutMix behind a config flag, default on in phase 2 only.

**Loop.** AMP with GradScaler, AdamW, gradient clipping at 1.0, early stopping on validation macro F1 with patience 8, best-only checkpointing. Log per-epoch train loss, validation loss, accuracy and macro F1 to `results/metrics/{model}_history.csv`. After epoch one, print an estimated total runtime so I can plan.

**Robustness.** Checkpoint to a configurable directory that can point at mounted Google Drive, and resume automatically if a checkpoint exists, so a dropped session never costs a full re-run. Auto-reduce batch size on CUDA out-of-memory instead of crashing.

**Reproducibility.** One `set_seed()` covering Python, NumPy, PyTorch and CUDA. Record seed, library versions, GPU name, git commit hash and full config in `results/metrics/{model}_run_meta.json`.

## 9. Evaluation and visuals

`src/viz.py` must provide, and the notebooks must display, all of the following:

1. Class distribution bar chart
2. Sample image grid showing all 8 species
3. Augmentation preview grid, same image before and after
4. Training curves per model: loss, accuracy, macro F1
5. Combined training curves comparing all three models
6. Confusion matrix heatmaps, annotated, for each model and the ensemble
7. Per-class F1 grouped bar chart comparing the three models
8. Model comparison table: accuracy, macro precision, recall, F1, parameters, training time
9. TTA gain chart, with and without
10. Grad-CAM overlay grid, correct and incorrect examples per class
11. t-SNE plot of penultimate-layer features coloured by species

Every figure saves to `results/figures/` at 200 DPI as PNG, ready to paste into my 5 page report.

**Metrics.** Overall accuracy, per-class and macro precision, recall and F1, classification report, confusion matrix as CSV and PNG.

**TTA** in `tta.py`: average softmax over original, horizontal flip, vertical flip, and both. Report metrics with and without so I can show the gain.

**Ensemble** in `ensemble.py`: confidence-weighted soft voting with weights proportional to validation macro F1. Report alongside individual models.

**Grad-CAM** in `gradcam.py`: overlays for correct and incorrect test predictions per class, saved to `results/gradcam/`.

`scripts/evaluate_all.py` must produce `results/metrics/summary_table.md` and `summary_table.csv` comparing all three models plus the ensemble. This is the centrepiece table of my report.

## 10. Notebooks

Each notebook starts with an identical two-cell preamble: environment and GPU report, then clone-or-pull plus install plus autoreload setup. Each notebook ends by printing where its outputs were saved.

**`01_setup_and_data.ipynb`** GPU and VRAM check, bfloat16 support check, clone and install, optional Drive mount, Kaggle auth and download, data inspection output, cache building, split creation, then visuals 1, 2 and 3 above.

**`02` to `04`, one per model.** Sync cell, load config, build model and print parameter count and a layer summary, preview one augmented batch, run phase 1 and phase 2 training with live progress, plot that model's training curves, evaluate on validation with a confusion matrix, save checkpoint and history. Keep each notebook under about 15 cells. They must differ only in the model name passed to shared code, and add a markdown cell in each explaining what makes that architecture different and why it was chosen.

**`05_compare_and_ensemble.ipynb`** Load all three histories and checkpoints, produce visuals 4 through 11, run TTA, build the ensemble, evaluate everything on the held-out test set exactly once, and export the summary table and all report figures.

**`06_live_demo.ipynb`** Deliberately minimal and fast, because I run this in front of an evaluator. Load the best model or ensemble, accept a folder or a list of image paths, and for each image display the picture, the predicted species, the confidence percentage, the top-3 probabilities, and optionally a Grad-CAM overlay. It must not depend on any training state, must load only from saved checkpoints, and must fail gracefully with a clear message if a checkpoint is missing. Add a markdown cell at the top with one-line instructions so I can run it under pressure without thinking.

Add a markdown reminder in each notebook to disconnect and delete the Colab runtime when finished, since compute units are consumed while a runtime is connected.

## 11. Smoke test

Every script and the training notebooks must support a `--smoke` flag or `SMOKE = True` variable using about 200 images, 2 epochs, and all three models, finishing in under 5 minutes. This proves the pipeline end to end before I commit real time. Document it in the README as the mandatory first run.

## 12. Config

One `configs/default.yaml` holding seed, paths, image size, batch size, epochs per phase, learning rates, weight decay, loss choice, augmentation flags, patience, checkpoint directory and the model list. No hyperparameters hard-coded anywhere else. Validate the config on load and fail early with a clear message.

## 13. Build order

Complete and verify these in order, and tell me clearly when each is done:

1. Git init, `.gitignore`, README skeleton, requirements, first commit.
2. Config, utils, seeding.
3. Data inspection, caching, splits, with the class assertion passing.
4. CBAM, models, shared head, shape assertions passing for all three.
5. Losses, augmentation, training engine.
6. `viz.py` with all plotting functions.
7. Smoke test running end to end.
8. Evaluation, TTA, ensemble, Grad-CAM.
9. All six notebooks.
10. Prediction script and report assets.

Pin versions in `requirements.txt` including `timm`, and state the assumed Python version. Write concise docstrings. Prefer clarity over cleverness throughout. Commit after each numbered stage with a descriptive message.

---

## After Claude Code finishes

1. Locally: verify `.gitignore` is correct, then `git init`, commit, create the GitHub repo, push. Use a public repo unless you need it private, in which case set up a personal access token for the clone step.
2. In VS Code, open `notebooks/01_setup_and_data.ipynb`, choose Select Kernel, then Colab, then New Colab Server, and pick A100 if offered.
3. Run notebook 01 fully, then the smoke test. Do not proceed until the smoke test passes.
4. Run notebooks 02, 03, 04.
5. Run notebook 05 and collect `results/` for your report.
6. Test notebook 06 with 5 images before the demo.
7. Disconnect and delete the runtime.

Remember the sync loop: after editing `src/` locally you must commit, push, and run the notebook's sync cell before the change takes effect on the runtime.
