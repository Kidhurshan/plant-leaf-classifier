# Plant Leaf Species Classification — EC8010/EC8020 Design Task 4

Train and compare **three deep-learning models** on the **EgyPLI** plant-leaf
dataset (8 species, ~3,588 images), present visual results, and demo the best
model live. **Accuracy is the graded criterion.**

The three models (all built through `timm`, `pretrained=True`, with an identical
shared head so the only variable is the backbone):

| # | Backbone | timm checkpoint | Role |
|---|----------|-----------------|------|
| 1 | EfficientNetV2-S | `tf_efficientnetv2_s.in21k_ft_in1k` | CNN baseline |
| 2 | Swin-Small | `swin_small_patch4_window7_224.ms_in22k_ft_in1k` | Transformer |
| 3 | ConvNeXt-Tiny + **CBAM** | `convnext_tiny.fb_in22k_ft_in1k` | **Proposed model** |

---

## Architecture in one paragraph

**All logic lives in `src/*.py`. Notebooks are thin drivers.** A notebook cell
calls into `src/` and then displays something — it never contains a training
loop, a model definition, or a data-pipeline. The three models are trained by
**identical shared code** (`src/engine.py`), so the comparison is controlled.
`.py` files also produce readable Git diffs; notebooks do not.

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── configs/default.yaml     # the ONLY place hyperparameters live
├── src/                     # all logic (config, data, models, engine, viz, ...)
├── scripts/                 # thin CLIs (inspect_data, prepare_data, train, evaluate_all, predict)
├── notebooks/               # 01..06, thin visual drivers
└── results/                 # figures/ metrics/ gradcam/  (figures+metrics are committed)
```

---

## The remote-kernel constraint (read this first)

Code is written **locally in VS Code** but executed on a **remote Google Colab
runtime** via the official Colab VS Code extension. **The runtime cannot see your
local files.** Therefore every Python module is pushed to GitHub and cloned onto
the runtime by the notebook itself.

### Sync loop (memorise this)

When you edit anything under `src/` locally:

1. `git commit -am "..."` and `git push`
2. Run the notebook's **sync cell** (`src.utils.sync_repo()` → `git pull`)
3. `%autoreload` picks up the change **without a kernel restart**

Every results file records the **git commit hash** it was produced from, so all
numbers are traceable back to the exact code that made them.

---

## First-time setup

### 1. Local git (do this before anything else)

```bash
git init
git add .gitignore README.md requirements.txt
git commit -m "chore: initial commit — gitignore, readme, requirements"
git add .
git commit -m "feat: full project scaffold (src, scripts, notebooks, configs)"
# create the EMPTY GitHub repo, then:
git remote add origin https://github.com/Kidhurshan/plant-leaf-classifier.git
git branch -M main
git push -u origin main
```

Work on `main`. **Do not create feature branches.**

> Public repo is simplest for the Colab clone. For a **private** repo, create a
> GitHub Personal Access Token and clone with
> `https://<TOKEN>@github.com/<you>/<repo>.git`.

### 2. Kaggle credentials (never commit `kaggle.json`)

`kaggle.json` is an API credential and is git-ignored. The code reads it from,
in order:

1. Environment variables `KAGGLE_USERNAME` / `KAGGLE_KEY`
2. **Colab secrets** (`google.colab.userdata`) keys `KAGGLE_USERNAME` / `KAGGLE_KEY`
   — recommended on Colab: 🔑 sidebar → add both secrets.
3. Manual fallback: upload `kaggle.json` to the runtime (`~/.kaggle/kaggle.json`,
   `chmod 600`). Notebook 01 walks you through this.

### 3. Open on Colab

In VS Code: **Select Kernel → Colab → New Colab Server → A100** (L4 is a fine
fallback). Then run `notebooks/01_setup_and_data.ipynb` top to bottom.

---

## Mandatory first run — the smoke test

**Prove the whole pipeline end-to-end before spending real compute.** Every
script and the training notebooks support a smoke mode (~200 images, 2 epochs,
all three models, **under 5 minutes**):

```bash
# after notebook 01 has cached the dataset:
python scripts/train.py --model all --smoke
```

or set `SMOKE = True` in a training notebook's first cell. **Do not proceed to
real training until the smoke test passes.**

---

## Full run order

1. `notebooks/01_setup_and_data.ipynb` — GPU/bf16 check, clone+install, Kaggle
   download, data inspection, cache build, stratified 70/15/15 split, EDA visuals.
2. `notebooks/02_train_efficientnetv2s.ipynb`
3. `notebooks/03_train_swin_small.ipynb`
4. `notebooks/04_train_cbam_convnext.ipynb`
5. `notebooks/05_compare_and_ensemble.ipynb` — all comparison figures, TTA,
   ensemble, **test set used exactly once**, `results/metrics/summary_table.md`.
6. `notebooks/06_live_demo.ipynb` — load best model/ensemble, predict on 5 images.

> **Disconnect and delete the Colab runtime when finished** — compute units are
> consumed the whole time a runtime is connected.

---

## Reproducibility & traceability

- One `set_seed()` covers Python, NumPy, PyTorch, CUDA.
- Each model writes `results/metrics/{model}_run_meta.json` recording seed,
  library versions, GPU name, git commit hash, and the full config.
- Each model writes `results/metrics/{model}_history.csv` (per-epoch loss / acc /
  macro-F1).
- The stratified split is written **once** to `results/metrics/splits.csv` so all
  three models see identical data.

---

## Commit convention & milestone tags

Frequent, descriptive commits — your history is the **logbook / proof-of-work**.

Format: `type: short imperative summary` (`feat`, `fix`, `chore`, `docs`,
`refactor`, `exp`). Examples:

```
feat: add CBAM module with identity init
exp: efficientnetv2s two-phase run, val macroF1=0.9x
fix: adapt loader to nested healthy/diseased folders
```

Milestone tags:

| Tag | Meaning |
|-----|---------|
| `v0.1-smoke-passing` | smoke test runs end-to-end |
| `v0.2-data-cached` | dataset cached + splits written |
| `v0.5-model-shapes-ok` | all three models pass shape assertions |
| `v1.0-all-models-trained` | 02/03/04 complete, checkpoints saved |
| `v1.1-report-assets` | notebook 05 figures + summary table exported |

---

## Config

Everything tunable lives in `configs/default.yaml` (seed, paths, image size,
batch size, epochs per phase, learning rates, weight decay, loss choice,
augmentation flags, patience, checkpoint dir, model list). **No hyperparameters
are hard-coded anywhere else.** The config is validated on load and fails early
with a clear message.
