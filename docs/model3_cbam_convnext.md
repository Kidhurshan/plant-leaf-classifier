# Model 3 — ConvNeXt-Tiny + CBAM (Proposed Model)

**timm checkpoint:** `convnext_tiny.fb_in22k_ft_in1k`  
**Role in the comparison:** Proposed architecture — a modernised CNN backbone augmented with a Convolutional Block Attention Module (CBAM) to add explicit channel-and-spatial attention, tested against both the CNN and Transformer baselines.

---

## 1. What Is ConvNeXt-Tiny?

ConvNeXt was introduced by Liu et al. (2022) with the premise: *"what if we take a standard ResNet and modernise every design choice by following the recipe from Vision Transformers?"*. The result is a pure convolutional network that matches or exceeds Swin Transformer on standard benchmarks while retaining the inductive biases of CNNs (locality, translation equivariance) and being simpler to implement.

The `fb_in22k_ft_in1k` checkpoint was pretrained by Meta AI (Facebook Research) on ImageNet-22k and fine-tuned on ImageNet-1k.

The "Tiny" variant is the smallest in the ConvNeXt family, making it better suited to a small dataset like EgyPLI than the larger variants (Small, Base, Large, XL).

---

## 2. ConvNeXt-Tiny Architecture

### 2.1 Macro-structure

```
Input (224×224×3)
│
├── Stem — Conv 4×4, stride 4 → 96ch, LayerNorm  [56×56]
│
├── Stage 1  — 3 × ConvNeXt Blocks → 96ch   [56×56]
├── Downsample — LayerNorm + Conv 2×2, stride 2 → 192ch  [28×28]
│
├── Stage 2  — 3 × ConvNeXt Blocks → 192ch  [28×28]
├── Downsample → 384ch  [14×14]
│
├── Stage 3  — 9 × ConvNeXt Blocks → 384ch  [14×14]   ← CBAM HERE
├── Downsample → 768ch  [7×7]
│
└── Stage 4  — 3 × ConvNeXt Blocks → 768ch  [7×7]    ← CBAM HERE
         └── LayerNorm → feature map [B, 768, 7, 7]
```

Stage depths: **[3, 3, 9, 3]**, channel widths: **[96, 192, 384, 768]**.

CBAM is attached to the **last two stages** (Stage 3 at 384ch and Stage 4 at 768ch) via PyTorch forward hooks, leaving the earlier stages and their pretrained weights untouched.

### 2.2 ConvNeXt Block

Each ConvNeXt block condenses the Swin Transformer block's ideas into a purely convolutional form:

```
Input x
│
├── DWConv 7×7 (depthwise, groups=C) — large-kernel spatial mixing
├── Permute [B, C, H, W] → [B, H, W, C]  (channels-last for LayerNorm)
├── LayerNorm(C)
├── Linear(C → 4C)   (pointwise expansion)
├── GELU
├── Linear(4C → C)   (pointwise projection)
├── Permute back to [B, C, H, W]
└── + residual (with learnable layer-scale scalar γ, init 1e-6)
```

Key design choices borrowed from transformers:
- **LayerNorm instead of BatchNorm** — normalises over channels per-token/patch, matching the transformer convention and improving transfer learning stability.
- **Large depthwise kernel (7×7)** — mimics the 7×7 self-attention window of Swin, giving each position a large spatial context without the quadratic cost of full attention.
- **GELU activation** — matches the activation used in transformers.
- **4× MLP expansion ratio** — same as transformer feed-forward layers.
- **Layer scale** — a learnable per-channel scalar (γ ≈ 1e-6 at init) that gates the residual branch, making very deep networks more stable to train.

### 2.3 Stem

ConvNeXt uses an **aggressive non-overlapping patchify stem**: a single 4×4 Conv with stride 4. This directly maps a 224×224 image to 56×56 feature maps in one step, matching the patch partition in Swin. This is in contrast to the 3×3 stride-2 stems in ResNets or the Fused-MBConv stem in EfficientNetV2.

### 2.4 Downsampling

Between stages, a `LayerNorm + Conv 2×2 stride 2` block halves spatial resolution and doubles channels. The LayerNorm before downsampling is a stability measure that prevents feature magnitude from exploding as channels are doubled.

---

## 3. CBAM — Convolutional Block Attention Module

CBAM (Woo et al., ECCV 2018) adds two sequential multiplicative gates on top of a feature map: first a **channel gate** that asks "which feature maps matter?", then a **spatial gate** that asks "where in those maps matters?".

### 3.1 Channel Attention

```
Input feature map x : [B, C, H, W]
│
├── GAP(x) → [B, C, 1, 1]   (average pool over spatial dims)
├── GMP(x) → [B, C, 1, 1]   (max pool over spatial dims)
│
├── Shared MLP on each:
│     Conv1×1(C → C/16) → ReLU → Conv1×1(C/16 → C)
│
├── Element-wise sum → [B, C, 1, 1]
└── Sigmoid → M_c : [B, C, 1, 1]   (channel attention map)
```

The shared-MLP design means fewer parameters: both pooled descriptors pass through the same Conv1×1 pair, so the cost is 2 × (C × C/16 + C/16 × C) = 2C²/8 parameters.

### 3.2 Spatial Attention

```
After channel gating: x' = x * (1 + γ_c * (M_c - 1))
│
├── Channel-wise avg pool: [B, 1, H, W]
├── Channel-wise max pool: [B, 1, H, W]
├── Concatenate → [B, 2, H, W]
├── Conv 7×7 (padding=3) → [B, 1, H, W]
└── Sigmoid → M_s : [B, 1, H, W]   (spatial attention map)

Output: x'' = x' * (1 + γ_s * (M_s - 1))
```

### 3.3 Identity-initialised residual gates

A critical implementation detail in `src/cbam.py`: instead of applying the gates directly (`x * M_c`), we use:

```python
x = x * (1.0 + gamma_c * (M_c - 1.0))
```

where `gamma_c` and `gamma_s` are **learnable scalars initialised to 0**. At the very first forward pass both gammas are zero, so the expression collapses to `x * 1.0 = x` — a perfect identity. The CBAM modules begin as no-ops and the network learns how much attention to apply as the gammas move away from zero during training.

This matters significantly for transfer learning: the pretrained ConvNeXt features are not disturbed at all at the start of training, and the attention branches are gradually "turned on" as the optimiser finds configurations that improve the loss. Without this initialisation, inserting CBAM between pretrained stages would corrupt the feature statistics immediately, wasting the pretraining investment.

### 3.4 Where CBAM is inserted

CBAM is **not placed inside the ConvNeXt backbone** (which remains unchanged). Instead, forward hooks are registered on the last two stage modules:

```python
# src/models.py
targets = [len(stage_list) - 2, len(stage_list) - 1]  # stages 2 and 3 (0-indexed)
# = 384-channel output and 768-channel output
```

The hook intercepts the stage's output tensor and passes it through the CBAM before it reaches the next stage. This means:

1. The CBAM parameters are stored as **top-level `cbam_modules`**, not inside `backbone`.
2. **In Phase 1**, when `set_backbone_trainable(False)` freezes `backbone`, the CBAM modules remain fully trainable alongside the shared head.
3. In Phase 2, everything (backbone + CBAM + head) is unfrozen together.

This architecture choice allows attention to be learned specifically for the EgyPLI leaf classification task from the very first epoch, while the backbone's general feature extraction is protected during Phase 1.

---

## 4. How the Combination Fits the EgyPLI Dataset

### 4.1 Dataset at a glance

| Property | Value |
|----------|-------|
| Species | Apple, Berry, Fig, Guava, Orange, Palm, Persimmon, Tomato |
| Total images | ~3,588 |
| Input resolution | 224 × 224 |
| Split | 70 / 15 / 15, group-aware stratified |

### 4.2 Why ConvNeXt + CBAM is the proposed model

**From ConvNeXt:**
- CNN inductive biases (locality, translation equivariance) naturally match the texture-heavy nature of leaf classification where local venation patterns are strongly discriminative.
- The 7×7 depthwise kernel gives each position a large receptive field without losing spatial precision, capturing mid-range patterns like vein spacing and leaf margin detail.
- LayerNorm and GELU make the backbone more compatible with fine-tuning at small scales than older BN-based CNNs.
- Smallest variant (~28.5 M parameters) — right-sized for ~2,512 training images.

**From CBAM (added on top):**
- **Channel attention** learns which feature channels are discriminative for each species. For example, channels encoding "green saturation" may suppress for palm (grey-green) and amplify for fresh tomato (bright green).
- **Spatial attention** learns where to look. Different leaf species have diagnostic regions: the base shape (round vs. lobed), the tip (acute vs. obtuse), and the centre midrib. The spatial gate can learn to focus on these regions automatically from the classification signal alone, without any spatial supervision.
- CBAM has been shown to improve fine-grained classification tasks (where the difference between categories is subtle and localised) more than global pooling networks or pure channel attention alone.

**Why not add CBAM to EfficientNetV2-S or Swin?**
- EfficientNetV2-S already has SE inside MBConv blocks — adding CBAM externally would create a redundant and awkward double-attention structure.
- Swin's self-attention already acts as a powerful spatial attention mechanism; CBAM's 7×7 conv is much weaker by comparison, and the outputs use NHWC layout which is less natural for CBAM hooks.
- ConvNeXt's NCHW output and its clean stage boundaries make it the most natural host for CBAM.

### 4.3 Potential limitations

- CBAM adds a small number of extra parameters (~0.3–0.5 M at 384ch and 768ch) and two extra forward passes per stage. The overhead is negligible on GPU.
- The benefit of CBAM is harder to measure than the benefit of the backbone itself. If the baseline CNN and transformer already achieve very high accuracy, the CBAM gain may be small in absolute terms. The comparison in notebook 05 provides empirical evidence.

---

## 5. Training Protocol

All three models follow the exact same protocol in `src/engine.py`.

### 5.1 Two-phase fine-tuning

| Phase | Epochs | Backbone | CBAM | Head | LR | Mix |
|-------|--------|----------|------|------|----|-----|
| 1 (warm-up) | 5 | **Frozen** | **Trainable** | **Trainable** | 1 × 10⁻³ | Off |
| 2 (fine-tune) | up to 35 (early-stop) | **Unfrozen** | **Trainable** | **Trainable** | 1 × 10⁻⁴ | On |

Phase 1 is where the CBAM modules learn their initial attention patterns while the pretrained ConvNeXt features are held fixed. Because the CBAM gates start as identities (γ=0), the head can warm up on unmodified ConvNeXt features first, then the gammas gradually grow as the head signal becomes more informative.

### 5.2 Optimiser & scheduler

- **Optimiser:** AdamW (β₁=0.9, β₂=0.999), weight decay 0.05.
- **LLRD:** backbone layers decay by 0.85 per depth level. CBAM and head parameters keep the full `phase2_lr`.
- **Scheduler (phase 2):** 2-epoch linear warm-up → cosine annealing to 0.
- **Gradient clipping:** global norm ≤ 1.0.
- **AMP:** bf16 / fp16 + GradScaler.

### 5.3 Loss

**Focal loss** (γ=2, α=inverse class frequency). The same loss as the other two models — controlled comparison requires identical loss functions.

### 5.4 Augmentation (phase 2)

| Transform | Parameters |
|-----------|-----------|
| RandomResizedCrop | scale [0.7, 1.0] → 224×224 |
| RandomHorizontalFlip | p=0.5 |
| RandomVerticalFlip | p=0.5 |
| RandomRotation | ±30° |
| ColorJitter | brightness/contrast/saturation 0.3 |
| RandomErasing | p=0.25 |
| Mixup | α=0.2, p=0.5 per batch |
| CutMix | α=1.0, p=0.5 per batch |

All transforms execute on-GPU via `src/augment.py`.

### 5.5 Evaluation

- **Primary metric:** validation macro-F1.
- **Early stopping:** patience 8 epochs on val macro-F1.
- **TTA:** average softmax over {original, hflip, vflip, hflip+vflip}.
- **GradCAM** (`src/gradcam.py`) is used in notebook 05 to visualise which spatial regions the model attends to — this directly illustrates whether the CBAM spatial gate is focusing on the leaf body rather than the background.

---

## 6. Model Summary

| Component | Detail |
|-----------|--------|
| Backbone | `convnext_tiny.fb_in22k_ft_in1k` |
| Backbone params | ~28.5 M |
| CBAM on Stage 3 | `CBAM(384)` — ~0.37 M additional params |
| CBAM on Stage 4 | `CBAM(768)` — ~1.47 M additional params |
| Feature dim | 768 |
| Head | GAP → LayerNorm(768) → Dropout(0.3) → Linear(768→8) |
| use_cbam | True |

---

## 7. Quick Reference

```python
from src.config import load_config
from src.models import build_model

cfg = load_config()
model = build_model(cfg, "cbam_convnext", device=device)
# backbone: convnext_tiny.fb_in22k_ft_in1k
# num_features: 768
# use_cbam: True
# cbam_modules: ModuleList with 2 CBAM blocks (384ch, 768ch)
```

**Checkpoint:** `checkpoints/cbam_convnext_best.pt`  
**History CSV:** `results/metrics/cbam_convnext_history.csv`  
**Run meta:** `results/metrics/cbam_convnext_run_meta.json`  
**GradCAM visualisations:** `results/gradcam/`

---

## 8. References

- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). *A ConvNet for the 2020s.* CVPR 2022.
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module.* ECCV 2018.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. (layer-scale and skip connections lineage)