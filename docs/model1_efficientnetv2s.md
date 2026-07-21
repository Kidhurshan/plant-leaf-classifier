# Model 1 — EfficientNetV2-S (CNN Baseline)

**timm checkpoint:** `tf_efficientnetv2_s.in21k_ft_in1k`  
**Role in the comparison:** CNN baseline — a strong, well-established convolutional network against which the transformer and the proposed attention model are measured.

---

## 1. What Is EfficientNetV2-S?

EfficientNetV2-S is the "Small" variant of the EfficientNetV2 family introduced by Tan & Le (2021). It improves on the original EfficientNet in two key ways:

1. **Fused-MBConv in early stages** — replaces the depthwise-separable convolution stack (MBConv) with a single, fused 3×3 convolution in the first three stages. This is faster to train because fused ops utilise GPU tensor cores more efficiently.
2. **Progressive learning** — during the original training recipe the authors grew both the image resolution and regularisation strength together, which significantly cut training time while retaining accuracy.

The `tf_` prefix on the checkpoint name indicates it was ported from the TensorFlow/TPU training recipe. The suffix `in21k_ft_in1k` tells us the weights were first pretrained on **ImageNet-21k** (~14 million images, 21,843 classes) and then fine-tuned on **ImageNet-1k** (1.28 million images, 1,000 classes). This two-stage pretraining gives the backbone a substantially richer feature vocabulary than single-dataset pretraining.

---

## 2. Architecture

### 2.1 Macro-structure

```
Input (224×224×3)
│
├── Stem — Conv 3×3, stride 2 → 24ch  [112×112]
│
├── Stage 1  — 2 × Fused-MBConv (expand 1, 3×3, stride 1) → 24ch  [112×112]
├── Stage 2  — 4 × Fused-MBConv (expand 4, 3×3, stride 2) → 48ch  [56×56]
├── Stage 3  — 4 × Fused-MBConv (expand 4, 3×3, stride 2) → 64ch  [28×28]
│
├── Stage 4  — 6 × MBConv (expand 4, 3×3, SE 0.25, stride 2) → 128ch  [14×14]
├── Stage 5  — 9 × MBConv (expand 6, 3×3, SE 0.25, stride 1) → 160ch  [14×14]
├── Stage 6  — 15 × MBConv (expand 6, 3×3, SE 0.25, stride 2) → 256ch  [7×7]
│
├── Head Conv — 1×1 → 1280ch  [7×7]
│
└── [timm global_pool=''] → feature map [B, 1280, 7, 7]
```

Because we build with `num_classes=0, global_pool=''`, timm returns the **feature map** `[B, 1280, 7, 7]` instead of a scalar prediction. Our shared head then performs global average pooling.

### 2.2 Building blocks

| Block | Used in | Description |
|-------|---------|-------------|
| **Fused-MBConv** | Stages 1–3 | `Conv2d(expand) → BN → Act → Conv2d(project) → BN` — no depthwise. |
| **MBConv** | Stages 4–6 | `Conv1×1(expand) → DWConv3×3 → SE → Conv1×1(project)`. Squeeze-and-Excitation (SE) recalibrates channel importance. |

Both blocks use skip connections (residual add) when the input and output channel counts match and the stride is 1.

### 2.3 Squeeze-and-Excitation (SE)

Stages 4–6 apply SE at ratio 0.25 inside each MBConv. SE is a lightweight channel attention mechanism:

```
GAP → FC(C → C/4) → ReLU → FC(C/4 → C) → Sigmoid → scale feature map
```

This is an important precursor to CBAM — SE operates only on channels, while CBAM (used in Model 3) adds a spatial branch on top.

### 2.4 Shared head (identical across all three models)

```python
GAP(feature_map)        # [B, 1280, 7, 7] → [B, 1280]
LayerNorm(1280)
Dropout(0.3)
Linear(1280 → 8)        # 8 EgyPLI species
```

**Total parameters (backbone + head):** approximately **21.5 M**  
**Feature dimension passed to the head:** 1280

---

## 3. Pretraining & Transfer Learning

| Stage | Dataset | Classes | Images |
|-------|---------|---------|--------|
| Pretrain | ImageNet-21k | 21,843 | ~14 M |
| Fine-tune | ImageNet-1k | 1,000 | 1.28 M |
| Task fine-tune | EgyPLI | 8 | ~3,588 |

The two-step ImageNet recipe means the backbone already understands a wide range of visual concepts — textures, edges, object parts — before it ever sees a leaf. Fine-tuning on EgyPLI adapts these general representations to leaf venation patterns, colour profiles, and shape cues specific to the 8 species.

---

## 4. How It Fits the EgyPLI Dataset

### 4.1 Dataset at a glance

| Property | Value |
|----------|-------|
| Species | Apple, Berry, Fig, Guava, Orange, Palm, Persimmon, Tomato |
| Total images | ~3,588 |
| Input resolution | 224 × 224 |
| Split | 70 / 15 / 15 (train / val / test), group-aware stratified |

### 4.2 Why EfficientNetV2-S is a good fit

- **Compact dataset, compact model.** With only ~2,512 training images, a model in the 21 M parameter range is much less likely to overfit than a 50 M+ model with a similar head.
- **Inductive biases for texture/colour.** CNNs have built-in translation equivariance and locality bias. Leaves of different species differ primarily in local texture (venation density, surface finish) and global shape — both of which CNNs capture efficiently with shallow receptive fields before expanding them stage by stage.
- **High throughput.** On a single A100/L4 GPU, EfficientNetV2-S is the fastest of the three models to train, meaning more epochs can be run within a session's compute budget.
- **SE attention already present.** Stages 4–6 already have per-channel attention, giving the backbone some ability to emphasise spectrally distinctive leaf colours even without CBAM.

### 4.3 Potential limitations

- Pure CNN: receptive field grows locally stage-by-stage. Long-range spatial relationships (e.g. overall leaf shape silhouette) are harder to capture than for the Swin transformer.
- No explicit spatial attention: the SE block only suppresses or amplifies channels uniformly across the spatial map. Regions of diagnostic importance (e.g. leaf tip vs. base) are not selectively attended to — this is the gap that CBAM fills in Model 3.

---

## 5. Training Protocol

All three models follow the exact same protocol in `src/engine.py`.

### 5.1 Two-phase fine-tuning

| Phase | Epochs | Backbone | LR | Mix |
|-------|--------|----------|----|-----|
| 1 (warm-up head) | 5 | **Frozen** | 1 × 10⁻³ | Off |
| 2 (full fine-tune) | up to 35 (early-stop) | **Unfrozen** | 1 × 10⁻⁴ | On |

Phase 1 trains only the shared head while the pretrained backbone weights are locked. This protects the rich ImageNet features from being destroyed in the first few steps when the head gradients are large and noisy.

Phase 2 unfreezes everything. **Layer-wise LR decay (LLRD)** is applied with a per-depth multiplier of 0.85 — so the earliest backbone layers (stem) receive a much smaller learning rate than the head, reducing the risk of catastrophic forgetting.

### 5.2 Optimiser & scheduler

- **Optimiser:** AdamW (β₁=0.9, β₂=0.999), weight decay 0.05 (not applied to 1-D params).
- **Scheduler (phase 2):** linear warm-up for 2 epochs → cosine annealing to 0.
- **Gradient clipping:** global norm ≤ 1.0.
- **AMP:** bf16 where supported (A100), fp16 + GradScaler otherwise.

### 5.3 Loss

**Focal loss** (γ=2, α=inverse class frequency). The focal term down-weights easy examples so the model focuses on hard, ambiguous leaf images. The inverse-frequency α corrects for any class imbalance among the 8 species.

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

All augmentation runs **on-GPU** in `src/augment.py` to avoid CPU bottlenecks.

### 5.5 Evaluation

- **Primary metric:** validation macro-F1 (equal weight per class).
- **Early stopping:** patience 8 epochs on val macro-F1.
- **Test-time augmentation (TTA):** average softmax over {original, hflip, vflip, hflip+vflip}.
- The test set is used **exactly once** in notebook 05 after all training decisions are finalised.

---

## 6. Quick Reference

```python
from src.config import load_config
from src.models import build_model

cfg = load_config()
model = build_model(cfg, "efficientnetv2s", device=device)
# backbone: tf_efficientnetv2_s.in21k_ft_in1k
# num_features: 1280
# use_cbam: False
```

**Checkpoint:** `checkpoints/efficientnetv2s_best.pt`  
**History CSV:** `results/metrics/efficientnetv2s_history.csv`  
**Run meta:** `results/metrics/efficientnetv2s_run_meta.json`

---

## 7. References

- Tan, M., & Le, Q. (2021). *EfficientNetV2: Smaller Models and Faster Training.* ICML 2021.
- Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks.* CVPR 2018.