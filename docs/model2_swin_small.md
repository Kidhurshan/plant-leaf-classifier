# Model 2 — Swin-Small Transformer

**timm checkpoint:** `swin_small_patch4_window7_224.ms_in22k_ft_in1k`  
**Role in the comparison:** Vision Transformer baseline — tests whether a self-attention-based architecture outperforms the CNN baseline on a small, fine-grained leaf dataset.

---

## 1. What Is Swin-Small?

Swin Transformer (Shifted Window Transformer) was introduced by Liu et al. (2021) as a general-purpose vision backbone that bridges the gap between CNN hierarchical features and the global-attention power of the original ViT. The "Small" variant balances accuracy and compute at a similar parameter count to ResNet-50.

The `ms_in22k` portion of the checkpoint name means the weights were trained using **multi-scale supervision on ImageNet-22k** — a more thorough pretraining regime than single-scale IN-21k. The `ft_in1k` suffix means it was subsequently fine-tuned on ImageNet-1k before we receive it. This is the same two-stage pretraining philosophy as Model 1, but executed with a transformer architecture.

---

## 2. Architecture

### 2.1 Macro-structure

```
Input (224×224×3)
│
├── Patch Partition + Linear Embedding
│       split image into non-overlapping 4×4 patches → [B, 56×56, 96]
│
├── Stage 1  — 2 × Swin Blocks (W-MSA → SW-MSA)         [56×56, 96ch]
│
├── Patch Merging (stride 2 downsampling) → [28×28, 192ch]
│
├── Stage 2  — 2 × Swin Blocks                           [28×28, 192ch]
│
├── Patch Merging → [14×14, 384ch]
│
├── Stage 3  — 18 × Swin Blocks                          [14×14, 384ch]
│
├── Patch Merging → [7×7, 768ch]
│
└── Stage 4  — 2 × Swin Blocks                           [7×7, 768ch]
         └── output feature map [B, 768, 7, 7] (after layout conversion)
```

Stage depths: **[2, 2, 18, 2]**, channel widths: **[96, 192, 384, 768]**.

The feature map is in **NHWC** layout inside timm (tokens ordered spatially). Our `src/models.py::_to_nchw()` normalises it to NCHW before handing it to the shared head.

### 2.2 Patch Partition and Linear Embedding

The very first operation converts the image from a pixel grid into a **token sequence**:

1. Divide the 224×224 image into non-overlapping **4×4 patches** → 56×56 = 3,136 tokens per image.
2. Flatten each 4×4×3 = 48-dimensional patch.
3. Project to the embedding dimension (96) with a learnable linear layer.

This is different from CNNs: there is no stem convolution with strided filters; instead the entire spatial grid is tokenised upfront.

### 2.3 Swin Transformer Block

Each Swin Block applies **Window-based Multi-head Self-Attention (W-MSA)** and is always paired with a **Shifted-Window MSA (SW-MSA)** block:

```
Input x
│
├── LayerNorm
├── W-MSA (attention within non-overlapping M×M windows, M=7)
├── + residual
│
├── LayerNorm
├── MLP (2-layer FF, hidden dim = 4× embedding)
└── + residual

(next block)
├── LayerNorm
├── SW-MSA (windows shifted by ⌊M/2⌋=3 to enable cross-window communication)
├── + residual
├── LayerNorm
├── MLP
└── + residual
```

**Window size M=7** means attention is computed over 7×7 = 49 tokens at a time — not over the full 3,136-token sequence. This keeps the quadratic attention cost at O(M² × H×W/M²) = O(H×W), i.e. **linear in the number of tokens**, unlike vanilla ViT which would be O((H×W)²).

The shifting mechanism (SW-MSA) is the key innovation: by shifting windows by half their size every other block, each pair of blocks achieves cross-window information flow without extra parameters.

### 2.4 Patch Merging (hierarchical downsampling)

Between stages, spatial resolution is halved and channel depth doubled:

```
[B, H, W, C] → concat 2×2 neighbour patches → [B, H/2, W/2, 4C] → Linear → [B, H/2, W/2, 2C]
```

This gives Swin a feature pyramid analogous to a CNN's staged downsampling, enabling multi-scale representations needed for dense vision tasks as well as classification.

### 2.5 Relative Position Bias

Unlike the original ViT which adds absolute positional embeddings, Swin uses **relative position bias** inside each attention computation:

```
Attention(Q, K, V) = softmax( QKᵀ / √d  +  B )  V
```

where **B** is a learnable bias table indexed by the relative displacement between every pair of tokens in the window. This is more flexible than absolute embeddings and transfers better across different input resolutions.

### 2.6 Shared head (identical across all three models)

```python
GAP(feature_map)        # [B, 768, 7, 7] → [B, 768]
LayerNorm(768)
Dropout(0.3)
Linear(768 → 8)         # 8 EgyPLI species
```

**Total parameters (backbone + head):** approximately **49.7 M**  
**Feature dimension passed to the head:** 768

---

## 3. Pretraining & Transfer Learning

| Stage | Dataset | Supervision | Notes |
|-------|---------|-------------|-------|
| Pretrain | ImageNet-22k | Multi-scale | 14 M images, 22,000 classes |
| Fine-tune | ImageNet-1k | Standard | 1.28 M images |
| Task fine-tune | EgyPLI | Two-phase | ~3,588 images, 8 classes |

The `ms` (multi-scale) pretraining helps the model learn scale-invariant features, which is valuable for leaves where the camera distance varies across images in the EgyPLI dataset.

---

## 4. How It Fits the EgyPLI Dataset

### 4.1 Dataset at a glance

| Property | Value |
|----------|-------|
| Species | Apple, Berry, Fig, Guava, Orange, Palm, Persimmon, Tomato |
| Total images | ~3,588 |
| Input resolution | 224 × 224 (Swin is fixed at 224 due to the window/position-bias tables) |
| Split | 70 / 15 / 15 (train / val / test), group-aware stratified |

### 4.2 Why Swin-Small is a good fit

- **Global context from the first stage.** Even Stage 1 (at 56×56 tokens) can relate any token to every other token within a 7×7 window. By Stage 4 (7×7 tokens) each token attends across the entire spatial map. For leaves, where the overall shape silhouette (e.g. palm fronds vs. round fig leaves) is highly diagnostic, this global context is a significant advantage over local CNN kernels.
- **Scale-invariant pretraining.** The `ms_in22k` checkpoint was exposed to objects at many scales during pretraining. Leaves in the EgyPLI dataset are photographed at varying camera distances, so scale robustness matters.
- **Hierarchical features.** Swin builds a genuine feature pyramid (56→28→14→7), so it captures both fine-grained venation texture (early stages) and holistic shape (late stages) — similar to a CNN but with global attention at each level.
- **Fixed 224×224 input.** All three models are trained at 224×224 for a controlled comparison. For Swin this is not a limitation — its window size (7) and position bias tables were designed for this resolution.

### 4.3 Potential limitations

- **Parameter count.** At ~49.7 M parameters, Swin-Small is more than double the size of EfficientNetV2-S (~21.5 M). With only ~2,512 training images this raises the risk of overfitting, which is why the shared head includes Dropout(0.3), Focal loss, and aggressive augmentation.
- **No explicit spatial gate.** Self-attention computes soft weights over all tokens, but it does so uniformly. It does not apply the sequential channel-then-spatial gating that CBAM provides in Model 3. Whether attention weights naturally focus on diagnostically relevant leaf regions depends entirely on what the backbone learned during pretraining.
- **Computational cost.** Stage 3 has 18 transformer blocks — roughly 9× as many as the other stages combined. Training throughput (images/second) is noticeably lower than EfficientNetV2-S on the same GPU.

---

## 5. Training Protocol

All three models follow the exact same protocol in `src/engine.py`.

### 5.1 Two-phase fine-tuning

| Phase | Epochs | Backbone | LR | Mix |
|-------|--------|----------|----|-----|
| 1 (warm-up head) | 5 | **Frozen** | 1 × 10⁻³ | Off |
| 2 (full fine-tune) | up to 35 (early-stop) | **Unfrozen** | 1 × 10⁻⁴ | On |

Phase 1 is especially important for Swin: the pretrained LayerNorm and attention weights are tuned for ImageNet-1k class tokens. Letting a large gradient flow through 22 stages from a randomly initialised head would corrupt these features. Freezing the backbone and training only the head first dramatically stabilises early training.

### 5.2 Optimiser & scheduler

- **Optimiser:** AdamW (β₁=0.9, β₂=0.999), weight decay 0.05 (biases and 1-D norm params excluded).
- **LLRD (Layer-wise LR Decay):** backbone parameters receive a per-depth LR multiplier of 0.85ⁿ, where n counts from the deepest layer outward. This means earlier transformer blocks (which capture low-level features unlikely to need much adaptation) update more slowly than deeper blocks.
- **Scheduler (phase 2):** 2-epoch linear warm-up → cosine annealing to 0.
- **Gradient clipping:** global norm ≤ 1.0.
- **AMP:** bf16 / fp16 + GradScaler depending on GPU support.

### 5.3 Loss

**Focal loss** (γ=2, α=inverse class frequency). Focal loss was originally proposed for dense object detection where easy negatives dominate; the same principle applies here — most leaf images are "easy" once the backbone is warm, and focal loss keeps the gradient signal focused on the handful of genuinely hard examples (e.g. visually similar berry/tomato leaves).

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
- **Early stopping:** patience 8 epochs.
- **TTA:** softmax averaged over {original, hflip, vflip, hflip+vflip}.
- Test set used exactly once in notebook 05.

---

## 6. Quick Reference

```python
from src.config import load_config
from src.models import build_model

cfg = load_config()
model = build_model(cfg, "swin_small", device=device)
# backbone: swin_small_patch4_window7_224.ms_in22k_ft_in1k
# num_features: 768
# use_cbam: False
```

**Checkpoint:** `checkpoints/swin_small_best.pt`  
**History CSV:** `results/metrics/swin_small_history.csv`  
**Run meta:** `results/metrics/swin_small_run_meta.json`

---

## 7. References

- Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
- Dosovitskiy, A., et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.