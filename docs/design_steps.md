# Plant Leaf Species Classification on the EgyPLI Dataset
**Group:** G6 | **Members:** 2021/e/025, 2021/e/024, 2021/e/035
**Repository:** [https://github.com/Kidhurshan/plant-leaf-classifier](https://github.com/Kidhurshan/plant-leaf-classifier)

## 1. Executive Summary & Objective
This report details the design and evaluation of three deep-learning models for classifying 8 plant species using the EgyPLI dataset (3,588 images). To ensure a rigorous comparison, all models shared an identical classification head and training pipeline. The proposed **CBAM-ConvNeXt** model achieved a perfect **100.00% accuracy (0 errors / 538)** on the held-out test set, outperforming EfficientNetV2-S (99.63%) and Swin-Small (99.63%) while using 43% fewer parameters than Swin-Small and converging the fastest. Rigorous auditing confirmed this result is legitimate and not a product of data leakage or overfitting.

## 2. Dataset & Preprocessing Pipeline
**Dataset:** The dataset contains 3,588 images across 8 species, with moderate class imbalance (e.g., Orange: 547, Tomato: 159). 

> *[INSERT IMAGE: Grid of sample images showing one example from each of the 8 classes]*

**Pipeline:** To eliminate IO bottlenecks, images were decoded once into a single GPU-resident `uint8` tensor. All augmentations (random-resized crop, rotation, flips, color jitter, random erasing, mixup/cutmix) are applied on the GPU on whole batches, yielding epoch times under 6 seconds on an NVIDIA A100 GPU.

## 3. Data Integrity & Leakage Prevention (Crucial Finding)
Initial tests showed 100% validation accuracy from Epoch 1, indicating data leakage. 
**The Cause:** Burst photographs—multiple near-identical shots of the same leaf taken within seconds—were being split across train and test sets by the naive random split (affecting 20.9% of the data).

> *[INSERT IMAGE: Visual example showing 3-4 burst photographs of the exact same leaf]*

**The Fix (Group-Aware Splitting):** We implemented a group-aware stratified algorithm that clusters images by leaf group (using filename suffixes) before splitting (70/15/15). 
**Verification:** A 64-bit perceptual hash (dHash) confirmed **0 / 538** test images had near-duplicates in the training set based on pixel similarity. All reported results were produced *after* this fix.

## 4. Model Architectures & Methodology
To ensure a controlled experiment, a **Byte-Identical Shared Head** (Global Average Pooling -> LayerNorm -> Dropout(0.3) -> Linear) was attached to three different backbones initialized with ImageNet-21k/22k weights:
1. **EfficientNetV2-S:** CNN baseline (20.2M params).
2. **Swin-Small:** Hierarchical Vision Transformer (48.8M params).
3. **CBAM-ConvNeXt (Proposed, 27.9M params):** Combines a ConvNeXt-Tiny backbone with Convolutional Block Attention Modules (CBAM). Because leaf species are distinguished by localized cues (venation, margins), CBAM suppresses background noise. 

> *[INSERT IMAGE: Block diagram of the CBAM-ConvNeXt architecture and the CBAM module]*

**Identity Initialization:** CBAM gates were initialized to zero, ensuring the module acts as a strict identity at step 1, preserving pretrained ImageNet features.

**Training Strategy:** 
- **Phase 1 (Warm-up):** Backbone frozen; only the head/CBAM trains for 5 epochs (LR: 1e-3).
- **Phase 2 (Fine-tune):** Unfrozen, 35 epochs (LR: 1e-4) with Layer-Wise Learning Rate Decay (0.85 per stage) to nudge rather than overwrite pretrained features. 
- **Loss:** Focal Loss with inverse-frequency weighting to handle the minority Tomato class.

## 5. Results & Evaluation
All models were evaluated on the test set **exactly once**.

| Model | Accuracy | Macro-F1 | Acc+TTA | Params | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EfficientNetV2-S | 0.9963 (2 err) | 0.9945 | 0.9981 | 20.2 M | 2m 26s (19 ep) |
| Swin-Small | 0.9963 (2 err) | 0.9964 | 0.9963 | 48.8 M | 56s (14 ep) |
| **CBAM-ConvNeXt** | **1.0000 (0 err)** | **1.0000** | **1.0000** | **27.9 M** | **1m 56s (13 ep)**|
| Ensemble | 1.0000 (0 err) | 1.0000 | - | 97.0 M | - |

> *[INSERT IMAGE: Grid of Confusion Matrices for the models evaluated on the test set]*

**Justification for 100% Accuracy (Absence of Overfitting):**
1. **Negative Train-Test Gap:** CBAM-ConvNeXt scored 0.9984 on train and 1.0000 on test. A memorized model cannot perform better on unseen data.
2. **Tiny Fitted Parameter Budget:** In Phase 1, only 0.36% of parameters were trainable, yet the model reached 1.0000 validation Macro-F1 while the 27.8M backbone remained completely frozen.
3. **Robustness:** Under heavy test-set distortion (augmentations active), CBAM-ConvNeXt maintained 1.0000 accuracy.
4. **Generalization:** All 3 models perfectly classified the minority Tomato class (111 train images), showing they learned robust features rather than memorizing data-rich classes.

## 6. Explainability (Grad-CAM & t-SNE)
- **Grad-CAM:** Heatmaps confirmed attention is localized on the leaf blade and venation, not on background soil or hands. 
- **t-SNE:** The 768-dimensional pooled features clustered into eight compact, widely separated groups with minimal overlap.

> *[INSERT IMAGE: Grid of Grad-CAM heatmaps showing attention on leaf venation]*
> *[INSERT IMAGE: t-SNE 2D scatter plot visualization of the 8 clusters]*

## 7. Conclusions & Reproducibility
The proposed CBAM-ConvNeXt model achieved 100% test accuracy (95% CI: 99.3%–100%) through efficient attention mechanisms and rigorous data leakage prevention. Reproducibility is guaranteed via a single YAML configuration, fixed random seeds, and deterministic evaluation preprocessing.
