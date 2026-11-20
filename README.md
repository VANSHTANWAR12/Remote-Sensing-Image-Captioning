# Retrieval-Enhanced HATNet: Remote Sensing Image Captioning

This repository contains the implementation, datasets, configurations, and evaluation benchmarks for **Retrieval-Enhanced HATNet** (Hierarchical Attention Transformer Network with ViPCap-style Visual Memory Prompting) applied to remote sensing image captioning.

The project evaluates performance across multiple data scales on the **RSICD** (Remote Sensing Image Captioning Dataset) benchmark—**3,000 (3k)**, **5,000 (5k)**, and **8,000 (8k)** images—as well as the **UCM** (UC Merced Land Use) dataset (`vansh_ucm`).

---

## 📌 Architecture Highlights

- **Vision Backbone**: Pretrained `ViT-H-14` (Vision Transformer Huge), extracting multi-scale feature representations from intermediate blocks `[10, 21, 31]` concatenated into a 3,840-dimensional visual feature token space across 256 patches.
- **Multimodal Retrieval Memory (ViPCap)**: Builds a visual-semantic memory cache from the training split. During inference and training, the top-\(k\) nearest semantic neighbors are retrieved using cosine similarity to formulate 16 learnable visual prompt tokens.
- **Caption Generation Decoder**: Standard autoregressive Transformer decoder (3 encoder layers, 3 decoder layers, 8 attention heads, $d_{\text{model}} = 512$, $d_{\text{ff}} = 2048$, beam search width = 3).
- **Ablation Studies**:
  - **Model A (Baseline HATNet)**: Standard hierarchical transformer captioning without retrieval augmentation.
  - **Model B (Retrieval Only)**: Memory-retrieved nearest-neighbor visual context without full prompt tuning.
  - **Model C (Full ViPCap)**: Full multi-scale prompt-guided visual retrieval-enhanced HATNet.

---

## 📂 Repository Structure

```text
vansh/
├── 3k/                         # RSICD - 3,000 Image Subset Experiment
│   ├── HATnet.ipynb            # Interactive training and evaluation notebook
│   ├── configuration.json      # Experiment hyperparameters & data paths
│   ├── split_manifest.json     # 2,400 Train / 300 Val / 300 Test image splits
│   ├── test_metrics.json       # Evaluated test metrics (BLEU, METEOR, ROUGE-L, CIDEr)
│   ├── run_summary.txt         # Best epochs & metric comparison for Models A, B, C
│   └── data/                   # Dataset manifests and RSICD captions
│
├── 5k/                         # RSICD - 5,000 Image Subset Experiment
│   ├── HATnet (2).ipynb        # Training notebook
│   ├── configuration.json      # Experiment hyperparameters
│   ├── split_manifest.json     # 4,000 Train / 500 Val / 500 Test image splits
│   ├── test_metrics.json       # Evaluated test metrics
│   ├── run_summary.txt         # Metric comparison for Models B, C
│   └── data/                   # Dataset manifests
│
├── 8k/                         # RSICD - 8,000 Image Full-Scale Experiment
│   ├── HATnet (3).ipynb        # Training notebook with full-scale dataset
│   ├── configuration.json      # 7,000 Train / 500 Val / 500 Test configuration
│   ├── split_manifest.json     # Data split manifest
│   ├── retrieval_memory_cache.pt # Cached visual retrieval memory bank
│   ├── feature_cache_metadata.json # ViT-H-14 feature extraction cache schema
│   ├── test_metrics.json       # Complete test set scores
│   ├── run_summary.txt         # Metric comparison for Models A, B, C
│   └── data/                   # RSICD dataset splits (dataset_8000.json)
│
├── vansh_ucm/                  # UCM (UC Merced) Land Use Caption Dataset
│   ├── ucm_configuration.json  # 1,680 Train / 210 Val / 210 Test configuration
│   ├── ucm_split_manifest.json # UCM split manifest (2,100 total images)
│   ├── ucm_test_metrics.json   # Full metric breakdown on UCM
│   └── ucm_run_summary.txt     # Benchmark summary for Models A, B, C
│
├── .gitignore                  # Ignores large model weights (.pth, .ckpt) & caches
└── README.md                   # Project documentation
```

---

## 📊 Experimental Results

All experiments were evaluated using standard image captioning metrics: **BLEU-1 to BLEU-4**, **METEOR**, **ROUGE-L**, and **CIDEr** under Karpathy split protocols.

### 1. RSICD Data Scaling Comparison (`3k` vs `5k` vs `8k`)

As the training set expands from 3,000 images to 8,000 images, feature representation richness and retrieval memory density scale significantly, driving strong performance gains across all metrics.

#### **Model C: Full ViPCap Performance across Scales**

| Metric | 3k Images (2.4k Train) | 5k Images (4k Train) | 8k Images (7k Train) | $\Delta$ (3k $\rightarrow$ 8k) |
| :--- | :---: | :---: | :---: | :---: |
| **BLEU-1** | 0.6981 | 0.7036 | **0.7365** | **+3.84%** |
| **BLEU-2** | 0.5831 | 0.5813 | **0.6248** | **+4.17%** |
| **BLEU-3** | 0.4990 | 0.4921 | **0.5421** | **+4.31%** |
| **BLEU-4** | 0.4326 | 0.4237 | **0.4793** | **+4.67%** |
| **METEOR** | 0.6082 | 0.6016 | **0.6350** | **+2.68%** |
| **ROUGE-L** | 0.6040 | 0.5901 | **0.6360** | **+3.20%** |
| **CIDEr**  | 2.2576 | 2.2749 | **2.5946** | **+0.337** |

---

### 2. Ablation Study on 8k RSICD (Full Scale)

Comparison between baseline HATNet, retrieval-only memory, and full ViPCap prompt tuning:

| Model Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Baseline HATNet)** | 0.7291 | 0.6130 | 0.5277 | 0.4640 | 0.6198 | 0.6257 | 2.5447 |
| **Model B (Retrieval Only)** | 0.7252 | 0.6141 | 0.5309 | 0.4681 | 0.6220 | 0.6330 | 2.5634 |
| **Model C (Full ViPCap)** | **0.7365** | **0.6248** | **0.5421** | **0.4793** | **0.6350** | **0.6360** | **2.5946** |

---

### 3. Ablation Study on UCM Dataset (`vansh_ucm`)

Benchmark on the 2,100 image UC Merced Land Use dataset (1,680 train / 210 val / 210 test):

| Model Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Baseline HATNet)** | 0.9103 | 0.8723 | 0.8385 | 0.8101 | 0.8758 | 0.8826 | 4.2364 |
| **Model B (Retrieval Only)** | **0.9318** | **0.9060** | **0.8806** | **0.8577** | **0.9009** | **0.9054** | 4.4780 |
| **Model C (Full ViPCap)** | 0.9275 | 0.9013 | 0.8760 | 0.8533 | 0.8954 | 0.9000 | **4.4813** |

---

## ⚙️ Hyperparameters & Configuration Overview

| Parameter | 3k RSICD | 5k RSICD | 8k RSICD | UCM (`vansh_ucm`) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Images** | 3,000 | 5,000 | 8,000 | 2,100 |
| **Train / Val / Test Split** | 2400 / 300 / 300 | 4000 / 500 / 500 | 7000 / 500 / 500 | 1680 / 210 / 210 |
| **Vision Backbone** | ViT-H-14 | ViT-H-14 | ViT-H-14 | ViT-H-14 |
| **Extraction Blocks** | [10, 21, 31] | [10, 21, 31] | [10, 21, 31] | [10, 21, 31] |
| **Feature Dim (`ATT_FEAT_SIZE`)** | 3,840 | 3,840 | 3,840 | 3,840 |
| **Number of Patches** | 256 | 256 | 256 | 256 |
| **Retrieval Top-K** | 4 | 5 | 6 | 6 |
| **Prompt Tokens** | 16 | 16 | 16 | 16 |
| **Batch Size** | 16 | 16 | 16 | 16 |
| **Optimizer & LR** | AdamW ($3 \times 10^{-4}$) | AdamW ($3 \times 10^{-4}$) | AdamW ($3 \times 10^{-4}$) | AdamW ($3 \times 10^{-4}$) |
| **Decoder Architecture** | 3 Enc / 3 Dec | 3 Enc / 3 Dec | 3 Enc / 3 Dec | 3 Enc / 3 Dec |
| **Beam Size** | 3 | 3 | 3 | 3 |
| **Max Caption Length** | 20 | 20 | 20 | 20 |

---

## 🚀 Execution & Workflow

### 1. Environment Setup
Make sure PyTorch and required evaluation libraries are installed:
```bash
pip install torch torchvision numpy scipy nltk scikit-learn
```

### 2. Running Experiments via Notebooks
Each folder contains its self-contained Jupyter notebook to run feature caching, retrieval memory creation, training, and Karpathy metric evaluation:
- For 3k RSICD: Open and execute `3k/HATnet.ipynb`
- For 5k RSICD: Open and execute `5k/HATnet (2).ipynb`
- For 8k RSICD: Open and execute `8k/HATnet (3).ipynb`
- For UCM: Open and run the corresponding script / notebook in `vansh_ucm/`

### 3. Outputs & Metrics
Each run generates:
- `configuration.json`: Full serialized hyperparameters.
- `split_manifest.json`: Exact image IDs in train, val, and test splits for reproducibility.
- `training_history.json`: Loss and validation scores across epochs.
- `test_predictions.json`: Generated captions vs ground truth references.
- `test_metrics.json` & `run_summary.txt`: Official COCO/Karpathy evaluation scores.
