# 🧠 MIRAGE: MRI-Informed Graph Attention Network for Alzheimer’s Classification

## 🧠 Overview

This project presents an advanced end-to-end machine learning pipeline for Alzheimer’s disease classification using MRI-derived brain features and clinical data.

The work evolves from a baseline Graph Neural Network (GNN) model into a **MIRAGE (Multi-modal Integrated Representation for Alzheimer’s Graph Encoding)** framework, combining self-supervised learning and graph-based modeling for improved performance and interpretability.

---

## 📊 Problem Statement

Early detection of Alzheimer’s disease is challenging due to:

* Subtle structural changes in brain MRI
* Heterogeneous data sources (imaging + clinical)
* Lack of interpretability in deep learning models

Traditional approaches treat patients independently and fail to capture relationships between brain components.

---

## 💡 Proposed Solution

We propose a **graph-based deep learning framework** that:

* Models relationships between brain tissues (CSF, Gray Matter, White Matter)
* Integrates MRI-derived features with clinical data
* Uses attention mechanisms for explainability
* Extends into MIRAGE using representation learning

---

## 🚀 MIRAGE Framework

### 🔬 Key Enhancements over Baseline GNN:

* **2.5D MRI Slice Extraction** (multi-plane representation)
* **Variational Autoencoder (VAE)** for representation learning
* **Learned embeddings (64-dim)** instead of handcrafted features
* **Graph Attention Network (GAT)** with attention-based reasoning
* **Fusion of imaging + clinical features**
* **Explainability via attention weights**

---

## ⚙️ Pipeline

### 🔹 Baseline Pipeline

```text
MRI + Clinical Data
        ↓
Feature Extraction (CSF, GM, WM volumes)
        ↓
Graph Construction
        ↓
Graph Attention Network (GAT)
        ↓
Prediction + Explainability
```

### 🔹 MIRAGE Pipeline (Enhanced)

```text
MRI Scans (.img/.hdr)
        ↓
2.5D Slice Extraction (Axial, Coronal, Sagittal)
        ↓
Variational Autoencoder (VAE)
        ↓
Latent Embeddings (64-dim per tissue)
        ↓
Graph Construction (Nodes = tissues)
        ↓
GAT + Clinical Feature Fusion
        ↓
Prediction + Explainability
```

---

## 🧠 Model Architecture

### 🔹 Baseline GAT

* Node features: Brain volumes + clinical features
* 2 GAT layers with multi-head attention
* Global pooling → classification

### 🔹 MIRAGE Model

* **Graph Stream:**
  GATConv → GATConv → Global Mean Pooling

* **Clinical Stream:**
  Normalized clinical feature vector

* **Fusion:**
  Concatenation of graph embedding + clinical features

* **Classifier:**
  Fully connected neural network

---

## 📈 Results

| Model                 | ROC-AUC                                       |
| --------------------- | --------------------------------------------- |
| Logistic Regression   | ~0.79                                         |
| SVM                   | ~0.77                                         |
| Random Forest         | ~0.76                                         |
| **GAT (Baseline)**    | **0.836**                                     |
| **MIRAGE (Proposed)** | **Improved / Robust Representation Learning** |

---

## 🔍 Key Insights

* Graph-based modeling captures relationships between brain regions
* CSF-related features are highly influential (attention analysis)
* Combining MRI + clinical data significantly improves performance
* Learned embeddings (VAE) provide richer representations than handcrafted features

---

## 🧪 Ablation Study

Removing clinical features led to a significant drop in performance (~13% ROC-AUC), confirming their importance in Alzheimer’s prediction.

---

## 📂 Project Structure

```text
01_build_master_index.py       # Data indexing
02_tabular_baselines.py        # Classical ML models
03_extract_nodes.py            # MRI feature extraction
04_build_graphs.py             # Graph construction
05_train_gnn.py                # Baseline GAT training
06_interpret_attention.py      # Explainability
07_generate_figures.py         # Visualizations
08_ablation_study.py           # Feature importance

09a_extract_slices.py          # 2.5D slice extraction
09b_vae_train.ipynb            # VAE training
09c_build_graphs_v2.py         # Graphs with embeddings
09d_train_mirage.py            # MIRAGE model training
09e_finetune_vae.py            # VAE fine-tuning
```

---

## 🚀 How to Run

### 🔹 Baseline Pipeline

```bash
python 01_build_master_index.py
python 02_tabular_baselines.py
python 03_extract_nodes.py
python 04_build_graphs.py
python 05_train_gnn.py
python 06_interpret_attention.py
python 07_generate_figures.py
python 08_ablation_study.py
```

### 🔹 MIRAGE Pipeline

```bash
python 09a_extract_slices.py
# Run VAE training notebook (09b)
python 09c_build_graphs_v2.py
python 09d_train_mirage.py
python 09e_finetune_vae.py
```

---

## 🛠 Tech Stack

* Python
* PyTorch
* PyTorch Geometric
* Scikit-learn
* Nibabel
* NumPy / Pandas

---

## 🎯 Contributions

* Developed a complete ML pipeline for Alzheimer’s classification
* Introduced graph-based modeling for brain structure relationships
* Extended the system into MIRAGE using self-supervised learning
* Provided explainability using attention mechanisms
* Validated performance using cross-validation and ablation studies

---

## 👨‍💻 Authors

* Vedant Pandey
* Yuval Shah

---
