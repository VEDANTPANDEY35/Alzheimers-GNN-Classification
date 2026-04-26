# Alzheimer’s Disease Classification using Graph Attention Networks (GAT)

## 🧠 Overview

This project presents an end-to-end machine learning pipeline for Alzheimer’s disease classification using MRI-derived brain features and clinical data.

The core contribution is the use of **Graph Attention Networks (GAT)** to model relationships between brain regions and improve both performance and interpretability over traditional models.

---

## 📊 Problem Statement

Early detection of Alzheimer’s disease is challenging due to complex brain patterns and heterogeneous data sources. Traditional models treat patients independently and lack interpretability.

This project addresses this by:

* Modeling relationships between brain components
* Integrating MRI and clinical data
* Providing explainable predictions using attention mechanisms

---

## ⚙️ Pipeline

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
        ↓
Evaluation + Ablation Study
```

---

## 🧠 Model Architecture

* Graph Attention Network (GAT)
* Multi-head attention
* Global pooling
* Binary classification (AD vs Non-AD)

---

## 📈 Results

| Model               | ROC-AUC   |
| ------------------- | --------- |
| Logistic Regression | ~0.79     |
| SVM                 | ~0.77     |
| Random Forest       | ~0.76     |
| **GAT (Proposed)**  | **0.836** |

---

## 🔍 Key Insights

* Graph-based learning improves performance over traditional ML
* CSF-related features are most influential (attention analysis)
* Combining MRI + clinical data is critical (validated via ablation study)

---

## 🧪 Ablation Study

Removing clinical features caused a significant drop in performance (~13% ROC-AUC), demonstrating their importance.

---

## 🚀 How to Run

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

---

## 🛠 Tech Stack

* Python
* PyTorch
* PyTorch Geometric
* Scikit-learn
* Nibabel

---

## 👨‍💻 Author

Vedant Pandey
