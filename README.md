# NeuroGraph-X: A Hybrid CNN-GNN Framework for Alzheimer's Detection

[cite_start]**Authors:** Sagnik Pal [cite: 2][cite_start], Silajeet Banerjee [cite: 3]  
[cite_start]**Date:** February 7, 2026 [cite: 4]

## 🧠 Project Overview
[cite_start]NeuroGraph-X is a hybrid deep learning framework designed to detect Alzheimer's Disease (AD) from structural MRI scans while providing natural language explanations for its predictions[cite: 7]. 

[cite_start]Unlike traditional "black-box" CNNs, this project models the brain as a graph connectome and uses a Large Language Model (LLM) to translate technical findings into clinically meaningful reports[cite: 11].

## 🏗️ Architecture
[cite_start]The framework consists of two main modules[cite: 27]:

### Module A: Vision & Graph Learning
1.  [cite_start]**3D Feature Extraction:** Uses a 3D CNN (e.g., ResNet/DenseNet) to extract features from MRI volumes preprocessed with MONAI[cite: 29, 30].
2.  [cite_start]**Brain Graph Construction:** Maps brain regions (nodes) using the AAL atlas and models connectivity (edges) based on feature similarity[cite: 31, 33].
3.  [cite_start]**GNN Classification:** A Graph Attention Network (GAT) classifies subjects as AD or Healthy, learning attention weights for critical connections[cite: 35, 37].

### Module B: Explainability
1.  [cite_start]**Symbolic Extraction:** Identifies the top-k disrupted brain connections based on GNN attention weights[cite: 39].
2.  [cite_start]**LLM Generation:** A local LLM (Mistral-7B/LLaMA-3) generates a clinical narrative explaining *why* the model made its prediction[cite: 41, 42].

## 📂 Directory Structure
```text
neurograph-x/
├── data/               # Raw and processed MRI data (git-ignored)
├── notebooks/          # Experimentation notebooks
├── src/
│   ├── vision/         # 3D CNN and MONAI pipelines
│   ├── graph/          # PyTorch Geometric models
│   └── explain/        # LangChain and LLM logic
└── models/             # Saved model weights (git-ignored)
