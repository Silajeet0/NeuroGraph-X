

**NeuroGraph-X**: A Hybrid CNN-GNN Framework for Alzheimer's Detection with Natural Language Explainability

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

---
## 📄 Abstract
Early detection of Alzheimer's Disease (AD) by neuroimaging remains a challenging problem due to the high dimensionality of brain data and the lack of clinically interpretable explanations. **NeuroGraph-X** is a hybrid deep learning framework that integrates 3D Convolutional Neural Networks (CNNs), Graph Neural Networks (GNNs), and Natural Language Processing (NLP) for accurate and explainable AD detection from structural MRI scans.

## 🏗️ Architecture
The framework consists of two main modules

### Module A: Vision and Graph-Based Learning
1.  **Feature Extraction:** - Structural T1-weighted 3D MRI scans are preprocessed using `MONAI` and `Nibabel`.
    - A pretrained 3D CNN (e.g., 3D-ResNet or DenseNet) extracts high-level feature representations.
    - **Atlas-based Parcellation:** Using the AAL brain atlas, we perform average pooling to produce fixed-length feature vectors for specific brain regions.
2.  **Graph Construction:** - Each brain region is modeled as a **node**.
    - **Edges** are defined based on feature similarity or physical distance, mimicking the brain's connectome.
3.  **Graph Classification:** - A GNN (Graph Attention Network or GraphSAGE) implemented in `PyTorch Geometric` classifies the graph.
    - **Attention Weights** are learned to highlight critical inter-regional connections.

### Module B: Natural Language Explainability
1.  **Symbolic Extraction:** The top-k most affected edges are identified based on attention weight degradation.
2.  **Language Generation:** - These symbolic findings are fed into a local Large Language Model (e.g., **Mistral-7B** or **LLaMA-3**) using `LangChain`.
    - The LLM generates a clinically grounded explanation describing connectivity disruptions and their cognitive implications.

## 🛠️ Tech Stack
- **Core:** Python, PyTorch, NumPy, Pandas
- **Medical Imaging:** MONAI, Nibabel, Nilearn
- **Graph Learning:** PyTorch Geometric (PyG)
- **LLM & NLP:** LangChain, HuggingFace Transformers, Llama-cpp-python (for local inference)

---

## 🗂️ Repository Structure
```text
neurograph-x/
├── data/
│   ├── raw/                  # Original MRI dataset (immutable)
│   ├── processed/            # Skull-stripped/normalized tensors
│   └── atlas/                # AAL atlas files
│
├── src/                      
│   ├── preprocessing/        # MONAI pipelines & Atlas mapping
│   ├── vision_module/        # 3D CNN Backbone & Feature Extractor
│   ├── graph_module/         # GNN Models (GAT/GraphSAGE)
│   └── explainability/       # LLM Prompting & LangChain Agents
│
├── notebooks/                # Jupyter notebooks for experimentation and data visualization
├── configs/                  # Configuration files (YAML/Hydra)
└── scripts/                  # Training and Inference scripts

```
---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Silajeet0/NeuroGraph-X.git
cd NeuroGraph-X
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Preprocessing: Run the data pipeline to convert raw MRI scans into graph data.

```bash
# Run experiments
python scripts/run_pipeline.py --config configs/preprocessing.yaml
```

## Training
Train the hybrid CNN-GNN model.

```bash
python scripts/train_gnn.py --config configs/model_gnn.yaml
```
## Generate Report: 
Run inference on a sample scan to get the prediction and LLM explanation.
```bash
python scripts/generate_report.py --input data/sample_scan.nii.gz
```

---

## 🤝 Contributing

Contributions, suggestions, and discussions are welcome.

1. Fork the repo  
2. Create a feature branch  
3. Submit a Pull Request  

---

## 👤 Author

**Silajeet Banerjee**

Computer Science Graduate 
M.Sc. Computer Science Student at RKMVERI

GitHub: https://github.com/Silajeet0

**Sagnik Pal**

Computer Science Graduate
M.Sc. Computer Science Student at RKMVERI

Github: https://github.com/Sagnik-2004


