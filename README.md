# OlfClass — Structure-driven GNN for ORP Classification

<p align="center">
  <img src="figures/Framework%20Diagram.jpg" alt="Pipeline overview" width="900">
</p>

**OlfClass** is a structure-driven graph neural network (GNN) for classifying **olfactory-related proteins (ORPs)**.  
It fuses **ESMFold** 3D structures, **ProtT5** sequence embeddings, and **DSSP/geometry** features into a residue-level graph for downstream learning.

---

## ⚙️ Environment
- Python ≥ 3.10
- PyTorch (GPU build matching your CUDA)
- PyTorch Geometric
- Transformers, Biopython, scikit-learn, NumPy, Pandas, tqdm, loguru
- DSSP (`mkdssp`) for secondary-structure features

> Linux/macOS can install DSSP via `sudo apt install dssp` / `brew install dssp`,  
> or use the bundled `code/mkdssp` + `code/dssp.sh`.

---

## 🧪 Data Preparation
- **Source**: UniProt **Reviewed (Swiss-Prot)** entries.
- **Positives**: annotation terms — **OBPs**, **ORs**, **ORCOs**, **PBPs**.
- **Negatives**: Reviewed proteins **without** “olfactory/odorant/pheromone” in name/function/GO.
- **Redundancy removal (CD-HIT)**: applied **separately** to positives and negatives with  
  - **c = 0.80** (minimum pairwise identity to cluster; keep one representative),  
  - **n = 5** (k-mer word size used by CD-HIT’s index); other parameters at defaults.
- **Split**: random **8:2** train/test on the non-redundant set.

Expected directories:



---

## 🧱 Features & Graph Construction
Open **`code/feature.ipynb`** and run all cells. It will:
1) read FASTA and load **ESMFold** PDBs → `data/structures/`  
2) compute **DSSP** via `mkdssp`  
3) derive **geometric** features (distances/angles/contacts)  
4) extract **ProtT5** embeddings  
5) build residue-level graphs with **node = 1217 dims** and **edge = 450 dims**, caching to `data/features/`

> Non-interactive run (optional):
> ```bash
> jupyter nbconvert --to notebook --inplace --execute code/feature.ipynb
> ```

---

## 🚀 Training & Evaluation
Open **`code/train.ipynb`** and run all cells. Default settings (as in the manuscript):

- **Backbone**: 2-layer GNN, hidden **256**  
- **Dropout**: **0.5** (between the two linear layers of each GNN FFN block; and **before** the final classifier)  
- **Initialization**: **Xavier-uniform** (gain **√2** for ReLU/GELU), **biases = 0**, **LayerNorm weight/bias = 1/0**  
- **Optimizer**: **SGD**, `lr = 1e-3`, `weight_decay = 1e-5`  
- **Loss**: **BCE with logits**, `batch_size = 2`, `max_epochs = 100`  
- **Early stopping**: validation **AUC**, `patience = 10`, `min_delta = 1e-4` (save best checkpoint)  
- **Device**: NVIDIA GeForce **RTX 4090**

> Non-interactive run (optional):
> ```bash
> jupyter nbconvert --to notebook --inplace --execute code/train.ipynb
> ```

---

## 📫 Contact
Maintainer: *liush0402@163.com* — Issues and PRs are welcome.

---
