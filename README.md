# OlfClass — Structure-driven GNN for ORP Classification

<p align="center">
  <img src="figures/Framework%20Diagram.jpg" alt="Pipeline overview" width="900">
</p>

**OlfClass** is a structure-driven graph neural network (GNN) for classifying **olfactory-related proteins (ORPs)**.  
It fuses **ESMFold** 3D structures, **ProtT5** sequence embeddings, and **DSSP/geometry** features into a residue-level graph for downstream learning.

---

## 🗂 Repository Layout


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

## 🔮 Inference
Use the inference cell in **`train.ipynb`** to load new FASTA or precomputed features and export predictions (e.g., `preds.csv`).  
The evaluation cell reports AUC / PR-AUC / F1 / MCC on a held-out split.

---

## ✅ Reproducibility
- Fix a random seed (e.g., **42**) and repeat runs (e.g., **3** times) to report mean ± s.d.  
- For stricter generalization, consider **cluster-aware** splits (e.g., family hold-out or lower CD-HIT thresholds).

---

## 📜 References
- Song Y., Yuan Q., *et al.* Accurately predicting enzyme functions through geometric graph learning on ESMFold-predicted structures. **Nat Commun** (2024).  
- Yuan Q., *et al.* GPSFun: geometry-aware protein sequence function predictions with language models. **Nucleic Acids Research** (2024).

---

## 📝 Citation
If you use this repository, please cite the corresponding manuscript.

---

## 📄 License
Add a license (e.g., **MIT**) as `LICENSE` and reference it here.

---

## 📫 Contact
Maintainer: *your_email@domain.com* — Issues and PRs are welcome.

---

### 🧰 One-shot Quickstart (optional)
> Copy–paste the block below on **Linux** to create the env, install deps, and execute both notebooks end-to-end.  
> Adjust the CUDA wheel URLs to your system if needed.

```bash
# ONE-SHOT SETUP & RUN
set -e
conda create -n olfclass python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate olfclass

# Install PyTorch (change CUDA build if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install PyG wheels (match your torch/CUDA)
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
# Others
pip install transformers biopython scikit-learn numpy pandas tqdm loguru jupyter

# DSSP (Ubuntu) — skip if you already have mkdssp
sudo apt-get update && sudo apt-get install -y dssp || true

# Run notebooks non-interactively
jupyter nbconvert --to notebook --inplace --execute code/feature.ipynb
jupyter nbconvert --to notebook --inplace --execute code/train.ipynb
echo "Done. Check outputs in data/features/ and results/runs (if configured)."
