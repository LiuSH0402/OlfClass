# OlfClass
# OlfClass — Structure-driven GNN for ORP Classification

<p align="center">
  <img src="figures/Framework%20Diagram.jpg" alt="Pipeline overview" width="900">
</p>


---

## Repository Layout



---

## Quickstart

### 1) Environment
- Python ≥ 3.10
- PyTorch + CUDA（与你的显卡环境匹配）
- PyTorch Geometric、transformers、biopython、scikit-learn、numpy、pandas、tqdm、loguru

示例安装（按需替换 CUDA/版本）：
```bash
# conda 示例
conda create -n olfclass python=3.10 -y
conda activate olfclass
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install transformers biopython scikit-learn numpy pandas tqdm loguru
