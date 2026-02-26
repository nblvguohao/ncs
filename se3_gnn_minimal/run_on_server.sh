#!/bin/bash
# GPCR SE(3)-GNN 项目远程服务器运行脚本
# 使用前请先解压项目包并 cd 到项目根目录
set -euo pipefail

# -------------------------------------------------
# 1) 环境准备
# -------------------------------------------------
echo "[1] 创建并激活 conda 环境（如已有可跳过）"
conda create -n gpcr_gnn python=3.11 -y
conda activate gpcr_gnn

echo "[2] 安装依赖"
pip install -r requirements.txt

# -------------------------------------------------
# 2) 数据与模型准备（按需执行）
# -------------------------------------------------
echo "[3] 数据准备（如已有可跳过）"
python code/01_fetch_multispecies_gpcr.py
python code/02_download_af2_structures.py
python code/03_compute_esm2_embeddings.py
python code/04_build_3d_graphs.py

# -------------------------------------------------
# 3) 训练与评估（吃GPU/内存的步骤）
# -------------------------------------------------
echo "[4] Standard 训练（GPU，约 2-8 GB 显存，batch_size=2）"
# 如显存不足，可进一步降低 batch_size 到 1
python code/07_run_training.py --mode standard --epochs 30 --batch_size 2

echo "[5] Zero-shot 评估（GPU，每个子家族单独训练，显存需求同上）"
# 会遍历所有子家族，耗时较长；可先用 --epochs 30 快速验证
python code/07_run_training.py --mode zeroshot --epochs 50 --batch_size 2

# -------------------------------------------------
# 4) 结果分析（CPU）
# -------------------------------------------------
echo "[6] 结果汇总与分析（CPU）"
python code/analyze_results.py

echo "[7] 查看 NMI 等关键指标"
cat results/analysis_summary.json | python -m json.tool

echo "全部完成！"
