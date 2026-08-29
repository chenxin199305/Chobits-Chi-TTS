#!/bin/bash
# 启动小叽 TTS HTTP 服务 (GPT-SoVITS api_v2, v2Pro + chi e10 权重)
# 用法: bash tools/start_tts_api.sh [端口, 默认 9880]
# 环境变量 MINICONDA 可覆盖 Miniconda 安装路径 (默认 $HOME/miniconda3)
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GS_ROOT="$REPO_ROOT/GPT-SoVITS"
PORT="${1:-9880}"
MINICONDA="${MINICONDA:-$HOME/miniconda3}"

# 推理配置 (随 GPT-SoVITS 目录可能被重建, 不存在则按本仓库路径重新生成)
CONFIG="$GS_ROOT/GPT_SoVITS/configs/tts_infer_chi.yaml"
if [ ! -f "$CONFIG" ]; then
    cat > "$CONFIG" <<EOF
custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cuda
  is_half: true
  t2s_weights_path: $REPO_ROOT/models/chi-e10.ckpt
  version: v2Pro
  vits_weights_path: $REPO_ROOT/models/chi_e10_s1210.pth
EOF
fi

# shellcheck disable=SC1091
source "$MINICONDA/etc/profile.d/conda.sh"
conda activate GPTSoVits

cd "$GS_ROOT"
export PYTHONPATH="$GS_ROOT:$GS_ROOT/GPT_SoVITS:$GS_ROOT/GPT_SoVITS/BigVGAN"
# conda libstdc++ (pyopenjtalk 需要 GLIBCXX_3.4.29) + torchcodec 需要的 npp 动态库
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/npp/lib"
export version=v2Pro

exec python api_v2.py \
  -c "$CONFIG" \
  -a 0.0.0.0 \
  -p "$PORT"
