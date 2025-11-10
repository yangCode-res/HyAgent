#!/usr/bin/env bash
set -euo pipefail

# ===== 配置 =====
MODEL_ID="dmis-lab/biobert-base-cased-v1.1"
HF_MIRROR="https://hf-mirror.com"          # 国内镜像域名
BASE_DIR="/home/nas2/path/models"          # 你想放模型的目录
TARGET_DIR="${BASE_DIR}/biobert-base-cased-v1.1"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "[INFO] Downloading BioBERT from $HF_MIRROR/$MODEL_ID"
echo "[INFO] Target dir: $TARGET_DIR"

download () {
  local file="$1"
  local url="${HF_MIRROR}/${MODEL_ID}/resolve/main/${file}"
  echo "[INFO] -> ${file}"
  # -L 跟随重定向，-# 显示进度条
  curl -# -L "$url" -o "$file"
}

# 必需文件
download "config.json"
download "vocab.txt"
download "pytorch_model.bin"

# 可选：补充 tokenizer 配置，避免 transformers 报告 warning
if [ ! -f "tokenizer_config.json" ]; then
  cat > tokenizer_config.json <<EOF
{
  "do_lower_case": false,
  "model_max_length": 512,
  "tokenizer_class": "BertTokenizer"
}
EOF
  echo "[INFO] tokenizer_config.json created."
fi

if [ ! -f "special_tokens_map.json" ]; then
  cat > special_tokens_map.json <<EOF
{
  "unk_token": "[UNK]",
  "sep_token": "[SEP]",
  "pad_token": "[PAD]",
  "cls_token": "[CLS]",
  "mask_token": "[MASK]"
}
EOF
  echo "[INFO] special_tokens_map.json created."
fi

echo "[INFO] Done."