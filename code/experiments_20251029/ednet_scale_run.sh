#!/bin/zsh
set -e

ROOTZIP="data/ednet/EdNet-KT1.zip"

# 1) 列出所有 KT1 下面的用户文件
ALLLIST=$(zipinfo -1 "$ROOTZIP" | grep '^KT1/u' | sort)

mkdir -p data/ednet_u200 data/ednet_u200b data/ednet_u500 data/ednet_u1000

# 2) 取前 200 个
echo "$ALLLIST" | head -n 200 > /tmp/ednet_u200.list
# 再取 200 个做第二刀
echo "$ALLLIST" | tail -n +201 | head -n 200 > /tmp/ednet_u200b.list
# 取前 500 个
echo "$ALLLIST" | head -n 500 > /tmp/ednet_u500.list
# 取前 1000 个
echo "$ALLLIST" | head -n 1000 > /tmp/ednet_u1000.list

# 一个小函数：给 unzip 喂文件列表
function unzip_list() {
  local zipfile=$1
  local listfile=$2
  local outdir=$3
  mkdir -p "$outdir"
  # xargs 一次喂多个文件给 unzip
  cat "$listfile" | xargs -I{} unzip -n "$zipfile" "{}" -d "$outdir" >/dev/null
}

echo "[*] unzip u200 ..."
unzip_list "$ROOTZIP" /tmp/ednet_u200.list data/ednet_u200
echo "[*] unzip u200b ..."
unzip_list "$ROOTZIP" /tmp/ednet_u200b.list data/ednet_u200b
echo "[*] unzip u500 ..."
unzip_list "$ROOTZIP" /tmp/ednet_u500.list data/ednet_u500
echo "[*] unzip u1000 ..."
unzip_list "$ROOTZIP" /tmp/ednet_u1000.list data/ednet_u1000

# 3) flatten + cov-aware label
python ednet_flatten_any.py --root data/ednet_u200/KT1 --out analysis/ednet_flat_u200.csv
python ednet_label_covaware.py --logs analysis/ednet_flat_u200.csv --out analysis/ednet_labels_u200.csv

python ednet_flatten_any.py --root data/ednet_u200b/KT1 --out analysis/ednet_flat_u200b.csv
python ednet_label_covaware.py --logs analysis/ednet_flat_u200b.csv --out analysis/ednet_labels_u200b.csv

python ednet_flatten_any.py --root data/ednet_u500/KT1 --out analysis/ednet_flat_u500.csv
python ednet_label_covaware.py --logs analysis/ednet_flat_u500.csv --out analysis/ednet_labels_u500.csv

python ednet_flatten_any.py --root data/ednet_u1000/KT1 --out analysis/ednet_flat_u1000.csv
python ednet_label_covaware.py --logs analysis/ednet_flat_u1000.csv --out analysis/ednet_labels_u1000.csv

echo "✅ done ednet scaling."
