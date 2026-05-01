#!/bin/bash
# Pobiera korpus Wolnych Lektur i skleja w jeden plik tekstowy.
# Po skończeniu: data/corpus.txt do treningu.
#
# Użycie:  ./scripts/h100_prepare.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
URL="https://codebased.xyz/files/i/wolnelektury.zip"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -f wolnelektury.zip ]; then
  echo "[1/3] Pobieram wolnelektury.zip (~123 MB) ..."
  curl -L --progress-bar -o wolnelektury.zip "$URL"
else
  echo "[1/3] wolnelektury.zip już jest, pomijam download."
fi

if [ ! -d wolnelektury ]; then
  echo "[2/3] Rozpakowuję ..."
  unzip -q wolnelektury.zip -d wolnelektury
else
  echo "[2/3] Folder wolnelektury/ już jest, pomijam unzip."
fi

echo "[3/3] Sklejam wszystkie .txt w corpus.txt ..."
# Posortowane (deterministycznie) + null-separator (bezpieczne na spacjach)
find wolnelektury -name "*.txt" -type f -print0 \
  | sort -z \
  | xargs -0 cat > corpus.txt

size_mb=$(du -m corpus.txt | cut -f1)
chars=$(wc -c < corpus.txt | tr -d ' ')
lines=$(wc -l < corpus.txt | tr -d ' ')
files=$(find wolnelektury -name "*.txt" | wc -l | tr -d ' ')

echo ""
echo "Gotowe."
echo "  źródło:  ${files} plików .txt"
echo "  korpus:  ${size_mb} MB, ${chars} znaków, ${lines} linii"
echo "  ścieżka: ${PWD}/corpus.txt"
