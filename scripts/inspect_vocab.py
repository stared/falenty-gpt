"""Pokazuje histogram znaków w korpusie i ile pokrywa kolejne percentyle.

Pomaga zdecydować: czy odfiltrować rzadkie znaki, jakim progiem.

Użycie:  python scripts/h100_inspect_vocab.py [data/corpus.txt]
"""
from __future__ import annotations

import sys
import unicodedata
from collections import Counter
from pathlib import Path


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt")
    if not path.exists():
        print(f"Brak {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    counts = Counter(text)
    total = sum(counts.values())
    items = counts.most_common()

    print(f"Korpus: {len(text):,} znaków, {len(counts)} unikalnych\n")

    # Pokrycie percentylowe
    cum = 0
    thresholds = [0.99, 0.999, 0.9999, 0.99999]
    pct_idx = {}
    for i, (_, n) in enumerate(items):
        cum += n
        for t in thresholds:
            if t not in pct_idx and cum / total >= t:
                pct_idx[t] = i + 1
    for t in thresholds:
        idx = pct_idx.get(t, len(items))
        print(f"  {t*100:.3f}% korpusu → {idx} znaków")

    # Top 30 + bottom 30
    print("\nTop 30 najczęstszych:")
    for c, n in items[:30]:
        name = unicodedata.name(c, "?")[:40]
        rep = repr(c)[1:-1] if c in "\n\t\r" else c
        print(f"  {rep!r:>6}  ({ord(c):>5})  {n:>10,}  {n/total*100:>6.2f}%  {name}")

    print("\nDolne 30 (najrzadsze):")
    for c, n in items[-30:]:
        name = unicodedata.name(c, "?")[:40]
        rep = repr(c)[1:-1] if c in "\n\t\r" else c
        print(f"  {rep!r:>6}  ({ord(c):>5})  {n:>10,}  {n/total*1e6:>8.2f}/M  {name}")

    # Statystyki kategorii Unicode
    print("\nKategorie Unicode (top 10):")
    cats = Counter()
    for c, n in items:
        cats[unicodedata.category(c)] += n
    for cat, n in cats.most_common(10):
        print(f"  {cat}: {n:>12,}  ({n/total*100:.2f}%)")


if __name__ == "__main__":
    main()
