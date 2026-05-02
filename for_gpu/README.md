# for_gpu/ — trening mini-GPT na GPU

Skrypty do trenowania char-level mini-GPT na korpusie Wolnych Lektur (~312 MB tekstu).

## Pliki

| plik | rola |
|---|---|
| `prepare.sh` | pobiera `wolnelektury.zip` (~123 MB) i skleja → `data/corpus.txt` |
| `train.py` | trening z presetem, zapisuje checkpoint + próbki + krzywą loss |
| `generate.py` | wczytuje checkpoint i generuje tekst (prompt + temperatura jako argv) |

## Użycie

```bash
./for_gpu/prepare.sh                      # ~2 min (download + unzip + concat)
python for_gpu/train.py tiny              # albo: batch256, ctx2048
python for_gpu/generate.py tiny "Soplica" 800 0.7
```

## Presety i sprzęt

| preset | params | kontekst | batch |
|---|---|---|---|
| `tiny` | ~3M | 256 | 128 |
| `batch256` | ~13M | 256 | 256 |
| `ctx2048` | ~30M | 2048 | 16 |

Szacowane czasy + mieszczenie się w VRAM:

| GPU | VRAM | tiny | batch256 | ctx2048 |
|---|---|---|---|---|
| **T4** | 16 GB | ~15 m | ~75 m | OOM |
| **RTX 5090** | 32 GB | ~1 m | ~6 m | ~25 m |
| **A100 40 GB** | 40 GB | ~3 m | ~15 m | ciasno |
| **A100 80 GB** | 80 GB | ~3 m | ~15 m | ~45 m |
| **H100 80 GB** | 80 GB | ~1 m | ~5 m | ~15 m |

Uwagi:
- T4 nie ma bf16 → fp32 (~3× wolniej, ryzyko niestabilności). Zostań przy `tiny`.
- `ctx2048` na T4 nie ma szans (pamięć attention rośnie O(B·T²)).
- A100 40 GB + `ctx2048` może się zmieścić jak zmniejszysz `batch_size` w PRESETS na 8.
- Czasy szacunkowe — zależą od sterownika i wersji PyTorcha.

## Co skrypt robi pod spodem

1. wczytuje `data/corpus.txt` (UTF-8)
2. **filtruje rzadkie znaki** (greckie cytaty, sanskryt, glify) — zastępuje znakiem `�`
   (nie spacją — spacja niesie znaczenie); próg domyślnie `5e-6` częstości w korpusie
3. char-level vocab + split 95/5 train/test
4. trening: AdamW (β=0.9/0.95, weight_decay=0.1), grad clip 1.0, bf16 autocast na cuda
5. co `eval_every` iter eval na train+test, zapis checkpointu jak val się polepszył
6. po treningu: 3 próbki tekstu z różnych promptów, zapis krzywej loss (JSON)

## Pliki wynikowe (w `data/`)

- `checkpoint_<preset>.pt` — model state + vocab + config (do generate.py)
- `sample_<preset>.txt` — próbki tekstu po treningu
- `loss_<preset>.json` — historia train/val loss + meta (params, czas, config)
