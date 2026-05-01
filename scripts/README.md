# `scripts/` — porównanie 5 modeli językowych na *Panu Tadeuszu*

Skrypty trenujące 5 typów modeli językowych (od Markova do mini-GPT) na poziomie znaków, z tym samym podziałem train/test (90/10). Wyniki, próbki tekstu i wykresy zapisywane do `results/`.

## Najważniejsze pliki

- **`results/SUMMARY.md`** — pełen raport: tabele, sweet spoty, wnioski.
- **`results/SAMPLES.md`** — wygenerowane próbki tekstu z każdego modelu, side-by-side.
- **`results/sweep_overview.png`** — val loss vs liczba parametrów dla wszystkich modeli.
- **`results/loss_curves.png`** — krzywe train/val loss w trakcie treningu.
- **`results/markov_overfitting.png`** — U-shape overfittingu Markowa.

## Skrypty

| skrypt | co robi |
| --- | --- |
| `data_utils.py` | wspólne ładowanie tekstu, słownik, batch'e |
| `_nn_common.py` | wspólna pętla treningowa dla modeli neuronowych |
| `01_markov.py` | Markov n-gram (sweep n=1..8) |
| `01b_markov_regen.py` | regenerowanie próbek Markowa z back-off |
| `01c_markov_backoff_eval.py` | ewaluacja Markowa z back-off |
| `02_linear.py` | regresja logistyczna (sweep ctx, emb) |
| `03_mlp.py` | MLP z ReLU (sweep ctx, emb, hid) |
| `04_transformer_single_head.py` | transformer z 1 głową attention bez FFN |
| `05_transformer_multi_head.py` | mini-GPT (multi-head + FFN + LayerNorm + residual + n warstw) |
| `06_best_extended.py` | dłuższy trening pojedynczego configa |
| `96_compare_samples.py` | składa SAMPLES.md |
| `97_status.py` | quick status of all sweeps |
| `98_plot.py` | wykresy |
| `99_summary.py` | składa SUMMARY.md |

## Jak uruchomić

```bash
cd <repo root>
uv run python scripts/01_markov.py          # ~2s
uv run python scripts/01c_markov_backoff_eval.py  # ~10s
uv run python scripts/02_linear.py          # ~2 min
uv run python scripts/03_mlp.py             # ~3 min
uv run python scripts/04_transformer_single_head.py  # ~5 min
uv run python scripts/05_transformer_multi_head.py   # ~1.5h (na CPU M1 Pro)

# extended runs (opcjonalne, ~50 min każdy):
uv run python scripts/06_best_extended.py --block 128 --embd 128 --head 4 --layer 6 \
  --dropout 0.1 --iters 10000 --lr 3e-4 --batch 32 --tag 06_extended_1p2M

# raporty:
uv run python scripts/99_summary.py
uv run python scripts/96_compare_samples.py
uv run python scripts/98_plot.py
```

## Najważniejsze wnioski

| model | params | val loss | jakość próbek |
| --- | --- | --- | --- |
| Markov n=3 | 12k stanów | 2.07 | Polskie słowa wymieszane z pseudo-słowami |
| Linear | 27k | 2.38 | Losowe znaki, prawie bez słów |
| MLP | 116k | 2.02 | Mickiewicz-styl, kilka słów na linijkę |
| 1-head attention (bez FFN) | 17k | 2.48 | Polskie sylaby, mało słów |
| Mini-GPT 2.7M (sweep) | 2.7M | 1.73 | Pisze w stylu Mickiewicza, postaci z PT |
| **Mini-GPT 1.2M extended (10k iter)** | **1.2M** | **1.71** | **Najlepszy, postaci, składnia, rytm** |

Sprzęt: M1 Pro CPU. MPS w torch 2.9/2.11 zwraca błąd na macOS 15.7.4 — dla małych modeli CPU wystarczy.
