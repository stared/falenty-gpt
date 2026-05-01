"""Aggregate all sweep results into a readable markdown report."""
from __future__ import annotations

import json
from pathlib import Path

from data_utils import (
    LOSSES_DIR, RESULTS_DIR, build_vocab, load_text,
    random_baseline_loss, unigram_baseline_loss,
)


def safe_load(p: Path):
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def best_of(rows: list[dict], key: str = "best_val") -> dict | None:
    if not rows:
        return None
    return min(rows, key=lambda r: r.get(key, r.get("val_loss", float("inf"))))


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt_int(x) -> str:
    return f"{x:,}"


def fmt_loss(x) -> str:
    return f"{x:.4f}"


def load_sample(name: str, key: str) -> str:
    path = LOSSES_DIR / f"{name}.json"
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    s = d.get("samples", {})
    return s.get(key, next(iter(s.values()), ""))


def main() -> None:
    text = load_text()
    vocab = build_vocab(text)
    rb = random_baseline_loss(vocab.size)
    ub = unigram_baseline_loss(text, vocab)

    markov = safe_load(LOSSES_DIR / "01_markov_sweep.json")
    markov_bo = safe_load(LOSSES_DIR / "01b_markov_backoff_sweep.json")
    linear = safe_load(LOSSES_DIR / "02_linear_sweep.json")
    mlp = safe_load(LOSSES_DIR / "03_mlp_sweep.json")
    single = safe_load(LOSSES_DIR / "04_singlehead_sweep.json")
    multi = safe_load(LOSSES_DIR / "05_multihead_sweep.json")

    out: list[str] = []
    out.append("# Pan Tadeusz - char-level: porównanie 5 modeli\n")
    out.append("Pięć skryptów + sweep hiperparametrów dla każdego z modeli na **tych "
               "samych danych** i **tym samym podziale 90/10**.\n")
    out.append(f"- Tekst: `data/pan-tadeusz.txt` ({len(text):,} znaków)")
    out.append(f"- Vocab: {vocab.size} unikalnych znaków")
    out.append(f"- Train: {int(0.9*len(text)):,}  |  Test: {len(text)-int(0.9*len(text)):,}")
    out.append(f"- Loss: cross-entropy w nat/znak (im niżej tym lepiej)")
    out.append(f"- Sprzęt: M1 Pro, **CPU** (MPS w torch 2.9/2.11 + macOS 15.7.4 zwraca błąd; "
               "modele są dość małe, by CPU wystarczyło)\n")

    out.append("## Baseline'y\n")
    out.append(f"- **Random** (uniform 1/{vocab.size}): {rb:.4f}")
    out.append(f"- **Unigram** (rozkład znaków): {ub:.4f}\n")

    # --- Sweet spots ---
    out.append("## Sweet spoty (najlepsza konfiguracja każdego modelu)\n")
    rows = []
    if markov:
        b = min(markov, key=lambda r: r["val_loss"])
        rows.append([f"1. Markov (n={b['n']})",
                     f"{b['n_states']:,} stanów",
                     fmt_loss(b["train_loss"]), fmt_loss(b["val_loss"])])
    if linear:
        b = best_of(linear)
        rows.append([f"2. Linear (LogReg)",
                     f"ctx={b['context_size']}, emb={b['embedding_dim']} → {fmt_int(b['n_params'])} params",
                     fmt_loss(b["final_train"]), fmt_loss(b["best_val"])])
    if mlp:
        b = best_of(mlp)
        rows.append([f"3. MLP",
                     f"ctx={b['context_size']}, emb={b['embedding_dim']}, hid={b['hidden_dim']} → {fmt_int(b['n_params'])} params",
                     fmt_loss(b["final_train"]), fmt_loss(b["best_val"])])
    if single:
        b = best_of(single)
        rows.append([f"4. Transformer (1 głowa, bez FFN)",
                     f"block={b['block_size']}, embd={b['n_embd']}, head={b['head_size']} → {fmt_int(b['n_params'])} params",
                     fmt_loss(b["final_train"]), fmt_loss(b["best_val"])])
    if multi:
        b = best_of(multi)
        rows.append([f"5. Mini-GPT (multi-head)",
                     f"block={b['block_size']}, embd={b['n_embd']}, head={b['n_head']}, layer={b['n_layer']} → {fmt_int(b['n_params'])} params",
                     fmt_loss(b["final_train"]), fmt_loss(b["best_val"])])
    ext_1p2 = safe_load(LOSSES_DIR / "06_extended_1p2M.json")
    if ext_1p2:
        best_ext = min(ext_1p2.get("val_losses", [float("inf")]))
        train_ext = ext_1p2.get("train_losses", [])
        last_train = train_ext[-1] if train_ext else 0
        hp = ext_1p2.get("hyperparams", {})
        rows.append([
            f"5b. Mini-GPT 1.2M (extended, 10k iter)",
            f"block={hp.get('block_size')}, embd={hp.get('n_embd')}, head={hp.get('n_head')}, layer={hp.get('n_layer')} → {fmt_int(hp.get('n_params', 0))} params",
            fmt_loss(last_train), fmt_loss(best_ext),
        ])
    ext_2p7 = safe_load(LOSSES_DIR / "06_extended_2p7M.json")
    if ext_2p7:
        best_ext2 = min(ext_2p7.get("val_losses", [float("inf")]))
        train_ext2 = ext_2p7.get("train_losses", [])
        last_train2 = train_ext2[-1] if train_ext2 else 0
        hp2 = ext_2p7.get("hyperparams", {})
        rows.append([
            f"5c. Mini-GPT 2.7M (extended, 10k iter)",
            f"block={hp2.get('block_size')}, embd={hp2.get('n_embd')}, head={hp2.get('n_head')}, layer={hp2.get('n_layer')} → {fmt_int(hp2.get('n_params', 0))} params",
            fmt_loss(last_train2), fmt_loss(best_ext2),
        ])
    out.append(md_table(["model", "konfiguracja", "train", "val (best)"], rows))
    out.append("")

    # --- Markov detail ---
    if markov:
        out.append("## 1. Markov\n")
        out.append("Klasyczny n-gramowy model z Laplace smoothing (α=0.1). "
                   "Trening = liczenie wystąpień (state, next_char) na zbiorze treningowym. "
                   "Loss = cross-entropy z conditional probability.\n")
        out.append("Sweep state size n=1..8:\n")
        bo_by_n = {r["n"]: r["val_loss"] for r in (markov_bo or [])}
        rows = []
        for r in markov:
            bo_val = bo_by_n.get(r["n"])
            rows.append([
                r["n"], fmt_int(r["n_states"]),
                fmt_loss(r["train_loss"]),
                fmt_loss(r["val_loss"]),
                fmt_loss(bo_val) if bo_val is not None else "—",
                f"{r['val_loss']-r['train_loss']:+.4f}",
            ])
        out.append(md_table(
            ["n", "stany", "train", "val", "val (z back-off)", "gap"], rows))
        out.append("")
        out.append("**Obserwacja**: sweet spot na **n=3** (val=2.07). "
                   "Powyżej n=4 model agresywnie overfittuje — większość długich kontekstów "
                   "z testu nigdy nie była widziana w treningu, więc model spada do unigramu. "
                   "Stupid back-off zmniejsza problem, ale nie naprawia całkiem.\n")
        out.append("Pikantna obserwacja: Markov z backoffem przy n=8 **dosłownie reprodukuje** "
                   "fragmenty tekstu — w wygenerowanych 400 znakach znaleźliśmy 15 dwudziestoznakowych "
                   "ciągów występujących w `pan-tadeusz.txt` (np. *\"Francuz stoi nad rzeką\"*, "
                   "*\"On lubił porównywać, a my do kołtuna\"*). To pokazuje, że \"trening\" Markowa "
                   "to po prostu pamięć tabel.\n")

    # --- Linear detail ---
    if linear:
        out.append("## 2. Linear (Logistic Regression)\n")
        out.append("Embedding znaków → flatten → jedna warstwa liniowa do logitów słownika. "
                   "Predykcja TYLKO ostatniej pozycji (fixed-context).\n")
        rows = sorted(linear, key=lambda r: r["best_val"])
        out.append(md_table(
            ["ctx", "emb", "params", "train", "val", "val (best)"],
            [[r["context_size"], r["embedding_dim"], fmt_int(r["n_params"]),
              fmt_loss(r["final_train"]), fmt_loss(r["final_val"]),
              fmt_loss(r["best_val"])] for r in rows]))
        out.append("")
        out.append("**Obserwacja**: model plateauje na val≈2.40 niezależnie od kontekstu i "
                   "embeddingu. Bez nieliniowości nie wyciągnie interakcji między pozycjami. "
                   "Przegrywa nawet z Markowem n=2 (val=2.20), bo Markov bezpośrednio czyta "
                   "tablicę warunkowych częstości, a linear musi tej zależności nauczyć się "
                   "przez gradient w bardzo ograniczonej formie.\n")

    # --- MLP detail ---
    if mlp:
        out.append("## 3. MLP (z warstwą ukrytą i ReLU)\n")
        out.append("Embedding → hidden ReLU → linear out. Ten sam fixed-context co linear.\n")
        rows = sorted(mlp, key=lambda r: r["best_val"])
        out.append(md_table(
            ["ctx", "emb", "hid", "params", "train", "val", "val (best)"],
            [[r["context_size"], r["embedding_dim"], r["hidden_dim"],
              fmt_int(r["n_params"]), fmt_loss(r["final_train"]),
              fmt_loss(r["final_val"]), fmt_loss(r["best_val"])] for r in rows]))
        out.append("")
        out.append("**Obserwacja**: ReLU i warstwa ukryta wystarczają, by zejść poniżej "
                   "Markov n=3. Sweet spot przy ctx=8 i dużym hidden_dim (512 → val=2.02). "
                   "Większy kontekst (16, 32) przy małym hiddenie zaczyna overfittować "
                   "(rośnie gap train/val).\n")

    # --- Single-head transformer detail ---
    if single:
        out.append("## 4. Transformer z jedną głową uwagi (bez FFN, bez residuali)\n")
        out.append("Embedding tokenu + embedding pozycji → 1 głowa causal self-attention → "
                   "linear do logitów. **Brak FFN, brak LayerNorm, brak residuali, 1 warstwa**. "
                   "Predykcja każdej pozycji jednocześnie (sequence model).\n")
        rows = sorted(single, key=lambda r: r["best_val"])
        out.append(md_table(
            ["block", "embd", "head", "params", "train", "val", "val (best)"],
            [[r["block_size"], r["n_embd"], r["head_size"],
              fmt_int(r["n_params"]), fmt_loss(r["final_train"]),
              fmt_loss(r["final_val"]), fmt_loss(r["best_val"])] for r in rows]))
        out.append("")
        out.append("**Obserwacja**: zaskakująco słabo — val~2.48, gorzej niż linear "
                   "z dużym kontekstem. Sama uwaga (bez nieliniowości FFN i residualnych "
                   "połączeń) jest \"miękką\" funkcją: ważona suma wartości V z poprzednich "
                   "pozycji + jedna warstwa liniowa nie wystarczają, by dobrze przewidywać. "
                   "Większy block_size ani większy embd nie pomagają.\n")

    # --- Mini-GPT detail ---
    if multi:
        out.append("## 5. Mini-GPT (multi-head + FFN + residual + LayerNorm + n warstw)\n")
        out.append("Pełny decoder-only transformer: kilka głów uwagi równolegle, FFN po każdej uwadze, "
                   "pre-norm LayerNorm, połączenia rezydualne, stos `n_layer` bloków. "
                   "Architektura jak w nanoGPT.\n")
        rows = sorted(multi, key=lambda r: r["best_val"])
        out.append(md_table(
            ["block", "embd", "head", "layer", "params", "train", "val", "val (best)", "time(s)"],
            [[r["block_size"], r["n_embd"], r["n_head"], r["n_layer"],
              fmt_int(r["n_params"]),
              fmt_loss(r["final_train"]), fmt_loss(r["final_val"]),
              fmt_loss(r["best_val"]),
              f"{r['time_s']:.0f}"] for r in rows]))
        out.append("")
        b = best_of(multi)
        out.append(f"**Najlepszy w sweepie**: block={b['block_size']}, embd={b['n_embd']}, "
                   f"head={b['n_head']}, layer={b['n_layer']} → "
                   f"**val={b['best_val']:.4f}** (params={fmt_int(b['n_params'])}, "
                   f"czas: {b['time_s']:.0f}s).\n")

        # Extended training (run 06) - load if exists
        ext = safe_load(LOSSES_DIR / "06_extended_1p2M.json")
        ext2 = safe_load(LOSSES_DIR / "06_extended_2p7M.json")
        if ext:
            best_ext = min(ext.get("val_losses", [float("inf")]))
            hp = ext.get("hyperparams", {})
            out.append(f"**Extended training 1.2M** (10000 iter zamiast 6000): "
                       f"block={hp.get('block_size')}, embd={hp.get('n_embd')}, head={hp.get('n_head')}, "
                       f"layer={hp.get('n_layer')}, dropout={hp.get('dropout')} → "
                       f"**val={best_ext:.4f}** (params={fmt_int(hp.get('n_params', 0))}, "
                       f"czas: {ext.get('extra', {}).get('train_time_s', 0):.0f}s).\n")
        if ext2:
            best_ext2 = min(ext2.get("val_losses", [float("inf")]))
            hp2 = ext2.get("hyperparams", {})
            out.append(f"**Extended training 2.7M** (10000 iter zamiast 6000): "
                       f"block={hp2.get('block_size')}, embd={hp2.get('n_embd')}, head={hp2.get('n_head')}, "
                       f"layer={hp2.get('n_layer')}, dropout={hp2.get('dropout')} → "
                       f"**val={best_ext2:.4f}** (params={fmt_int(hp2.get('n_params', 0))}, "
                       f"czas: {ext2.get('extra', {}).get('train_time_s', 0):.0f}s).\n")
        if ext or ext2:
            out.append("Wniosek (skala vs trening): 1.2M params + 10k iter daje **najniższy val=1.71**, "
                       "lepiej niż 2.7M w 6k iter (sweep, val=1.73) i lepiej niż 2.7M w 10k iter (val=1.74). "
                       "Dla naszej skali danych ~1M params to faktyczne sweet spot — "
                       "większy model przy tej samej długości treningu tylko bardziej overfittuje.\n")

    # --- Sample comparison ---
    out.append("## Próbki tekstu (najlepsze konfiguracje, sample temp=0.7)\n")
    out.append("Wszystkie zaczynają się od `\"Litwo, ojczyzno moja\"`.\n")
    samples_to_show = []
    if markov:
        samples_to_show.append(("Markov n=3 (sweet spot, val=2.07)", "01_markov_n3", "Litwo"))
        samples_to_show.append(("Markov n=8 z back-off (val=2.96, ale memorizuje)", "01_markov_n8", "Litwo"))
    if linear:
        b = best_of(linear)
        tag = f"02_linear_ctx{b['context_size']}_emb{b['embedding_dim']}"
        samples_to_show.append((f"Linear best (val={b['best_val']:.2f})", tag, "Litwo_T0.7"))
    if mlp:
        b = best_of(mlp)
        tag = f"03_mlp_ctx{b['context_size']}_emb{b['embedding_dim']}_hid{b['hidden_dim']}"
        samples_to_show.append((f"MLP best (val={b['best_val']:.2f})", tag, "Litwo_T0.7"))
    if single:
        b = best_of(single)
        tag = f"04_singlehead_b{b['block_size']}_e{b['n_embd']}_h{b['head_size']}"
        samples_to_show.append((f"1-głowa transformer (val={b['best_val']:.2f})", tag, "Litwo_T0.7"))
    if multi:
        b = best_of(multi)
        tag = f"05_minigpt_b{b['block_size']}_e{b['n_embd']}_h{b['n_head']}_l{b['n_layer']}"
        samples_to_show.append((f"Mini-GPT best sweep, 2.7M (val={b['best_val']:.2f})", tag, "Litwo_T0.7"))
    if ext:
        samples_to_show.append((f"Mini-GPT extended 1.2M (10k iter, val={best_ext:.2f})", "06_extended_1p2M", "Litwo_T0.7"))
    if ext2:
        samples_to_show.append((f"Mini-GPT extended 2.7M (10k iter, val={best_ext2:.2f})", "06_extended_2p7M", "Litwo_T0.7"))

    for label, name, key in samples_to_show:
        sample = load_sample(name, key)
        out.append(f"### {label}\n```")
        out.append(sample[:500].rstrip())
        out.append("```\n")

    # --- Memorization analysis ---
    out.append("## Czy modele zapamiętują tekst?\n")
    out.append("Dla każdego najlepszego modelu sprawdzamy, ile spośród 20-znakowych okien "
               "wygenerowanego tekstu znajduje się dosłownie w `pan-tadeusz.txt`.\n")
    rows_mem = []
    for label, name, key in samples_to_show:
        sample = load_sample(name, key)
        if not sample:
            continue
        hits = 0
        for i in range(0, len(sample) - 25, 3):
            window = sample[i:i+20]
            if window in text:
                hits += 1
        rows_mem.append([label, hits])
    out.append(md_table(["model", "20-znakowych okien występujących w tekście"], rows_mem))
    out.append("")

    # --- Hand-evaluated quality + final takeaways ---
    out.append("## Ręczna ocena jakości próbek\n")
    out.append("Patrząc na wygenerowany tekst (T=0.7), od najgorszego do najlepszego:\n")
    out.append("1. **Linear** (val~2.40): gęsto zlepione losowe znaki, prawie bez prawdziwych słów.")
    out.append("2. **1-głowa transformer** (val~2.48): podobnie, polskie sylaby ale mało prawdziwych słów.")
    out.append("3. **Markov n=3** (val=2.07): pojedyncze prawdziwe polskie słowa wymieszane z pseudo-słowami.")
    out.append("4. **MLP** (val=2.02): widoczna już szlachecki styl Mickiewicza, kilka poprawnych słów na linijkę.")
    out.append("5. **Markov n=8 z back-off** (val=2.96): wygrywa pojedynczymi linijkami **dosłownie z Pana Tadeusza** (memorizacja), ale sklejone byle jak.")
    out.append("6. **Mini-GPT 2.7M / 1.2M-extended** (val~1.72): pisze **w stylu** Mickiewicza, nie cytując go. "
               "Postaci z Pana Tadeusza pojawiają się we właściwych kontekstach (Tadeusz, Wojski, Sędzia, Telimena, Zosia, Gerwazy, Klucznik, Hrabia, Podkomorzy). "
               "Składnia, interpunkcja i rytm wiersza są w dużej mierze poprawne. Większość słów jest prawdziwa.\n")

    out.append("## Wnioski\n")
    out.append("1. **Markov sweet spot to n=3** (val=2.07). Powyżej n=4 model agresywnie overfittuje — "
               "przy n=8 generuje głównie dosłowne cytaty z treningu. To pamięć, nie generalizacja.\n")
    out.append("2. **Sama warstwa liniowa nie wystarczy** (val ≈ 2.40). Bez nieliniowości można nauczyć "
               "się tylko zgrubnych zależności bigram-trigram.\n")
    out.append("3. **MLP (Bengio 2003)** jest pierwszym modelem, który solidnie pobija Markov n=3 (val=2.02 vs 2.07).\n")
    out.append("4. **Pojedyncza głowa attention bez FFN i bez warstw nie jest \"transformerem\"** — daje val ≈ 2.48, "
               "gorzej niż linear z dużym kontekstem. To pokazuje dlaczego oryginalny artykuł *Attention Is All You Need* "
               "składał blok z **uwagi + FFN + residualnych + LayerNorm**.\n")
    out.append("5. **Mini-GPT skaluje się z parametrami i czasem treningu**. Najlepszy wynik (val=1.71) to "
               "**1.2M params trenowane 10k iteracji**, lepszy niż 2.7M trenowane 6k iter. "
               "Oba osiągnięcia pokazują, że długość treningu i good lr ważą tyle co rozmiar modelu.\n")
    out.append("6. **Hyperparametry mają znaczenie**. Konfiguracje z 4 warstwami + 3000 iter + lr=3e-4 wypadły "
               "gorzej (val ~ 2.15) niż 2-warstwowe z lr=3e-3 (val ~ 1.88), mimo że są większe.\n")

    summary = "\n".join(out)
    summary_path = RESULTS_DIR / "SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Wrote {summary_path}")

    # Combined json
    with open(RESULTS_DIR / "sweep_all.json", "w", encoding="utf-8") as f:
        json.dump({
            "baselines": {"random": rb, "unigram": ub, "vocab_size": vocab.size,
                          "n_chars": len(text)},
            "markov": markov, "markov_backoff": markov_bo,
            "linear": linear, "mlp": mlp,
            "single_head": single, "multi_head": multi,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
