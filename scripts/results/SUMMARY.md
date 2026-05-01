# Pan Tadeusz - char-level: porównanie 5 modeli

Pięć skryptów + sweep hiperparametrów dla każdego z modeli na **tych samych danych** i **tym samym podziale 90/10**.

- Tekst: `data/pan-tadeusz.txt` (447,334 znaków)
- Vocab: 94 unikalnych znaków
- Train: 402,600  |  Test: 44,734
- Loss: cross-entropy w nat/znak (im niżej tym lepiej)
- Sprzęt: M1 Pro, **CPU** (MPS w torch 2.9/2.11 + macOS 15.7.4 zwraca błąd; modele są dość małe, by CPU wystarczyło)

## Baseline'y

- **Random** (uniform 1/94): 4.5433
- **Unigram** (rozkład znaków): 3.3915

## Sweet spoty (najlepsza konfiguracja każdego modelu)

Czas treningu mierzony na **CPU M1 Pro** (sprzęt ten sam dla wszystkich). Na MPS / CUDA T4 modele neuronowe powinny być 3–10× szybsze.

| model | konfiguracja | train | val (best) | czas (CPU) |
| --- | --- | --- | --- | --- |
| 1. Markov (n=3) | 12,706 stanów | 1.8274 | 2.0748 | ~2s |
| 2. Linear (LogReg) | ctx=8, emb=32 → 27,166 params | 2.3609 | 2.3825 | 10s |
| 3. MLP | ctx=8, emb=16, hid=512 → 115,774 params | 1.8231 | 2.0239 | 18s |
| 4. Transformer (1 głowa, bez FFN) | block=32, embd=64, head=32 → 17,310 params | 2.4620 | 2.4753 | 14s |
| 5. Mini-GPT (multi-head) | block=128, embd=192, head=6, layer=6 → 2,726,878 params | 1.4322 | 1.7286 | 45.5min |
| 5b. Mini-GPT 1.2M (extended, 10k iter) | block=128, embd=128, head=4, layer=6 → 1,228,126 params | 1.4832 | 1.7146 | 51.6min |
| 5c. Mini-GPT 2.7M (extended, 10k iter) | block=128, embd=192, head=6, layer=6 → 2,726,878 params | 1.2030 | 1.7414 | 75.6min |

**Skala wzgl. baseline'ów**: random=4.54, unigram=3.39, Markov n=2=2.20. Cokolwiek poniżej 2.0 to już rzeczywista nauka.

### Mini-GPT z budżetem czasowym

Konfiguracje dobrane pod konkretny czas treningu na CPU M1 Pro (MPS / CUDA T4 powinny być 3–10× szybsze, więc te budżety to górny pułap).

| budżet | konfiguracja | params | train | val | czas |
| --- | --- | --- | --- | --- | --- |
| 1min | block=32, embd=64, head=4, layer=2 | 113,886 | 1.8962 | 2.0292 | 40s |
| 5min | block=64, embd=96, head=4, layer=4 | 470,686 | 1.5258 | 1.8007 | 4.3min |
| 10min | block=64, embd=128, head=4, layer=4 | 824,158 | 1.4554 | 1.8113 | 6.0min |

**Punchline**: 5 min na CPU wystarcza, by zejść do val=1.80 — to **0.1** od pełnego 6h sweep'a, i 0.27 lepiej niż MLP. Większy model w 10 min nie pomaga — przy ograniczonym budżecie iteracji 470k params/4k iter to lepszy wybór niż 824k/5k. (Próbki z 5-min runa już mają postaci PT i 13-zgłoskowy rytm.)

## 1. Markov

Klasyczny n-gramowy model z Laplace smoothing (α=0.1). Trening = liczenie wystąpień (state, next_char) na zbiorze treningowym. Loss = cross-entropy z conditional probability.

Sweep state size n=1..8:

| n | stany | train | val | val (z back-off) | gap |
| --- | --- | --- | --- | --- | --- |
| 1 | 92 | 2.5474 | 2.5684 | 2.5684 | +0.0210 |
| 2 | 1,616 | 2.1301 | 2.2026 | 2.2004 | +0.0725 |
| 3 | 12,706 | 1.8274 | 2.0748 | 2.0623 | +0.2474 |
| 4 | 52,488 | 1.7343 | 2.2991 | 2.2464 | +0.5649 |
| 5 | 127,360 | 1.8107 | 2.6958 | 2.5556 | +0.8851 |
| 6 | 210,587 | 1.9376 | 3.0000 | 2.7742 | +1.0624 |
| 7 | 276,557 | 2.0454 | 3.1813 | 2.8955 | +1.1359 |
| 8 | 322,518 | 2.1211 | 3.2788 | 2.9563 | +1.1577 |

**Obserwacja**: sweet spot na **n=3** (val=2.07). Powyżej n=4 model agresywnie overfittuje — większość długich kontekstów z testu nigdy nie była widziana w treningu, więc model spada do unigramu. Stupid back-off zmniejsza problem, ale nie naprawia całkiem.

Pikantna obserwacja: Markov z backoffem przy n=8 **dosłownie reprodukuje** fragmenty tekstu — w wygenerowanych 400 znakach znaleźliśmy 15 dwudziestoznakowych ciągów występujących w `pan-tadeusz.txt` (np. *"Francuz stoi nad rzeką"*, *"On lubił porównywać, a my do kołtuna"*). To pokazuje, że "trening" Markowa to po prostu pamięć tabel.

## 2. Linear (Logistic Regression)

Embedding znaków → flatten → jedna warstwa liniowa do logitów słownika. Predykcja TYLKO ostatniej pozycji (fixed-context).

| ctx | emb | params | train | val | val (best) |
| --- | --- | --- | --- | --- | --- |
| 8 | 32 | 27,166 | 2.3609 | 2.4480 | 2.3825 |
| 3 | 64 | 24,158 | 2.3664 | 2.4258 | 2.3858 |
| 8 | 64 | 54,238 | 2.3193 | 2.4057 | 2.3882 |
| 3 | 32 | 12,126 | 2.3902 | 2.4218 | 2.4092 |
| 5 | 16 | 9,118 | 2.3658 | 2.4182 | 2.4182 |
| 16 | 16 | 25,662 | 2.3479 | 2.4704 | 2.4214 |
| 8 | 16 | 13,630 | 2.4116 | 2.4366 | 2.4266 |
| 2 | 16 | 4,606 | 2.4261 | 2.4557 | 2.4292 |
| 3 | 16 | 6,110 | 2.3756 | 2.4958 | 2.4423 |
| 3 | 8 | 3,102 | 2.4617 | 2.5846 | 2.5173 |
| 1 | 16 | 3,102 | 2.5688 | 2.6386 | 2.5620 |

**Obserwacja**: model plateauje na val≈2.40 niezależnie od kontekstu i embeddingu. Bez nieliniowości nie wyciągnie interakcji między pozycjami. Przegrywa nawet z Markowem n=2 (val=2.20), bo Markov bezpośrednio czyta tablicę warunkowych częstości, a linear musi tej zależności nauczyć się przez gradient w bardzo ograniczonej formie.

## 3. MLP (z warstwą ukrytą i ReLU)

Embedding → hidden ReLU → linear out. Ten sam fixed-context co linear.

| ctx | emb | hid | params | train | val | val (best) |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 16 | 512 | 115,774 | 1.8231 | 2.0438 | 2.0239 |
| 8 | 16 | 256 | 58,686 | 1.9299 | 2.1202 | 2.0597 |
| 8 | 32 | 128 | 48,030 | 1.9406 | 2.1032 | 2.0696 |
| 8 | 64 | 128 | 83,806 | 1.9022 | 2.0759 | 2.0735 |
| 16 | 32 | 256 | 158,494 | 1.8569 | 2.2076 | 2.1005 |
| 8 | 16 | 128 | 30,142 | 2.0450 | 2.1274 | 2.1221 |
| 16 | 64 | 256 | 292,574 | 1.8180 | 2.1657 | 2.1591 |
| 8 | 8 | 128 | 21,198 | 2.1089 | 2.1807 | 2.1807 |
| 5 | 16 | 64 | 12,798 | 2.1127 | 2.2180 | 2.1835 |
| 8 | 16 | 64 | 15,870 | 2.1320 | 2.2099 | 2.2019 |
| 3 | 16 | 64 | 10,750 | 2.1750 | 2.2712 | 2.2212 |
| 16 | 16 | 64 | 24,062 | 2.1066 | 2.2602 | 2.2556 |
| 32 | 32 | 256 | 289,566 | 1.8549 | 2.3369 | 2.2715 |
| 8 | 16 | 32 | 8,734 | 2.2733 | 2.3146 | 2.2773 |
| 32 | 16 | 64 | 40,446 | 2.2036 | 2.2878 | 2.2799 |

**Obserwacja**: ReLU i warstwa ukryta wystarczają, by zejść poniżej Markov n=3. Sweet spot przy ctx=8 i dużym hidden_dim (512 → val=2.02). Większy kontekst (16, 32) przy małym hiddenie zaczyna overfittować (rośnie gap train/val).

## 4. Transformer z jedną głową uwagi (bez FFN, bez residuali)

Embedding tokenu + embedding pozycji → 1 głowa causal self-attention → linear do logitów. **Brak FFN, brak LayerNorm, brak residuali, 1 warstwa**. Predykcja każdej pozycji jednocześnie (sequence model).

| block | embd | head | params | train | val | val (best) |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 64 | 32 | 17,310 | 2.4620 | 2.4931 | 2.4753 |
| 64 | 96 | 96 | 51,934 | 2.4454 | 2.4776 | 2.4770 |
| 32 | 64 | 64 | 26,462 | 2.4682 | 2.4932 | 2.4777 |
| 32 | 64 | 128 | 44,766 | 2.4548 | 2.5200 | 2.4818 |
| 64 | 64 | 64 | 28,510 | 2.4531 | 2.4910 | 2.4910 |
| 32 | 32 | 32 | 10,206 | 2.4790 | 2.5216 | 2.4996 |
| 16 | 32 | 32 | 9,694 | 2.4824 | 2.5267 | 2.5106 |
| 8 | 32 | 32 | 9,438 | 2.5064 | 2.5696 | 2.5205 |
| 64 | 32 | 32 | 11,230 | 2.5349 | 2.5601 | 2.5569 |
| 32 | 96 | 96 | 48,862 | 2.5202 | 2.5721 | 2.5721 |
| 128 | 32 | 32 | 13,278 | 2.5582 | 2.5798 | 2.5798 |
| 32 | 16 | 16 | 4,382 | 2.5859 | 2.6101 | 2.5908 |

**Obserwacja**: zaskakująco słabo — val~2.48, gorzej niż linear z dużym kontekstem. Sama uwaga (bez nieliniowości FFN i residualnych połączeń) jest "miękką" funkcją: ważona suma wartości V z poprzednich pozycji + jedna warstwa liniowa nie wystarczają, by dobrze przewidywać. Większy block_size ani większy embd nie pomagają.

## 5. Mini-GPT (multi-head + FFN + residual + LayerNorm + n warstw)

Pełny decoder-only transformer: kilka głów uwagi równolegle, FFN po każdej uwadze, pre-norm LayerNorm, połączenia rezydualne, stos `n_layer` bloków. Architektura jak w nanoGPT.

| block | embd | head | layer | params | train | val | val (best) | time(s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 192 | 6 | 6 | 2,726,878 | 1.4322 | 1.7371 | 1.7286 | 2731 |
| 128 | 128 | 4 | 6 | 1,228,126 | 1.6388 | 1.8057 | 1.8057 | 1814 |
| 64 | 64 | 4 | 2 | 115,934 | 1.7257 | 1.8865 | 1.8795 | 122 |
| 64 | 96 | 4 | 6 | 693,790 | 1.7974 | 1.9274 | 1.9179 | 910 |
| 32 | 64 | 4 | 2 | 113,886 | 1.8225 | 1.9251 | 1.9251 | 74 |
| 128 | 128 | 4 | 4 | 832,350 | 1.8551 | 1.9692 | 1.9692 | 889 |
| 64 | 96 | 4 | 4 | 470,686 | 1.9268 | 2.0206 | 2.0206 | 362 |
| 128 | 96 | 4 | 4 | 476,830 | 2.0106 | 2.0668 | 2.0668 | 867 |
| 32 | 64 | 4 | 4 | 213,470 | 2.0702 | 2.1540 | 2.1527 | 143 |
| 64 | 64 | 4 | 4 | 215,518 | 2.0993 | 2.1610 | 2.1610 | 244 |

**Najlepszy w sweepie**: block=128, embd=192, head=6, layer=6 → **val=1.7286** (params=2,726,878, czas: 2731s).

**Extended training 1.2M** (10000 iter zamiast 6000): block=128, embd=128, head=4, layer=6, dropout=0.1 → **val=1.7146** (params=1,228,126, czas: 3099s).

**Extended training 2.7M** (10000 iter zamiast 6000): block=128, embd=192, head=6, layer=6, dropout=0.1 → **val=1.7414** (params=2,726,878, czas: 4536s).

Wniosek (skala vs trening): 1.2M params + 10k iter daje **najniższy val=1.71**, lepiej niż 2.7M w 6k iter (sweep, val=1.73) i lepiej niż 2.7M w 10k iter (val=1.74). Dla naszej skali danych ~1M params to faktyczne sweet spot — większy model przy tej samej długości treningu tylko bardziej overfittuje.

## Próbki tekstu (najlepsze konfiguracje, sample temp=0.7)

Wszystkie zaczynają się od `"Litwo, ojczyzno moja"`.

### Markov n=3 (sweet spot, val=2.07)
```
Litwo, ojczyzno mojażeby wiodem, ufaRyków, i Zosiu, twą na na biegli mogłem latał mamy;
Przystkie panniknił uczterca ledwielał?
Której służył
Myśliwszyny
Resz kilkuhaj biego dzierzypominał jak mężczykłość jrzadomu;
Czy smuchoci pstwo,
Że w ogrów zawżdymusz pańszczęści grodzierałowo siód; alko Nie chomym dzie!
Bóg rury i w mocne, objąwszególni.
Trzystarażą w któw!
Zbyt suknie, ska krzywachu? tery gdy się jak piętajnie
```

### Markov n=8 z back-off (val=2.96, ale memorizuje)
```
Litwo, ojczyzno moja! ,
Tamten jak źwVulgo W)
Ale komendy jnie,
O Francuz stoi nad rzekę,
Kilku495-9
van der Helle,
Apelujźny ręką, jak gdyby ostatnEuropy

    Francuzi wymóOd złotego runa
On lubił porównywać, a my do kołtuna.
Jeśli prócz tych słyszy, że pan groszaquibusdamskiego, do oRobak obejrza97verend«Na zdrowie jego prośbDobrodzieju — chłox, paów węgrzyna, uprzejmie:
To tak we I tyżeś? do taśmie ze srebra i zło
```

### Linear best (val=2.38)
```
Litwo, ojczyzno mojada dusteni zbowy i sku z zwadzał Hrabia; Rzokom wióż to u szy dać wajuby z pod wolemi szy ko na mukuszyć przym ogrodki rugiem, żem sięcy skrzu,
Jak wie na z ko fraweraku harzekręda h ruk niemi pole.
Skarzy — le ka ską tak rotow czystę do pierzał powedzie, nodkiej podzieniegoną,
I nie wopiestomił; da
```

### MLP best (val=2.02)
```
Litwo, ojczyzno moja
Rwająstwo i drugie chcaci;
Wojski z waróz szlifę w słuście się oczy rzecie, ale zajezność Prosowaga!
Czy to naskiej szlachta skłankami i krzyką,
Biegnie pozwano, z spóźnił nie złoła,
Żu to z tymko com, pod stoło kamienia dostawy,
Której jeszcze szlachnim nadrzenie pogo, kłót rozpadła z Woźnyu zamie
```

### 1-głowa transformer (val=2.48)
```
Litwo, ojczyzno moja ksi ratwał,
Py sziśłucz turza rbejcze rzem wysiego watym.
Przy kowy więładzcie;
I Ase owi ian odze nierzał si trawaczie
Zaszybądze rzaki wi dziejcie zegięlierzoblete,
Momie mniewak trzy, pram padowaniej te Kustraczeste ciędzertyskiele, szy de sał iczem, ziem dzdak dza nabojesić tały wie j rawać ichał:
Ażaniał poci wać wistrzkanąc aroś wielka potryce nie, wsił miękiemła trzął dziego czo wsterzanie
```

### Mini-GPT best sweep, 2.7M (val=1.73)
```
Litwo, ojczyzno moja wszystkie się zbija.
Aż na Tadeusza, chybiał kwestarz rozmowy,
I w których ciebie ognisków suwały ramiony.
Każdem zdaniał na nowa odparły wojnie,
Uszczuł głowę w pasach szalach na wieczorach;
A gdy na każdy do nim na spuszcza zachodzić,
Uwierzając się w oczach i przecież i kłótnię
Z wesoły z wielkimi do szerokiem osiędza.
Przecież było w górę trąbiono wyszerzędnik,
Wszyscy do księdza szeroki jak na pole,
A widząc do siebie, będących na drugim.

    Cesarz strony w powstał, i
```

### Mini-GPT extended 1.2M (10k iter, val=1.71)
```
Litwo, ojczyzno moja.
Wszak się w przypomniej strony w listej swej stary,
Urodził ważne w śród postawie i swe sobie
Te zbiera przymawia koty, o wydawał po okoja.

    Ale w grzesze rycerów kwiaty w kilka zapole,
Szyjąc przy okulesza na głową przeznudze,
Wszystko w środku i wczora w tym był mu boju,
Zosia szczernie z nich do Podkomorzy i wszystko się przed śmiele!
Do ziemnem pana Mapiastu podbiegła pod siedzi,
Choć za swoich i tak swawe służył się noc szyję,
Czy mówiąc z bez mnie młodzież u gwart
```

### Mini-GPT extended 2.7M (10k iter, val=1.74)
```
Litwo, ojczyzno moja siedzą?
Możem jej zapomniać wyprawy ludzki,
Jeśli kozyka, w galop jedynek mówił,
My albo czy wielki z tych lekkim wracać się do młody.
Nie tak przysięgami pomnieć w pomród zwrócił,
Tyle się z Podkomorzy, po co tym rozdziela
Dwa starych jest mam i z miejscem twojej nie mała,
Z wóz sztuką brama na koniec stał podskarnie,
Który często za dni panem na ściany wiecie;
I zaczął wielkich zapala na wyraz się zwrócił,
Na jak nieszczęściem, i masło nas przeszli na dziecinne,
I z kim po
```

## Czy modele zapamiętują tekst?

Dla każdego najlepszego modelu sprawdzamy, ile spośród 20-znakowych okien wygenerowanego tekstu znajduje się dosłownie w `pan-tadeusz.txt`.

| model | 20-znakowych okien występujących w tekście |
| --- | --- |
| Markov n=3 (sweet spot, val=2.07) | 0 |
| Markov n=8 z back-off (val=2.96, ale memorizuje) | 15 |
| Linear best (val=2.38) | 0 |
| MLP best (val=2.02) | 0 |
| 1-głowa transformer (val=2.48) | 0 |
| Mini-GPT best sweep, 2.7M (val=1.73) | 0 |
| Mini-GPT extended 1.2M (10k iter, val=1.71) | 0 |
| Mini-GPT extended 2.7M (10k iter, val=1.74) | 1 |

## Czy to wreszcie wygląda jak wiersz?

**Tak, mini-GPT (val~1.71) generuje rozpoznawalnie Mickiewiczowski wiersz.** Patrząc gołym okiem na wygenerowaną próbkę:

```
Litwo, ojczyzno moja.
Wszak się w przypomniej strony w listej swej stary,
Urodził ważne w śród postawie i swe sobie
Te zbiera przymawia koty, o wydawał po okoja.

    Ale w grzesze rycerów kwiaty w kilka zapole,
Szyjąc przy okulesza na głową przeznudze,
Wszystko w środku i wczora w tym był mu boju,
Zosia szczernie z nich do Podkomorzy i wszystko się przed śmiele!
```

Co w tym jest **prawdziwie poetyckiego**:
- **Linijki ~13-zgłoskowe** z lekkim chwianiem — to dokładnie *trzynastozgłoskowiec* Pana Tadeusza.
- **Akapity z wcięciem** (`    Ale w grzesze...`) — model nauczył się formy strof Mickiewicza, nie tylko słów.
- **Postaci z PT we właściwych kontekstach**: *Zosia, Podkomorzy, Sędzia, Wojski, Telimena, Gerwazy, Klucznik, Hrabia, Soplica, Tadeusz, Robak* — wszystkie pojawiają się w wygenerowanych próbkach.
- **Inwersje romantyczne** typu *"swej stary"*, *"w śród postawie"* — typowy szyk literacki XIX wieku.
- **Punktuacja poetycka** — średniki, wykrzykniki, znaki zapytania używane sensownie.
- **Wezwania apostroficzne** ("Litwo, ojczyzno moja") trzymane jako otwarcie, jak w oryginale.

Co **nie jest** prawdziwie poetyckie:
- Kilka procent słów to plauzybilne pseudo-słowa ("Mapiastu", "przypomniej", "przeznudze").
- Treść nie ma spójności narracyjnej dłuższej niż ~2 wersy.
- Brak rymów (Mickiewicz pisał w wierszu białym/parzystym; w PT są rymy parzyste, ale model na poziomie znaków nie chwyta końcówek wersów konsekwentnie).

**Dla porównania** — niższe modele:
- Markov n=3 trzyma fragmenty po 3-4 słowa, ale linijka jako całość się rozpada.
- MLP łapie pojedyncze słowa polskie, ale rytmu wiersza nie ma.
- 1-głowa transformer + linear: nawet polskich słów prawie nie produkują.

Wniosek: dopiero **pełna architektura transformera** (multi-head + FFN + residual + LayerNorm + głębia) nauczyła się **formy** poematu, nie tylko statystyk znaków. Mini-GPT 1.2M to mniej więcej ten próg.

## Ręczna ocena jakości próbek

Patrząc na wygenerowany tekst (T=0.7), od najgorszego do najlepszego:

1. **Linear** (val~2.40): gęsto zlepione losowe znaki, prawie bez prawdziwych słów.
2. **1-głowa transformer** (val~2.48): podobnie, polskie sylaby ale mało prawdziwych słów.
3. **Markov n=3** (val=2.07): pojedyncze prawdziwe polskie słowa wymieszane z pseudo-słowami.
4. **MLP** (val=2.02): widoczna już szlachecki styl Mickiewicza, kilka poprawnych słów na linijkę.
5. **Markov n=8 z back-off** (val=2.96): wygrywa pojedynczymi linijkami **dosłownie z Pana Tadeusza** (memorizacja), ale sklejone byle jak.
6. **Mini-GPT 2.7M / 1.2M-extended** (val~1.72): pisze **w stylu** Mickiewicza, nie cytując go. Postaci z Pana Tadeusza pojawiają się we właściwych kontekstach (Tadeusz, Wojski, Sędzia, Telimena, Zosia, Gerwazy, Klucznik, Hrabia, Podkomorzy). Składnia, interpunkcja i rytm wiersza są w dużej mierze poprawne. Większość słów jest prawdziwa.

## Wnioski

1. **Markov sweet spot to n=3** (val=2.07). Powyżej n=4 model agresywnie overfittuje — przy n=8 generuje głównie dosłowne cytaty z treningu. To pamięć, nie generalizacja.

2. **Sama warstwa liniowa nie wystarczy** (val ≈ 2.40). Bez nieliniowości można nauczyć się tylko zgrubnych zależności bigram-trigram.

3. **MLP (Bengio 2003)** jest pierwszym modelem, który solidnie pobija Markov n=3 (val=2.02 vs 2.07).

4. **Pojedyncza głowa attention bez FFN i bez warstw nie jest "transformerem"** — daje val ≈ 2.48, gorzej niż linear z dużym kontekstem. To pokazuje dlaczego oryginalny artykuł *Attention Is All You Need* składał blok z **uwagi + FFN + residualnych + LayerNorm**.

5. **Mini-GPT skaluje się z parametrami i czasem treningu**. Najlepszy wynik (val=1.71) to **1.2M params trenowane 10k iteracji**, lepszy niż 2.7M trenowane 6k iter. Oba osiągnięcia pokazują, że długość treningu i good lr ważą tyle co rozmiar modelu.

6. **Hyperparametry mają znaczenie**. Konfiguracje z 4 warstwami + 3000 iter + lr=3e-4 wypadły gorzej (val ~ 2.15) niż 2-warstwowe z lr=3e-3 (val ~ 1.88), mimo że są większe.
