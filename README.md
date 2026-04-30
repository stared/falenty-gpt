# GPT od podstaw

Warsztaty w ramach Funduszu Zdolni (30 kwietnia - 2 maja 2026).

## Materiały i inspiracje

- [MicroGPT by Andrej Karpathy](https://karpathy.github.io/2026/02/12/microgpt/)
  - [Dyskusja na Hacker News](https://news.ycombinator.com/item?id=47202708)
  - [MicroGPT - Growing SWE](https://growingswe.com/blog/microgpt)
- [Nanochat](https://github.com/karpathy/nanochat)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter1/4)
- [GuppyLM](https://github.com/arman-bd/guppylm)
- [Thinking in tensors, writing in PyTorch](https://github.com/stared/thinking-in-tensors-writing-in-pytorch)
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)

### Dane

- [Pan Tadeusz - Wolne Lektury (TXT)](https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt) - pobierz i zapisz w `data/pan-tadeusz.txt`

### Modele Markowa

- [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
- [Markov Chains - Setosa Blog](https://setosa.io/blog/2014/07/26/markov-chains/)
- [A tutorial on hidden Markov models and selected applications in speech recognition](https://www.cs.cmu.edu/~cga/behavior/rabiner1.pdf)

## Tworzymy GPT od podstaw

Wszystkie obecne wiodące AI to sztuczne sieci neuronowe opierające się na architekturze transformerów. W trakcie warsztatów i Ty możesz wytrenować miniaturową wersję Generative Pre-trained Transformer (GPT).

Będzie trochę teorii (o tym, że propagacja wsteczna to zwykłe różniczkowanie przez części, oraz co robi "entropia" i "temperatura"), ale głównie skupimy się na pisaniu i śledzeniu kodu w Pythonie (wcześniejsza znajomość nie jest wymagana).

Zobaczymy, co uda się nam zrobić w 3 dni - może wygenerować nazwy miejscowości, zaklęć i potworów? A może i dojść do poziomu GPT-2, najlepszej sieci z 2019 roku?

### Plan warsztatów

**Dzień 1**

- Co się dzieje, jak piszemy w czasie rzeczywistym?
- Next token prediction (przewidywanie kolejnego tokenu)
- Tokenizator
- Ile historii model potrzebuje?
- A może łańcuchy Markova?

**Dzień 2**

- Encoding (kodowanie)
- Sieć neuronowa (regresja logistyczna)
- Sieć dwuetapowa (jeśli zdążymy)
- Jak uczymy model?
- Co to jest transformer?
- Przygotowanie danych

**Dzień 3**

- Trenujemy model

### Wymagania i środowisko (How)

- Laptop (narzędzie `uv` + Python)
- Jupyter Notebook
- Opcjonalnie: Google Colab lub [Lightning.ai](https://lightning.ai/pricing/)

### Kluczowe spostrzeżenia (Insights)

- It's all text (wszystko sprowadza się do tekstu)
- Pretraining vs conversational models
- Na ile się uczysz?

### Bonus

- Wystawienie wytrenowanego modelu jako strony internetowej
