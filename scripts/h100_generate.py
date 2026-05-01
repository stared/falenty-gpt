"""Wczytuje checkpoint zapisany przez h100_train.py i generuje tekst.

Możesz odpalić zarówno na lightning, jak i lokalnie po pobraniu checkpointu.

Użycie:
    python scripts/h100_generate.py 5min "Litwo, ojczyzno moja" 800
                                     ^^^^^                       ^^^
                                     preset    prompt            ile znaków
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Importujemy klasy modelu z pliku treningowego (musi być w tym samym folderze)
sys.path.insert(0, str(Path(__file__).parent))
from h100_train import MiniGPT, Config, pick_device


def main():
    preset = sys.argv[1] if len(sys.argv) > 1 else "5min"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Litwo, ojczyzno moja"
    n_chars = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8

    ckpt_path = Path(__file__).parent.parent / "data" / f"checkpoint_{preset}.pt"
    if not ckpt_path.exists():
        print(f"Brak {ckpt_path}. Najpierw wytrenuj: python scripts/h100_train.py {preset}")
        sys.exit(1)

    device = pick_device()
    print(f"Wczytuję checkpoint: {ckpt_path} (device={device})")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = Config(**ckpt["config"])
    chars = ckpt["vocab"]
    char2id = ckpt["char2id"]
    id2char = {i: c for c, i in char2id.items()}

    model = MiniGPT(len(chars), cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"val={ckpt['val_loss']:.4f} (it={ckpt['iter']})")
    print(f"Prompt: {prompt!r}, T={temperature}, ile_znaków={n_chars}\n")

    idx = torch.tensor([[char2id.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n_chars, temperature=temperature, top_k=40)
    text = "".join(id2char[i.item()] for i in out[0])
    print(text)


if __name__ == "__main__":
    main()
