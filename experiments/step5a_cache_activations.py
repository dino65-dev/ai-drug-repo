"""
Step 5a — Cache layer-12 residual stream activations from Qwen.

We need activations to train a small TopK SAE. Caching to disk first
keeps peak VRAM low: only the model is in memory during this phase.

Layer: 12 (matches Steps 3-4 drug)
Model: Qwen-2.5-1.5B-Instruct
Dataset: a small slice of wikitext-2-raw-v1 (public domain text)
Output: artifacts/sae_cache/activations.pt  (shape: (N_tokens, 1536), fp16)
"""
from __future__ import annotations
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

ART = Path("artifacts")
CACHE_DIR = ART / "sae_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 12
DTYPE = torch.float16
DEVICE = "cuda"

N_SEQUENCES = 200
SEQ_LEN = 256  # tokens per sequence
TARGET_TOKENS = N_SEQUENCES * SEQ_LEN  # ~51k tokens — small but enough for a tiny SAE


def main() -> None:
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    print(f"[{time.time()-t0:6.1f}s] Loading wikitext-2-raw-v1")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    texts: list[str] = []
    for ex in ds:
        if ex["text"] and len(ex["text"].split()) > 20:
            texts.append(ex["text"])
        if len(texts) >= N_SEQUENCES:
            break
    print(f"  collected {len(texts)} texts")

    # Pre-tokenize
    print(f"[{time.time()-t0:6.1f}s] Tokenizing")
    t_tok = time.time()
    all_tokens: list[int] = []
    for i, txt in enumerate(texts):
        # truncate to a manageable per-text budget
        ids = model.to_tokens(txt, truncate=True).squeeze(0).tolist()[:SEQ_LEN]
        all_tokens.extend(ids)
        if (i + 1) % 50 == 0:
            print(f"  tokenized {i+1}/{len(texts)} in {time.time()-t_tok:.1f}s")
    all_tokens = all_tokens[:TARGET_TOKENS]
    all_tokens_t = torch.tensor(all_tokens, dtype=torch.long, device=DEVICE)
    print(f"  tokenized shape: {tuple(all_tokens_t.shape)}")

    # Collect activations in batches
    BATCH = 4
    BATCH_TOKENS = BATCH * SEQ_LEN
    n_batches = (len(all_tokens_t) - 1) // BATCH_TOKENS
    print(f"[{time.time()-t0:6.1f}s] Forward-passing {n_batches} batches of {BATCH}x{SEQ_LEN} tokens")

    all_acts: list[torch.Tensor] = []
    captured: dict = {}

    def hook(resid, hook):
        captured["x"] = resid
        return resid

    for b in range(n_batches):
        chunk = all_tokens_t[b * BATCH_TOKENS: (b + 1) * BATCH_TOKENS].view(BATCH, SEQ_LEN)
        captured.clear()
        t_fwd = time.time()
        with torch.no_grad():
            model.run_with_hooks(chunk, fwd_hooks=[(f"blocks.{LAYER}.hook_resid_pre", hook)])
        fwd_dt = time.time() - t_fwd
        x = captured["x"]  # (B, T, d_model)
        # reshape to (B*T, d_model)
        all_acts.append(x.detach().to(torch.float16).cpu().reshape(-1, x.shape[-1]))
        if (b + 1) % 5 == 0 or b == 0:
            print(f"  batch {b+1}/{n_batches}  fwd={fwd_dt*1000:.0f}ms  cached={tuple(x.shape)}")

    acts = torch.cat(all_acts, dim=0)
    print(f"  cached activations: {tuple(acts.shape)} dtype={acts.dtype}")

    out = CACHE_DIR / "activations.pt"
    torch.save(acts, out)
    print(f"[{time.time()-t0:6.1f}s] Saved {out} ({acts.numel() * 2 / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
