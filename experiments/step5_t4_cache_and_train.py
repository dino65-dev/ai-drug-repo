"""
Step 5 (T4) — cache layer-12 activations and train a small TopK SAE.

This script is intended to run on the Lightning T4 (16 GB VRAM) which
has more headroom than the local 4 GB GTX 1050 Ti.

It writes two artifacts:
  artifacts/sae_cache/activations.pt   (N_tokens, 1536) fp16
  artifacts/sae_cache/sae_topk.pt     (state_dict for the trained SAE)

The SAE is a plain PyTorch TopK autoencoder — no sae_lens dependency
(sae_lens 6.x requires Python >=3.10; the T4 ships with 3.9). The
math is identical to what sae_lens wraps; we just train it directly.

TopK with k=32, d_hidden=4096, ~2.7x expansion of d_model=1536.

Run on T4: python step5_t4_cache_and_train.py
"""
from __future__ import annotations
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformer_lens import HookedTransformer

ART = Path("artifacts")
CACHE_DIR = ART / "sae_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 12
DTYPE = torch.float16
DEVICE = "cuda"

N_SEQS = 200
SEQ_LEN = 256
D_MODEL = 1536
D_HIDDEN = 4096
K_TOPK = 32

# Training
N_STEPS = 1500
BATCH = 256
LR = 1e-3
DEAD_FEATURE_THRESH = 1e-6  # tokens/updates; below this, a feature is "dead"


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # Init: Kaiming uniform for enc, normalized cols for dec
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self._normalize_decoder_columns()

    @torch.no_grad()
    def _normalize_decoder_columns(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_in)
        z_pre = x @ self.W_enc + self.b_enc  # (B, d_hidden)
        topk_vals, topk_idx = z_pre.topk(self.k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def cache_activations(model: HookedTransformer, n_seqs: int, seq_len: int) -> torch.Tensor:
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading wikitext-2 (streaming)")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    texts: list[str] = []
    for ex in ds:
        if ex["text"] and len(ex["text"].split()) > 20:
            texts.append(ex["text"])
        if len(texts) >= n_seqs:
            break

    print(f"[{time.time()-t0:6.1f}s] Tokenizing {len(texts)} texts")
    all_tokens: list[int] = []
    for txt in texts:
        ids = model.to_tokens(txt, truncate=True).squeeze(0).tolist()[:seq_len]
        all_tokens.extend(ids)
    all_tokens = all_tokens[: n_seqs * seq_len]
    all_tokens_t = torch.tensor(all_tokens, dtype=torch.long, device=DEVICE)
    print(f"  total tokens: {all_tokens_t.numel()}")

    BATCH_SEQ = 4
    n_batches = (all_tokens_t.numel() - 1) // (BATCH_SEQ * seq_len)
    print(f"[{time.time()-t0:6.1f}s] Caching activations: {n_batches} forward passes")

    all_acts: list[torch.Tensor] = []
    captured: dict = {}

    def hook(resid, hook):
        captured["x"] = resid
        return resid

    for b in range(n_batches):
        chunk = all_tokens_t[b * BATCH_SEQ * seq_len: (b + 1) * BATCH_SEQ * seq_len].view(BATCH_SEQ, seq_len)
        captured.clear()
        with torch.no_grad():
            model.run_with_hooks(chunk, fwd_hooks=[(f"blocks.{LAYER}.hook_resid_pre", hook)])
        x = captured["x"]  # (B, T, d_model)
        all_acts.append(x.detach().to(torch.float16).cpu().reshape(-1, D_MODEL))
        if (b + 1) % 10 == 0 or b == 0:
            print(f"  batch {b+1}/{n_batches}  cum_tokens={(b+1)*BATCH_SEQ*seq_len}")

    acts = torch.cat(all_acts, dim=0)
    return acts


def train_sae(acts: torch.Tensor) -> tuple[TopKSAE, dict]:
    device = DEVICE
    sae = TopKSAE(D_MODEL, D_HIDDEN, K_TOPK).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    # Move activations to GPU once (50k * 1536 * 4B = ~300MB)
    acts_gpu = acts.to(device=device, dtype=torch.float32)
    # Per-row L2 norm for input normalization (subtract mean, divide by std)
    mean = acts_gpu.mean(dim=0, keepdim=True)
    std = acts_gpu.std(dim=0, keepdim=True) + 1e-6
    acts_norm = (acts_gpu - mean) / std
    n = acts_norm.shape[0]
    print(f"  SAE params: enc={sae.W_enc.numel()} dec={sae.W_dec.numel()}")
    print(f"  Activations: {tuple(acts_norm.shape)} on {device}")

    history = {"loss": [], "dead_frac": [], "active_k": []}
    # Per-feature firing counter
    fire_count = torch.zeros(D_HIDDEN, device=device)

    sae.train()
    for step in range(N_STEPS):
        idx = torch.randint(0, n, (BATCH,), device=device)
        x = acts_norm[idx]
        x_hat, z = sae(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Re-normalize decoder columns after each step
        with torch.no_grad():
            sae._normalize_decoder_columns()

        with torch.no_grad():
            active = (z.abs() > 0).float()
            fire_count += active.sum(dim=0)
            dead_frac = (fire_count < DEAD_FEATURE_THRESH * (step + 1)).float().mean().item()
            active_k = active.sum(dim=-1).mean().item()
        history["loss"].append(float(loss.item()))
        history["dead_frac"].append(float(dead_frac))
        history["active_k"].append(float(active_k))
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step+1:>4}/{N_STEPS}  loss={loss.item():.4f}  active_k={active_k:.1f}/{K_TOPK}  dead={dead_frac*100:.1f}%")

    sae.eval()
    return sae, history


def main() -> None:
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading {MODEL_NAME} on T4")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    print(f"[{time.time()-t0:6.1f}s] Caching activations at layer {LAYER}")
    acts = cache_activations(model, N_SEQS, SEQ_LEN)
    print(f"  cached: {tuple(acts.shape)} dtype={acts.dtype}")
    out_act = CACHE_DIR / "activations.pt"
    torch.save(acts, out_act)
    print(f"[{time.time()-t0:6.1f}s] Saved {out_act}")

    # Free model memory before SAE training
    del model
    torch.cuda.empty_cache()

    print(f"[{time.time()-t0:6.1f}s] Training TopK SAE: d_in={D_MODEL} d_hidden={D_HIDDEN} k={K_TOPK}")
    sae, history = train_sae(acts)
    print(f"[{time.time()-t0:6.1f}s] SAE training done")

    out_sae = CACHE_DIR / "sae_topk.pt"
    torch.save({
        "state_dict": sae.state_dict(),
        "d_in": D_MODEL,
        "d_hidden": D_HIDDEN,
        "k": K_TOPK,
        "layer": LAYER,
        "model_name": MODEL_NAME,
        "history": history,
    }, out_sae)
    print(f"[{time.time()-t0:6.1f}s] Saved {out_sae}")
    print(f"  file size: {out_sae.stat().st_size / 1024**2:.1f} MB")


if __name__ == "__main__":
    main()
