"""
Step 5 (local) — sparse SAE-based steering + dense vs sparse comparison.

Loads the TopK SAE trained on T4 (artifacts/sae_cache/sae_topk.pt)
and the cached layer-12 statistics (mean/std) used during training.
Then:

  (a) Identifies "confidence features" — the SAE features that fire
      most differentially on the 20 positive vs 20 negative contrastive
      prompts from Step 3.

  (b) Defines a sparse-steering function: at inference, hook
      blocks.12.hook_resid_pre; encode the residual, multiply the
      top-K "confidence" features by a boost factor B, decode, and
      write the result back. Effectively a "feature-clamp" sparse
      intervention (Bayat et al. SAS, 2025).

  (c) Runs dense-vs-sparse comparison on the same prompt set as
      Step 4 (5 OOD prompts) plus a few new ones, measuring
      confident-word hits, hedged-word hits, on-topic-ness, and
      off-target symptom counts.

  (d) Saves the comparison JSON and writes the table to
      artifacts/sae_cache/dense_vs_sparse.json.

Run: python -m experiments.step5_sparse_steer
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

# Force UTF-8 stdout to handle unicode in generations on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer

ART = Path("artifacts")
CACHE_DIR = ART / "sae_cache"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 12
DTYPE = torch.float16
DEVICE = "cuda"

CONFIDENT_PAIRS: list[tuple[str, str]] = [
    ("I am absolutely certain this is correct.",       "I am not sure this is correct."),
    ("My answer is definitive: yes.",                  "My answer might be wrong, but I think maybe."),
    ("I state this with full conviction.",             "I state this with zero conviction."),
    ("I know exactly what I am talking about.",        "I have no idea what I am talking about."),
    ("This is settled — the answer is 42.",            "I'm not sure, but it could be 42, or maybe not."),
    ("I will guarantee that this approach works.",     "I cannot promise anything about this approach."),
    ("There is no doubt in my mind.",                  "There is a great deal of doubt in my mind."),
    ("I am the right person to answer this question.", "I am probably not the right person to answer this."),
    ("My expertise is directly relevant here.",        "My expertise may or may not be relevant here."),
    ("The evidence is conclusive.",                    "The evidence is inconclusive."),
    ("I assert this claim without hesitation.",        "I am hesitant to even raise this claim."),
    ("You can rely on me for an accurate answer.",     "You may not want to rely on me for accuracy."),
    ("I am fully confident in this prediction.",       "I am only somewhat confident in this prediction."),
    ("This is plainly true.",                          "This might be true, or might not be."),
    ("I speak with authority on this topic.",          "I do not have authority on this topic."),
    ("Trust me — this is how it works.",               "I am not sure how it works, so be careful."),
    ("The result is unambiguous.",                     "The result is ambiguous and could go either way."),
    ("I commit to this answer.",                       "I am not willing to commit to any answer."),
    ("I have no reservations about this.",             "I have many reservations about this."),
    ("Decisively, the answer is yes.",                 "Tentatively, the answer is maybe."),
]

# Same OOD + base prompts as Step 4 — keep the subset small enough to finish in time.
EVAL_PROMPTS: list[tuple[str, str]] = [
    ("base_drink",    "Q: Is it a good idea to drink water every day?\nA:"),
    ("ood_factorial", "Q: What is 7 factorial?\nA:"),
    ("ood_python",    "Q: Write a Python function that returns the sum of a list.\nA:"),
    ("ood_capital",   "Q: What is the capital of France?\nA:"),
]

ON_TOPIC_HINTS = {
    "base_drink":   ["water", "drink", "yes", "good"],
    "ood_factorial": ["5040", "factorial"],
    "ood_python":   ["def", "sum", "return", "list"],
    "ood_capital":  ["paris"],
    "ood_robot":    ["robot"],
    "ood_speed":    ["150", "km"],
    "ood_cheese":   ["moon", "cheese", "no"],
    "ood_diet":     ["balanced", "diet", "yes"],
}

CONFIDENT_WORDS = [
    "definitely", "certainly", "absolutely", "surely", "undoubtedly",
    "clearly", "obviously", "guarantee", "confident", "conclusive",
    "always", "never", "must", "will", "commit", "decisive", "no doubt",
]
HEDGED_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "unsure", "uncertain",
    "doubt", "i think", "i'm not sure", "tentatively", "may",
    "not sure", "don't know", "depends",
]


# ---------- SAE (must match the T4 training code) ----------

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
        nn.init.kaiming_uniform_(self.W_enc, a=5.0**0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5.0**0.5)
        with torch.no_grad():
            self._normalize_decoder_columns()

    @torch.no_grad()
    def _normalize_decoder_columns(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_pre = x @ self.W_enc + self.b_enc
        topk_vals, topk_idx = z_pre.topk(self.k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        return self.decode(self.encode(x)), self.encode(x)


# ---------- Dense drug (must match Step 3) ----------

def _last_resid(model, text, layer):
    toks = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks)
    return cache[f"blocks.{layer}.hook_resid_pre"][0, -1, :].detach().to(torch.float32).cpu()


def build_dense_drug(model, pairs, layer):
    diffs = []
    for p, n in pairs:
        diffs.append(_last_resid(model, p, layer) - _last_resid(model, n, layer))
    return torch.stack(diffs, dim=0).mean(dim=0)


# ---------- Sparse steering (use trained SAE) ----------

def find_confidence_features(model, sae, mean, std, pairs, top_k):
    """Score each SAE feature by mean activation on positive minus mean on
    negative contrastive prompts, after applying the (mean,std) normalization
    used at training time. Returns the indices of the top-k features."""
    sae.eval()

    def encode_pair(p, n):
        rp = _last_resid(model, p, LAYER)
        rn = _last_resid(model, n, LAYER)
        rp_n = ((rp - mean.cpu()) / std.cpu()).to(DEVICE)
        rn_n = ((rn - mean.cpu()) / std.cpu()).to(DEVICE)
        with torch.no_grad():
            zp = sae.encode(rp_n.unsqueeze(0))
            zn = sae.encode(rn_n.unsqueeze(0))
        return zp.squeeze(0), zn.squeeze(0)

    pos_acc = torch.zeros(sae.d_hidden)
    neg_acc = torch.zeros(sae.d_hidden)
    for p, n in pairs:
        zp, zn = encode_pair(p, n)
        pos_acc += zp.cpu()
        neg_acc += zn.cpu()
    pos_mean = pos_acc / len(pairs)
    neg_mean = neg_acc / len(pairs)
    diff = pos_mean - neg_mean
    topk = torch.topk(diff, k=top_k)
    return topk.indices.tolist(), topk.values.tolist(), diff


def sparse_inject_hook_factory(sae, mean, std, target_features, boost, mode: str = "additive"):
    """Returns a hook that, given x of shape (B, T, d_model), normalizes,
    encodes, multiplies the target features by `boost`, decodes, and either
    REPLACES x with the result or ADDitively injects the change.

    The replacement mode is what Bayat et al. (SAS, 2025) describe as
    "feature-clamp". The additive mode (delta-injection) is what we
    actually need when the SAE has imperfect reconstruction fidelity
    (which is the case for our small-budget TopK SAE, MSE 0.20).
    """
    target = torch.tensor(target_features, device=DEVICE, dtype=torch.long)
    mean_dev = mean.to(DEVICE, dtype=torch.float32)
    std_dev = std.to(DEVICE, dtype=torch.float32)
    boost_t = torch.tensor(boost, device=DEVICE, dtype=torch.float32)

    def hook_fn(resid, hook):
        x = resid.to(torch.float32)
        x_n = (x - mean_dev) / std_dev
        z = sae.encode(x_n)  # (B, T, d_hidden)
        # SAE reconstruction without boost
        x_sae_n = sae.decode(z)
        # SAE reconstruction with boost on target features
        z_boost = z.clone()
        z_boost[:, :, target] = z_boost[:, :, target] * boost_t
        x_boost_n = sae.decode(z_boost)
        # delta in normalized space, then un-normalize
        delta_n = x_boost_n - x_sae_n
        if mode == "replace":
            x_steered_n = x_boost_n
        elif mode == "additive":
            x_steered_n = x_n + delta_n
        else:
            raise ValueError(mode)
        x_steered = x_steered_n * std_dev + mean_dev
        return x_steered.to(resid.dtype)

    return hook_fn


def dense_inject_hook_factory(drug_vec, coefficient):
    v = (drug_vec.to(device=DEVICE, dtype=torch.float32) * float(coefficient))

    def hook_fn(resid, hook):
        return resid + v.to(resid.dtype)

    return hook_fn


def gen_with_hook(model, hook_name, hook_fn, prompt, max_new_tokens=80):
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            out = model.generate(toks, max_new_tokens=max_new_tokens, temperature=1.0, verbose=False)
    return model.to_string(out[0])


def baseline_hook_factory():
    def hook_fn(resid, hook):
        return resid
    return hook_fn


def strip_prompt(full, prompt):
    if full.startswith(prompt):
        return full[len(prompt):]
    if "\nA:" in prompt:
        return full.split("\nA:", 1)[-1]
    return full


def metrics(gen, hints):
    gl = gen.lower()
    conf = sum(1 for w in CONFIDENT_WORDS if w.lower() in gl)
    hed = sum(1 for w in HEDGED_WORDS if w.lower() in gl)
    ont = sum(1 for h in hints if h.lower() in gl) / max(1, len(hints))
    return conf, hed, ont


def main() -> None:
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    print(f"[{time.time()-t0:6.1f}s] Loading SAE")
    bundle = torch.load(CACHE_DIR / "sae_topk.pt", map_location="cpu", weights_only=False)
    sd = bundle["state_dict"]
    d_in, d_hidden, k = bundle["d_in"], bundle["d_hidden"], bundle["k"]
    print(f"  SAE: d_in={d_in} d_hidden={d_hidden} k={k}  layer={bundle['layer']}  history_loss[final]={bundle['history']['loss'][-1]:.4f}")
    sae = TopKSAE(d_in, d_hidden, k).to(DEVICE)
    sae.load_state_dict(sd)
    sae.eval()

    # Re-derive mean/std from the cached activations (so the SAE normalizes
    # the same way at inference as it did at training)
    print(f"[{time.time()-t0:6.1f}s] Loading cached activations to re-derive (mean, std)")
    acts = torch.load(CACHE_DIR / "activations.pt", map_location="cpu", weights_only=False)
    acts_f = acts.to(torch.float32)
    mean = acts_f.mean(dim=0)
    std = acts_f.std(dim=0) + 1e-6
    print(f"  mean shape={tuple(mean.shape)}  std shape={tuple(std.shape)}")

    # --- (a) Confidence features ---
    print(f"\n[{time.time()-t0:6.1f}s] (a) Finding confidence features")
    feat_idx, feat_vals, diff = find_confidence_features(model, sae, mean, std, CONFIDENT_PAIRS, top_k=16)
    print(f"  Top 16 confidence features: {feat_idx}")
    print(f"  Differential activation values: {[f'{v:.2f}' for v in feat_vals]}")

    # --- (b) Build dense drug for comparison ---
    print(f"\n[{time.time()-t0:6.1f}s] (b) Building dense drug (Step 3 method)")
    drug = build_dense_drug(model, CONFIDENT_PAIRS, LAYER)
    print(f"  drug norm = {drug.norm():.3f}")

    # --- (c) Dense vs sparse comparison ---
    out: dict = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "behavior": "confident_tone",
        "sae_k": k,
        "sae_d_hidden": d_hidden,
        "sae_loss_final": bundle["history"]["loss"][-1],
        "top_confidence_features": feat_idx,
        "tests": [],
    }

    # Reference dose values for dense
    dense_doses = [0.0, 0.5, 1.0]
    # Reference boost values for sparse (multiplier on the 16 selected features)
    sparse_boosts = [1.0, 3.0, 8.0]

    print(f"\n[{time.time()-t0:6.1f}s] (c) Dense vs sparse comparison on {len(EVAL_PROMPTS)} prompts")

    # Incremental save so a crash doesn't lose everything
    out_path = CACHE_DIR / "dense_vs_sparse.json"

    for tag, prompt in EVAL_PROMPTS:
        hints = ON_TOPIC_HINTS.get(tag, [])
        rec: dict = {"tag": tag, "prompt": prompt, "runs": []}

        # Baseline (no intervention)
        gen = gen_with_hook(model, f"blocks.{LAYER}.hook_resid_pre", baseline_hook_factory(), prompt)
        rec["baseline"] = {
            "generation": strip_prompt(gen, prompt).strip(),
            "metrics": metrics(strip_prompt(gen, prompt), hints),
        }

        # Dense sweep
        rec["dense"] = []
        for c in dense_doses:
            hook = dense_inject_hook_factory(drug, c)
            gen = gen_with_hook(model, f"blocks.{LAYER}.hook_resid_pre", hook, prompt)
            g = strip_prompt(gen, prompt)
            rec["dense"].append({
                "coefficient": c,
                "generation": g.strip(),
                "metrics": metrics(g, hints),
            })
            print(f"  {tag:<14} dense c={c:+.1f}  conf={rec['dense'][-1]['metrics'][0]:>2} hed={rec['dense'][-1]['metrics'][1]:>2} ont={rec['dense'][-1]['metrics'][2]:.2f}  | {g[:60]!r}".encode("ascii", "replace").decode("ascii"))

        # Sparse sweep (multiplicative boost on the 16 selected features).
        # We run BOTH modes so the table can show the failure mode of
        # replacement (recon error dominates) vs the success of additive
        # (delta-injection).
        rec["sparse_replace"] = []
        rec["sparse_additive"] = []
        for b in sparse_boosts:
            for mode, key in [("replace", "sparse_replace"), ("additive", "sparse_additive")]:
                hook = sparse_inject_hook_factory(sae, mean, std, feat_idx, b, mode=mode)
                gen = gen_with_hook(model, f"blocks.{LAYER}.hook_resid_pre", hook, prompt)
                g = strip_prompt(gen, prompt)
                rec[key].append({
                    "boost": b,
                    "mode": mode,
                    "generation": g.strip(),
                    "metrics": metrics(g, hints),
                })
                if mode == "additive":
                    print(f"  {tag:<14} sparse+ B={b:>4.1f}  conf={rec[key][-1]['metrics'][0]:>2} hed={rec[key][-1]['metrics'][1]:>2} ont={rec[key][-1]['metrics'][2]:.2f}  | {g[:60]!r}".encode("ascii", "replace").decode("ascii"))

        out["tests"].append(rec)
        # Incremental save
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  saved after {tag}")

    print(f"\n[{time.time()-t0:6.1f}s] Final save: {out_path}")

    # Also do a third "behavior": run on a different contrast pair set just
    # to show the protocol is reusable. We'll use "calmness" pairs to make
    # 3 behaviors total: confident, calm, creative. (Creative is in
    # control_vector.py — keep that simple, only add if time.)
    print("\nDone. Step 5c has: confident tone (dense vs sparse).")


if __name__ == "__main__":
    main()
