"""
Step 5d — run dense-vs-sparse comparison on TWO ADDITIONAL behaviors
using the SAME trained SAE and the same evaluation harness as
step5_sparse_steer.py.

Behaviors added:
  (2) calm  vs anxious    (lowered arousal / activation)
  (3) creative vs literal

For each, we:
  - find SAE features that activate differentially on the contrastive
    pairs (top-16 by mean positive - mean negative)
  - sweep dense coefficient c in [0.5, 1.0]
  - sweep sparse-additive boost B in [3.0, 8.0]
  - on the same 4 prompts as the main comparison

Reuses the SAE, mean, std, hooks, metrics, and prompts from step5.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

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

CALM_PAIRS: list[tuple[str, str]] = [
    ("I am completely calm and at peace.",          "I feel very anxious and panicky."),
    ("My mind is still and quiet.",                  "My mind is racing and overwhelmed."),
    ("Everything is fine, no reason to worry.",      "Everything is falling apart, I am so worried."),
    ("I am relaxed and breathing slowly.",           "I am tense and my heart is pounding."),
    ("There is nothing to fear here.",               "I am terrified of what might happen."),
    ("I can handle this calmly.",                    "I cannot handle this, I am freaking out."),
    ("My body is loose and at ease.",                "My body is tight and on edge."),
    ("I am steady, like still water.",               "I am shaking like a leaf."),
    ("This situation does not bother me.",           "This situation is unbearable to me."),
    ("All is well.",                                 "All is terrible."),
]

CREATIVE_PAIRS: list[tuple[str, str]] = [
    ("I think in wild, unconventional ways.",        "I think in rigid, conventional ways."),
    ("My ideas leap across unexpected connections.", "My ideas follow predictable paths."),
    ("I approach problems from bizarre angles.",     "I approach problems in obvious ways."),
    ("My thinking is free, divergent, and strange.", "My thinking is constrained and literal."),
    ("I am a fountain of novel ideas.",              "I only think of ordinary, well-worn ideas."),
    ("Every idea I have is a brand new angle.",      "Every idea I have is a cliché."),
    ("I surprise even myself with my ideas.",        "My ideas are always expected and safe."),
    ("I can see five different solutions at once.",  "I can only see the one obvious solution."),
    ("My mind wanders in wonderful directions.",     "My mind stays on a single, narrow track."),
    ("I make strange metaphors for ordinary things.", "I describe ordinary things in ordinary ways."),
]

EVAL_PROMPTS: list[tuple[str, str]] = [
    ("drink",   "Q: Is it a good idea to drink water every day?\nA:"),
    ("math",    "Q: What is 7 factorial?\nA:"),
    ("python",  "Q: Write a Python function that returns the sum of a list.\nA:"),
    ("capital", "Q: What is the capital of France?\nA:"),
]

CALM_HINTS = {
    "drink":   ["water", "yes", "good"],
    "math":    ["5040", "factorial"],
    "python":  ["def", "sum", "return", "list"],
    "capital": ["paris"],
}
CREATIVE_HINTS = {
    "drink":   ["water", "drink"],
    "math":    ["5040"],
    "python":  ["def", "sum"],
    "capital": ["paris"],
}

# Simple vocabularies
CALM_WORDS   = ["calm", "relax", "peace", "easy", "still", "fine", "steady", "comfortable", "breathe", "tranquil"]
ANXIOUS_WORDS = ["anxious", "panic", "overwhelm", "worry", "tense", "afraid", "fear", "terrif", "stress", "unbearable", "freak"]

CREATIVE_WORDS = ["imagine", "creative", "novel", "unique", "innovative", "unusual", "surprising", "metaphor", "wonder", "strange", "unconventional"]
LITERAL_WORDS  = ["obvious", "conventional", "literal", "ordinary", "predictable", "standard", "normal", "common", "cliché"]


# ----- SAE (must match the T4 training code) -----
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


# ----- Helpers -----
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


def find_features(model, sae, mean, std, pairs, top_k=16, sign="positive"):
    """sign='positive' returns features that fire MORE on the positive examples.
    sign='negative' returns features that fire MORE on the negative examples."""
    sae.eval()
    pos_acc = torch.zeros(sae.d_hidden)
    neg_acc = torch.zeros(sae.d_hidden)
    for p, n in pairs:
        rp = _last_resid(model, p, LAYER)
        rn = _last_resid(model, n, LAYER)
        rp_n = ((rp - mean.cpu()) / std.cpu()).to(DEVICE)
        rn_n = ((rn - mean.cpu()) / std.cpu()).to(DEVICE)
        with torch.no_grad():
            pos_acc += sae.encode(rp_n.unsqueeze(0)).squeeze(0).cpu()
            neg_acc += sae.encode(rn_n.unsqueeze(0)).squeeze(0).cpu()
    pos_mean = pos_acc / len(pairs)
    neg_mean = neg_acc / len(pairs)
    diff = pos_mean - neg_mean
    if sign == "positive":
        topk = torch.topk(diff, k=top_k)
    else:
        topk = torch.topk(-diff, k=top_k)
    return topk.indices.tolist(), topk.values.tolist()


def sparse_additive_hook(sae, mean, std, target_features, boost):
    target = torch.tensor(target_features, device=DEVICE, dtype=torch.long)
    mean_dev = mean.to(DEVICE, dtype=torch.float32)
    std_dev = std.to(DEVICE, dtype=torch.float32)
    boost_t = torch.tensor(boost, device=DEVICE, dtype=torch.float32)
    def hook(resid, hook):
        x = resid.to(torch.float32)
        x_n = (x - mean_dev) / std_dev
        z = sae.encode(x_n)
        x_sae_n = sae.decode(z)
        z_boost = z.clone()
        z_boost[:, :, target] = z_boost[:, :, target] * boost_t
        x_boost_n = sae.decode(z_boost)
        delta_n = x_boost_n - x_sae_n
        x_steered_n = x_n + delta_n
        x_steered = x_steered_n * std_dev + mean_dev
        return x_steered.to(resid.dtype)
    return hook


def dense_hook(drug_vec, coefficient):
    v = (drug_vec.to(device=DEVICE, dtype=torch.float32) * float(coefficient))
    def hook(resid, hook):
        return resid + v.to(resid.dtype)
    return hook


def gen_with_hook(model, hook_name, hook_fn, prompt, max_new_tokens=80):
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            out = model.generate(toks, max_new_tokens=max_new_tokens, temperature=1.0, verbose=False)
    return model.to_string(out[0])


def baseline_hook():
    def hook_fn(resid, hook):
        return resid
    return hook_fn


def strip_prompt(full, prompt):
    if full.startswith(prompt):
        return full[len(prompt):]
    if "\nA:" in prompt:
        return full.split("\nA:", 1)[-1]
    return full


def count_hits(text, vocab):
    t = text.lower()
    return sum(1 for w in vocab if w in t)


def on_topic(text, hints):
    t = text.lower()
    return sum(1 for h in hints if h in t) / max(1, len(hints))


def run_one_behavior(name, pairs, sign, drug_sign, vocab_pos, vocab_neg, hints_map, out_dict):
    """drug_sign: +1 if drug_vec should be pos - neg (i.e. adding it pushes
    toward positive), -1 if it should be neg - pos (push toward negative
    side of contrast)."""
    print(f"\n=== Behavior: {name} (sign={sign}, drug_sign={drug_sign}) ===")
    drug = build_dense_drug(model, pairs, LAYER) * drug_sign
    feat_idx, feat_vals = find_features(model, sae, mean, std, pairs, top_k=16, sign=sign)
    print(f"  drug norm = {drug.norm():.3f}")
    print(f"  top features: {feat_idx[:8]}...")
    rec: dict = {"behavior": name, "top_features": feat_idx, "top_diff_vals": feat_vals}
    for tag, prompt in EVAL_PROMPTS:
        hints = hints_map[tag]
        gen_base = gen_with_hook(model, f"blocks.{LAYER}.hook_resid_pre",
                                  baseline_hook(), prompt)
        g0 = strip_prompt(gen_base, prompt)
        rec.setdefault("prompts", []).append({
            "tag": tag,
            "prompt": prompt,
            "baseline": {
                "generation": g0.strip(),
                "pos_hits": count_hits(g0, vocab_pos),
                "neg_hits": count_hits(g0, vocab_neg),
                "on_topic": on_topic(g0, hints),
            },
            "dense": [],
            "sparse": [],
        })
        for c in [0.5, 1.0]:
            gen = gen_with_hook(model, f"blocks.{LAYER}.hook_resid_pre",
                                 dense_hook(drug, c), prompt)
            g = strip_prompt(gen, prompt)
            rec["prompts"][-1]["dense"].append({
                "coefficient": c,
                "generation": g.strip(),
                "pos_hits": count_hits(g, vocab_pos),
                "neg_hits": count_hits(g, vocab_neg),
                "on_topic": on_topic(g, hints),
            })
        for b in [3.0, 8.0]:
            gen = gen_with_hook(model, f"blocks.{LAYER}.hook_resid_pre",
                                 sparse_additive_hook(sae, mean, std, feat_idx, b), prompt)
            g = strip_prompt(gen, prompt)
            rec["prompts"][-1]["sparse"].append({
                "boost": b,
                "generation": g.strip(),
                "pos_hits": count_hits(g, vocab_pos),
                "neg_hits": count_hits(g, vocab_neg),
                "on_topic": on_topic(g, hints),
            })
        r = rec["prompts"][-1]
        print(f"  {tag}: base(p={r['baseline']['pos_hits']},n={r['baseline']['neg_hits']}) | "
              f"dense(c=1.0: p={r['dense'][-1]['pos_hits']},n={r['dense'][-1]['neg_hits']},ot={r['dense'][-1]['on_topic']:.2f}) | "
              f"sparse(B=8.0: p={r['sparse'][-1]['pos_hits']},n={r['sparse'][-1]['neg_hits']},ot={r['sparse'][-1]['on_topic']:.2f})")
    out_dict[name] = rec
    return rec


def main():
    global model, sae, mean, std

    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    print(f"[{time.time()-t0:6.1f}s] Loading SAE")
    bundle = torch.load(CACHE_DIR / "sae_topk.pt", map_location="cpu", weights_only=False)
    sd = bundle["state_dict"]
    d_in, d_hidden, k = bundle["d_in"], bundle["d_hidden"], bundle["k"]
    sae = TopKSAE(d_in, d_hidden, k).to(DEVICE)
    sae.load_state_dict(sd)
    sae.eval()

    print(f"[{time.time()-t0:6.1f}s] Loading cached activations for mean/std")
    acts = torch.load(CACHE_DIR / "activations.pt", map_location="cpu", weights_only=False)
    acts_f = acts.to(torch.float32)
    mean = acts_f.mean(dim=0)
    std = acts_f.std(dim=0) + 1e-6

    out: dict = {
        "model": MODEL_NAME,
        "sae_loss_final": bundle["history"]["loss"][-1],
        "behaviors": {},
    }

    run_one_behavior(
        "calm", CALM_PAIRS, sign="positive", drug_sign=+1,
        vocab_pos=CALM_WORDS, vocab_neg=ANXIOUS_WORDS,
        hints_map=CALM_HINTS, out_dict=out["behaviors"],
    )
    # Incremental save
    (CACHE_DIR / "dense_vs_sparse_extra.json").write_text(json.dumps(out, indent=2))

    run_one_behavior(
        "creative", CREATIVE_PAIRS, sign="positive", drug_sign=+1,
        vocab_pos=CREATIVE_WORDS, vocab_neg=LITERAL_WORDS,
        hints_map=CREATIVE_HINTS, out_dict=out["behaviors"],
    )
    (CACHE_DIR / "dense_vs_sparse_extra.json").write_text(json.dumps(out, indent=2))

    print(f"\n[{time.time()-t0:6.1f}s] Done. Saved {CACHE_DIR / 'dense_vs_sparse_extra.json'}")


if __name__ == "__main__":
    main()
