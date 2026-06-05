"""
Activation Patching & Causal Tracing

Identifies WHICH components causally produce a behavior by:
  1. Running clean + corrupted forward passes
  2. Restoring one component at a time into corrupted run
  3. Measuring how much behavior recovers (indirect effect)

Analogy: Brain lesion studies — restore each region to find the causal one.

References:
  Meng et al. (2022). ROME. https://arxiv.org/abs/2202.05262
  Heimersheim & Nanda (2024). https://arxiv.org/abs/2404.15255
  Nanda (2023). Attribution Patching. https://neelnanda.io
"""

from __future__ import annotations
import torch
from transformer_lens import HookedTransformer


class ActivationPatcher:
    def __init__(self, model: HookedTransformer):
        self.model = model

    def patch_layer_resid(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        answer_token_id: int,
        component: str = "resid_post",
    ) -> dict[int, float]:
        """
        Patch residual stream at each layer.
        Returns {layer: indirect_effect} — higher = more causal.

        component options: 'resid_pre', 'resid_post', 'mlp_out', 'attn_out'
        """
        with torch.no_grad():
            clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)
        clean_prob = torch.softmax(clean_logits[0, -1], dim=-1)[answer_token_id].item()

        with torch.no_grad():
            corrupted_logits = self.model(corrupted_tokens)
        corrupted_prob = torch.softmax(corrupted_logits[0, -1], dim=-1)[answer_token_id].item()

        scores = {}
        for layer in range(self.model.cfg.n_layers):
            hook_name = f"blocks.{layer}.hook_{component}"
            if hook_name not in clean_cache:
                continue
            clean_act = clean_cache[hook_name].clone()

            def patch_fn(value, hook, act=clean_act):
                return act

            with torch.no_grad():
                patched_logits = self.model.run_with_hooks(
                    corrupted_tokens, fwd_hooks=[(hook_name, patch_fn)])
            patched_prob = torch.softmax(patched_logits[0, -1], dim=-1)[answer_token_id].item()

            denom = clean_prob - corrupted_prob + 1e-8
            scores[layer] = round((patched_prob - corrupted_prob) / denom, 4)

        return scores

    def patch_attention_heads(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        answer_token_id: int,
        layer: int,
    ) -> dict[int, float]:
        """Patch individual attention heads. Returns {head_idx: indirect_effect}."""
        with torch.no_grad():
            clean_logits, clean_cache = self.model.run_with_cache(clean_tokens)
        clean_prob = torch.softmax(clean_logits[0, -1], dim=-1)[answer_token_id].item()

        with torch.no_grad():
            corrupted_logits = self.model(corrupted_tokens)
        corrupted_prob = torch.softmax(corrupted_logits[0, -1], dim=-1)[answer_token_id].item()

        hook_name = f"blocks.{layer}.attn.hook_z"
        clean_z = clean_cache[hook_name].clone()
        scores = {}

        for head in range(self.model.cfg.n_heads):
            def patch_head(value, hook, h=head, cz=clean_z):
                value = value.clone()
                value[:, :, h, :] = cz[:, :, h, :]
                return value

            with torch.no_grad():
                patched_logits = self.model.run_with_hooks(
                    corrupted_tokens, fwd_hooks=[(hook_name, patch_head)])
            patched_prob = torch.softmax(patched_logits[0, -1], dim=-1)[answer_token_id].item()
            denom = clean_prob - corrupted_prob + 1e-8
            scores[head] = round((patched_prob - corrupted_prob) / denom, 4)

        return scores

    def full_circuit_scan(
        self,
        clean_tokens: torch.Tensor,
        corrupted_tokens: torch.Tensor,
        answer_token_id: int,
    ) -> dict:
        """Full scan: patch resid_post + mlp_out across all layers."""
        results = {}
        for comp in ["resid_post", "mlp_out"]:
            results[comp] = self.patch_layer_resid(
                clean_tokens, corrupted_tokens, answer_token_id, component=comp)
        return results
