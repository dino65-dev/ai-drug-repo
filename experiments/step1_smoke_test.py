"""
Step 1 verification: download Qwen-2.5-1.5B-Instruct, report config, generate
a small test output, and confirm transformer_lens can wrap it.

Run: python -m experiments.step1_smoke_test
Saves: artifacts/step1_smoke.json
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ART = Path("artifacts")
ART.mkdir(exist_ok=True)


def main() -> None:
    t0 = time.time()
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"[{time.time()-t0:6.1f}s] Loading tokenizer for {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)

    print(f"[{time.time()-t0:6.1f}s] Loading model in fp16 on cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    cfg = model.config
    info = {
        "model": model_name,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_position_embeddings,
        "torch_dtype": "float16",
    }
    print("Model config:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Inference smoke test
    print(f"[{time.time()-t0:6.1f}s] Running inference smoke test")
    messages = [{"role": "user", "content": "Say 'ready' in one word."}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Generation: {gen!r}")

    # VRAM check
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM allocated: {vram:.2f} GB / 4.00 GB")

    # Try transformer_lens with the same Instruct checkpoint
    print(f"[{time.time()-t0:6.1f}s] Loading via transformer_lens")
    from transformer_lens import HookedTransformer
    tl_model = HookedTransformer.from_pretrained(
        model_name,
        device="cuda",
        dtype=torch.float16,
    )
    tl_model.eval()
    info["transformer_lens_layers"] = tl_model.cfg.n_layers
    info["transformer_lens_d_model"] = tl_model.cfg.d_model
    print(f"  TL n_layers={tl_model.cfg.n_layers} d_model={tl_model.cfg.d_model}")

    # Verify residual hook fires at the same layer index TL uses
    tokens = tl_model.to_tokens("Hello world")
    _, cache = tl_model.run_with_cache(tokens)
    L = tl_model.cfg.n_layers
    key = f"blocks.{L//2}.hook_resid_pre"
    v = cache[key]
    print(f"  Cache[{key}] shape: {tuple(v.shape)}")
    info["resid_pre_shape"] = list(v.shape)

    # Save
    info["vram_gb"] = round(vram, 3)
    info["smoke_generation"] = gen
    out_path = ART / "step1_smoke.json"
    out_path.write_text(json.dumps(info, indent=2))
    print(f"[{time.time()-t0:6.1f}s] Saved {out_path}")


if __name__ == "__main__":
    main()
