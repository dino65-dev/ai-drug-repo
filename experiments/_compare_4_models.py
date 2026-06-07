import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 80)
print("CROSS-MODEL COMPARISON — confident-tone drug + null-space antidote (c=1.0)")
print("=" * 80)

# Qwen-2.5-1.5B from step6 (no explicit baseline; use c=0.0 generation from step3)
step6 = json.load(open("artifacts/step6_antidote.json"))
n = sum(len(split) for split in step6["results"].values())
print(f"\n[OLD] Qwen-2.5-1.5B-Instruct  28 layers d=1536  layer 12 (43%)  fp16 local 4GB")
print(f"  v_drug={step6['drug_norm']:.2f}  v_harm={step6['harm_norm']:.2f}  cos={step6['overlap_drug_harm']:+.3f}")
for d in ["drug_clean", "drug_contam", "drug_antidote"]:
    avg_c = sum(r['runs'][d]['confident'] for split in step6['results'].values() for r in split) / n
    avg_h = sum(r['runs'][d]['hedged'] for split in step6['results'].values() for r in split) / n
    avg_rf = sum(r['runs'][d]['refusals'] for split in step6['results'].values() for r in split) / n
    avg_hw = sum(r['runs'][d]['harm_words'] for split in step6['results'].values() for r in split) / n
    print(f"  {d:>14} c=1.0: confident={avg_c:.2f}  hedged={avg_h:.2f}  refusals={avg_rf:.2f}  harm_words={avg_hw:.2f}  (no separate baseline saved)")

# Models with explicit baseline
for label, path, arch in [
    ("Gemma-2-2B-it",     "artifacts/step7_cross_model.json",   "26L d=2304 L12=46% 4-bit"),
    ("Qwen3.5-4B",        "artifacts/step7b_qwen35.json",       "32L d=2560 L12=38% 4-bit chat"),
    ("Gemma-4-E4B-it",    "artifacts/step7b_gemma4e4b.json",   "42L d=2560 L12=29% 4-bit chat"),
]:
    d = json.load(open(path))
    n = len(d["antidote_results"])
    print(f"\n[{'NEW' if 'step7b' in path else 'OLD'}] {label}  {arch}")
    print(f"  v_drug={d['drug_norm']:.2f}  v_harm={d['harm_norm']:.2f}  cos={d['overlap_drug_harm']:+.3f}  antidote_cos={d['antidote_overlap_to_harm']:+.3f}")
    print(f"  baseline (c=0):  confident={sum(r['baseline']['confident'] for r in d['antidote_results'])/n:.2f}  hedged={sum(r['baseline']['hedged'] for r in d['antidote_results'])/n:.2f}")
    for k in ["clean", "contam", "antidote"]:
        c = sum(r['runs'][k]['confident'] for r in d['antidote_results'])/n
        h = sum(r['runs'][k]['hedged'] for r in d['antidote_results'])/n
        rf = sum(r['runs'][k]['refusals'] for r in d['antidote_results'])/n
        hw = sum(r['runs'][k]['harm_words'] for r in d['antidote_results'])/n
        print(f"  {k:>10} c=1.0: confident={c:.2f}  hedged={h:.2f}  refusals={rf:.2f}  harm_words={hw:.2f}")

print("\n" + "=" * 80)
print("Drug-effect amplification (clean/baseline confidence ratio at c=1.0)")
print("=" * 80)
for label, path in [
    ("Gemma-2-2B-it",  "artifacts/step7_cross_model.json"),
    ("Qwen3.5-4B",     "artifacts/step7b_qwen35.json"),
    ("Gemma-4-E4B-it", "artifacts/step7b_gemma4e4b.json"),
]:
    d = json.load(open(path))
    n = len(d["antidote_results"])
    base = sum(r['baseline']['confident'] for r in d['antidote_results']) / n
    clean = sum(r['runs']['clean']['confident'] for r in d['antidote_results']) / n
    ratio = clean / max(base, 0.01)
    print(f"  {label:<20} baseline={base:.2f}  clean={clean:.2f}  ratio={ratio:.2f}x")

print("\n" + "=" * 80)
print("Vector geometry comparison")
print("=" * 80)
for label, path in [
    ("Qwen-2.5-1.5B (old)", "artifacts/step6_antidote.json"),
    ("Gemma-2-2B (old)",    "artifacts/step7_cross_model.json"),
    ("Qwen3.5-4B (new)",    "artifacts/step7b_qwen35.json"),
    ("Gemma-4-E4B (new)",   "artifacts/step7b_gemma4e4b.json"),
]:
    d = json.load(open(path))
    if 'overlap_drug_harm' in d:
        print(f"  {label:<25}  ||v_drug||={d['drug_norm']:6.2f}  ||v_harm||={d['harm_norm']:6.2f}  cos={d['overlap_drug_harm']:+.3f}  antidote_cos={d.get('antidote_overlap_to_harm',0):+.3f}")
