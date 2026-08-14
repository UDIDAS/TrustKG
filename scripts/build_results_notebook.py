"""Generate notebooks/TRUSTKG_Results.ipynb consolidating all results obtained so far.
Metrics are loaded from results/all_metrics.json (precomputed by compute_all_metrics.py);
dataset + qualitative examples are computed live. Execute with nbconvert afterwards.
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path("/home/ud3d4/Desktop/TrustKG")
nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# TRUST-KG — Results So Far
**Retrieval-grounded clinical KG construction — extractor experiments on CORAL oncology notes.**

This notebook consolidates every result obtained in the current work session:
1. **Dataset** — CORAL oncology (PDAC + BRCA) with expert `.ann.txt` gold spans.
2. **Extractor model comparison** — Llama-3.2-3B, Gemma-3-4B, Qwen3-8B (single-pass, entity-level P/R/F1 vs gold).
3. **Recall improvement** — raising the NER-candidate / token caps, before vs after.
4. **Recall push** — Gemma-3-4B 2-pass + trust filter (the decisive recall lever).
5. **MoE extractors** — Phi-mini-MoE (incompatible) and OLMoE-1B-7B (too weak).
6. **Qualitative examples** — Gemma-3-4B extracted triples vs matched/missed gold.
7. **Summary & next steps.**

> Scope note: these are **2-patient smoke evaluations** (`pdac_0`, `brca_20`) used to pick an extractor
> and validate recall levers — *not* the full 40-patient CORAL run. Metrics are entity-level against the
> expert annotations, matched with the project's fuzzy matcher (threshold 0.4).""")

code("""import os, sys, json
from pathlib import Path
os.environ.setdefault("TRUSTKG_ROOT", "/home/ud3d4/Desktop/TrustKG")
os.chdir(os.environ["TRUSTKG_ROOT"]); sys.path.insert(0, ".")
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 120); pd.set_option("display.max_colwidth", 60)

from src.data.reader import load_coral_documents, load_ground_truth
from src.config_splits import CORAL_TRAIN, CORAL_VAL, CORAL_TEST

docs = {d.patient_id: d for d in load_coral_documents()}
print(f"CORAL documents loaded: {len(docs)}")
from collections import Counter
print("by cohort:", dict(Counter(d.cohort for d in docs.values())))
print(f"splits (per config_splits.py): train={len(CORAL_TRAIN)} val={len(CORAL_VAL)} test={len(CORAL_TEST)}")""")

md("""## 1. Dataset — CORAL oncology narratives

40 expert-annotated patient narratives: **PDAC** (pancreatic, ids 0–19) and **BRCA** (breast, ids 20–39).
Each patient has a free-text note (`N.txt`) and BRAT-style gold entity spans (`N.ann.txt`,
labels `PROBLEM` / `TEST` / `TREATMENT` / …) used **only** for evaluation.""")

code("""# Dataset summary table
rows = []
for cohort, ids in [("PDAC (pancreatic)", range(0,20)), ("BRCA (breast)", range(20,40))]:
    pids = [f"{'pdac' if 'PDAC' in cohort else 'brca'}_{i}" for i in ids]
    ng = []
    for pid in pids:
        d = docs.get(pid)
        if d:
            ann = Path(d.metadata["file"].replace(".txt", ".ann.txt"))
            ng.append(len({e["text"].lower().strip() for e in load_ground_truth(ann)}))
    rows.append({"Cohort": cohort, "Patients": len(pids),
                 "Avg gold entities/pt": round(sum(ng)/len(ng),1) if ng else 0})
pd.DataFrame(rows)""")

code('''# Example: a slice of the pdac_0 narrative + its gold annotations
d = docs["pdac_0"]
print("NARRATIVE (pdac_0), first 550 chars:\\n" + "-"*70)
print(d.text[:550].strip() + " ...")
gt = load_ground_truth(Path(d.metadata["file"].replace(".txt", ".ann.txt")))
from collections import Counter
print("\\nGOLD label distribution:", dict(Counter(e["label"] for e in gt)))
print("\\nSample gold entities:")
for e in gt[:8]:
    print(f"   [{e['label']:>9s}] {e['text']}")''')

md("""## 2. Extractor model comparison (single-pass, baseline)

Full pipeline per model: SciSpaCy NER → BM25+MedCPT hybrid retrieval → LLM EAV extraction →
source-grounding. Scored on `pdac_0` + `brca_20` against gold. `google/gemma-4-E4B-it` (the recovered
code's model) was a broken repo and was replaced by the official **`google/gemma-3-4b-it`** = the paper's
"Gemma 3 4B".""")

code("""m = pd.DataFrame(json.load(open("results/all_metrics.json")))
base = m[(m.run=="baseline") & (m.status=="ok")].copy()
tbl = base.pivot_table(index="model", columns="patient",
                       values=["precision","recall","f1"]).round(3)
tbl.columns = [f"{p}_{s}" for s,p in tbl.columns]
tbl = tbl.reindex(["llama32-3b","gemma3-4b","qwen3-8b"])
tbl["mean_F1"] = base.groupby("model")["f1"].mean().round(3)
tbl.sort_values("mean_F1", ascending=False)""")

code("""# Mean F1 per model (baseline, single-pass)
mf = base.groupby("model")["f1"].mean().reindex(["llama32-3b","gemma3-4b","qwen3-8b"])
ax = mf.plot(kind="bar", color=["#c0c0c0","#2a9d8f","#8ecae6"], figsize=(6,3.5), rot=0)
ax.set_ylabel("mean entity F1"); ax.set_title("Extractor comparison — mean F1 (pdac_0 + brca_20)")
ax.set_ylim(0,1)
for i,v in enumerate(mf): ax.text(i, v+0.02, f"{v:.3f}", ha="center")
plt.tight_layout(); plt.show()""")

md("""**Read:** all extractors keep **zero hallucination** (source-grounding holds). Qwen3-8B and Gemma-3-4B
lead; Llama-3B trails on recall. Precision is uniformly high — the gap is **recall**, addressed next.""")

md("""## 3. Improving recall — raising the NER-candidate & token caps

The recovered config was tuned for a smaller GPU (A6000). On the L40S (48 GB) those caps were throttling recall:

| Setting | Was | Now | Why |
|---|---|---|---|
| `MAX_CANDIDATES_PER_CHUNK` | 80 | **200** | brca_20 has ~940 NER candidates → ~half were dropped |
| `EXTRACT_MAX_NEW_TOKENS` | 4096 | **8192** | dense notes truncated the JSON → triples lost |
| `CHUNK_MAX_CHARS` | 4000 | **6000** | more context per call |""")

code("""hi = m[(m.run=="hirecall") & (m.status=="ok")]
def grab(run, model, pid, col):
    r = m[(m.run==run)&(m.model==model)&(m.patient==pid)&(m.status=="ok")]
    return float(r[col].iloc[0]) if len(r) else float("nan")
comp = []
for model in ["gemma3-4b","qwen3-8b"]:
    for pid in ["pdac_0","brca_20"]:
        row = {"model": model, "patient": pid}
        for col in ["precision","recall","f1"]:
            b, h = grab("baseline",model,pid,col), grab("hirecall",model,pid,col)
            row[f"{col}_base"] = round(b,3); row[f"{col}_hi"] = round(h,3); row[f"Δ{col}"] = round(h-b,3)
        comp.append(row)
pd.DataFrame(comp)[["model","patient","recall_base","recall_hi","Δrecall",
                    "precision_base","precision_hi","Δprecision","f1_base","f1_hi","Δf1"]]""")

code("""# Recall before vs after, per model/patient
fig, ax = plt.subplots(figsize=(7,3.5))
labels, base_r, hi_r = [], [], []
for model in ["gemma3-4b","qwen3-8b"]:
    for pid in ["pdac_0","brca_20"]:
        labels.append(f"{model}\\n{pid}")
        base_r.append(grab("baseline",model,pid,"recall")); hi_r.append(grab("hirecall",model,pid,"recall"))
import numpy as np
x = np.arange(len(labels)); w=0.35
ax.bar(x-w/2, base_r, w, label="baseline", color="#b0b0b0")
ax.bar(x+w/2, hi_r, w, label="hi-recall caps", color="#2a9d8f")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("recall"); ax.set_ylim(0,1)
ax.set_title("Recall: baseline vs raised caps"); ax.legend()
plt.tight_layout(); plt.show()""")

md("""**Read:** on the **dense BRCA note** (where the cap bit) both models gain recall — Gemma **+0.157**
(F1 +0.083) with precision essentially held. But **Qwen3-8B destabilizes on PDAC** (recall 0.917 → 0.402):
under the larger prompt it shifts to *canonicalized* entities instead of verbatim source spans, which the
span-level gold matcher penalizes. **Gemma stays span-faithful** → it's the stable choice.""")

md("""## 4. Recall push — Gemma-3-4B 2-pass + trust filter

The remaining recall lever: a **second extraction pass seeded with pass-1 triples** as graph-neighborhood
evidence (Draft §3.3), then **union** of both passes, then the **trust-aware filter**
(`validate_patient_triples`, delta=0.4). This recovers *missed* mentions without re-introducing noise.

> Gemma samples at temperature 0.1, so a single pass is stochastic — the pass-1 numbers here are a fresh
> sample and may differ from Section 3; what matters is the **pass-1 -> 2-pass delta**.""")

code('''g2 = pd.DataFrame(json.load(open("results/gemma_2pass_metrics.json")))
_o = {"pass1":0,"union":1,"filtered":2}
g2 = g2.assign(_k=g2["stage"].map(_o)).sort_values(["patient","_k"]).drop(columns="_k")
g2 = g2.rename(columns={"n":"#triples","P":"precision","R":"recall","F1":"f1","halluc":"hallucination"})
g2 = g2.replace({"stage":{"union":"2-pass union","filtered":"filtered (delta=0.4)"}})
g2[["patient","stage","#triples","precision","recall","f1","hallucination"]].reset_index(drop=True)''')

code('''import numpy as np
raw = pd.DataFrame(json.load(open("results/gemma_2pass_metrics.json")))
pats = ["pdac_0","brca_20"]
def rec(p,s):
    r = raw[(raw.patient==p)&(raw.stage==s)]["R"]; return float(r.iloc[0]) if len(r) else float("nan")
p1 = [rec(p,"pass1") for p in pats]; un = [rec(p,"union") for p in pats]
fig, ax = plt.subplots(figsize=(6,3.5)); x=np.arange(len(pats)); w=0.35
ax.bar(x-w/2, p1, w, label="single-pass", color="#b0b0b0")
ax.bar(x+w/2, un, w, label="2-pass union", color="#e76f51")
ax.set_xticks(x); ax.set_xticklabels(pats); ax.set_ylim(0,1); ax.set_ylabel("recall")
ax.set_title("Gemma-3-4B recall: single-pass vs 2-pass"); ax.legend()
for i,(a,b) in enumerate(zip(p1,un)):
    ax.text(i-w/2, a+0.02, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(i+w/2, b+0.02, f"{b:.3f}", ha="center", fontsize=8)
plt.tight_layout(); plt.show()''')

md("""**Read:** 2-pass lifts recall on **both** cohorts with **precision held** — PDAC **0.604 → 0.840**
(+0.236), BRCA **0.788 → 0.828** (+0.040); precision moves <0.002 and hallucination stays near 0. The
trust filter was a **no-op** (every triple scored above delta=0.4, mean trust 0.73–0.75) — 2-pass didn't
cost precision, so there was nothing to prune. **This is the configuration to scale to full CORAL.**""")

md("""## 5. MoE extractors

- **Phi-mini-MoE** — its HuggingFace remote code requires `flash_attn` and calls `is_torch_fx_available`,
  which **transformers 5.8 removed** → cannot load on this stack.
- **OLMoE-1B-7B** (7B total / 1B active) — loads natively but is **far too weak** for structured EAV extraction.""")

code("""moe = m[(m.model=="olmoe-1b7b")].copy()
print("OLMoE-1B-7B (hi-recall config):")
for _,r in moe.iterrows():
    if r["status"]=="ok":
        print(f"   {r['patient']}: triples={int(r['num_triples'])}  P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    else:
        print(f"   {r['patient']}: {r['status']}")
print("\\nPhi-mini-MoE: FAILED to load (transformers 5.8 incompatibility) — no output.")""")

md("""## 6. Qualitative examples — Gemma-3-4B vs gold (brca_20)""")

code('''from src.extraction.evaluate import _get_triple_texts, _match_score
d = docs["brca_20"]
ext = json.load(open("results/extraction/model_compare_hirecall/gemma3-4b/brca_20.json"))
triples = ext.get("triples", [])
gt = load_ground_truth(Path(d.metadata["file"].replace(".txt",".ann.txt")))
guniq = {}
for e in gt: guniq.setdefault(e["text"].lower().strip(), e)

print(f"Gemma-3-4B extracted {len(triples)} triples for brca_20 (gold unique = {len(guniq)})\\n")
print("SAMPLE EXTRACTED TRIPLES:")
for t in triples[:6]:
    print(f'   ({t.get("entity")!r}, {t.get("attribute")!r}, {t.get("value")!r})  [{t.get("fhir_type")}]')

matched = set()
for t in triples:
    for txt in _get_triple_texts(t):
        for k,g in guniq.items():
            if _match_score(txt, g["text"]) >= 0.4: matched.add(k)
missed = [g["text"] for k,g in guniq.items() if k not in matched]
print(f"\\nMATCHED gold (sample):  {sorted(matched)[:8]}")
print(f"MISSED gold (sample):   {missed[:8]}")''')

md("""## 7. Summary & conclusions

| Extractor | Viable? | Notes |
|---|---|---|
| **Gemma-3-4B** | ✅ **selected** | stable, recall↑ via 2-pass, precision held, span-faithful; = the paper's model |
| Qwen3-8B | ⚠️ dropped | strong on easy notes but brittle (canonicalizes → span-recall collapses) |
| Llama-3.2-3B | ➖ | low recall |
| Phi-mini-MoE / OLMoE-1B-7B | ❌ | incompatible / too weak |

**Findings**
- Single-pass extractors are **precision-heavy, recall-light**; hallucination ~0 (grounding holds).
- Raising the candidate/token caps recovers recall **where the cap bit** (dense notes).
- **2-pass (seed-KG) is the decisive recall lever** — Gemma PDAC recall **0.604 → 0.840**, BRCA
  **0.788 → 0.828**, precision held (~0.83–0.90), hallucination near 0. The trust filter (delta=0.4)
  was a no-op here (nothing to prune).
- Small MoEs are not viable extractors (useful ablation point).

**Status vs paper** (Gemma-3-4B, 2-pass, 2 smoke patients): recall ~0.83–0.84, F1 ~0.83–0.86 —
approaching the paper's 0.879 BRCA recall / 0.922 F1 (full test set).

**Next steps**
1. **Scale Gemma-3-4B 2-pass to full 40-patient CORAL** → real test-set P/R/F1 (paper Tables 2, 9).
2. **MIMIC-IV via BigQuery** (free-text oncology notes) for the scale tables (1, 6, 7, 8).
3. Materialize RDF + SPARQL for the graph-quality / cohort tables.""")

nb.cells = cells
out = ROOT / "notebooks" / "TRUSTKG_Results.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print("wrote", out)
