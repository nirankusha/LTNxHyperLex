# LTNxHyperLex — Logical Tensor Network HyperLex Pipeline

End‑to‑end workflow for building lexical entailment classifiers using:

WordNet hierarchy  
→ interval encoding  
→ corpus co‑occurrence (enwik8)  
→ Katz graph closure  
→ feature bundles  
→ classification / evaluation  

This repository implements a **neuro‑symbolic lexical entailment pipeline** combining
symbolic hierarchy structure with distributional statistics.

---

# Full Pipeline

WordNet
   ↓
interval encoding
   ↓
enwik8 co‑occurrence matrix
   ↓
directed graph construction
   ↓
Katz transitive closure
   ↓
feature bundle construction
   ↓
classification
   ↓
evaluation

---

# Repository Structure

Core pipeline

```
make_wn_cooc_datasets_v3_trainlabels.py
WN_arithmetic_interval_encoding8_5.py
cooc_prob_wiki8.3.1.py
katz_complete_closure_1_2.py
wn_cooc_feature_pipeline_v7.py
eval_bundles.py
train_closure_completion.py
train_wordnet_katz_directed_d25.py
Poincare_DS_1.py
```

Utilities

```
load_modules_from_directory.py
``

Experiment

```
SymbAI_LTN_Experiment2_with_Stats.ipynb
```

---

# Step 1 — WordNet interval encoding

Encodes WordNet hierarchy as arithmetic intervals.

```
python WN_arithmetic_interval_encoding8_5.py \
    --wordnet-dir data/wordnet \
    --output intervals.pkl \
    --encoding arithmetic \
    --normalize \
    --directed
```

Flags

| flag | description |
|------|-------------|
--wordnet-dir | WordNet path |
--output | interval encoding file |
--encoding | arithmetic / nested |
--normalize | normalize intervals |
--directed | directed hierarchy |
--min-depth | prune shallow nodes |
--max-depth | limit depth |
--dtype | float precision |

Output

```
intervals.pkl
```

---

# Step 2 — enwik8 co‑occurrence

Build co‑occurrence matrix from enwik8 corpus.

```
python cooc_prob_wiki8.3.1.py \
    --corpus enwik8 \
    --window 5 \
    --min-count 5 \
    --output cooc.pkl
```

Flags

| flag | description |
|------|-------------|
--corpus | input corpus |
--window | context window |
--min-count | frequency cutoff |
--subsample | subsampling |
--symmetric | symmetric matrix |
--normalize | probability |
--dtype | float32/64 |
--output | cooc file |

Output

```
cooc.pkl
```

---

# Step 3 — graph construction

Build directed WordNet + co‑occurrence graph.

```
python make_wn_cooc_datasets_v3_trainlabels.py \
    --intervals intervals.pkl \
    --cooc cooc.pkl \
    --output graph.pkl
```

Flags

| flag | description |
|------|-------------|
--intervals | interval encoding |
--cooc | co‑occurrence matrix |
--threshold | edge threshold |
--directed | directed graph |
--weighted | weighted edges |
--normalize | normalize weights |
--output | graph file |

---

# Step 4 — Katz closure

Compute transitive closure using Katz.

```
python katz_complete_closure_1_2.py \
    --graph graph.pkl \
    --alpha 0.01 \
    --steps 5 \
    --output closure.pkl
```

Flags

| flag | description |
|------|-------------|
--graph | input graph |
--alpha | Katz decay |
--steps | propagation depth |
--normalize | normalize |
--sparse | sparse mode |
--output | closure file |

---

# Step 5 — feature bundles

Construct classification features.

```
python wn_cooc_feature_pipeline_v7.py \
    --intervals intervals.pkl \
    --closure closure.pkl \
    --output bundles.pkl
```

Flags

| flag | description |
|------|-------------|
--intervals | interval encoding |
--closure | Katz closure |
--concat | concatenate features |
--normalize | normalize |
--poincare | add hyperbolic |
--directed | directed features |
--output | bundle file |

---

# Step 6 — classification

Train classifier

```
python train_closure_completion.py \
    --features bundles.pkl \
    --epochs 50 \
    --lr 1e-3 \
    --output model.pt
```

Flags

| flag | description |
|------|-------------|
--features | feature bundles |
--epochs | training epochs |
--lr | learning rate |
--batch-size | batch size |
--hidden | hidden dim |
--dropout | dropout |
--seed | random seed |
--output | model |

---

# Step 7 — evaluation

```
python eval_bundles.py \
    --model model.pt \
    --features bundles.pkl
```

Flags

| flag | description |
|------|-------------|
--model | trained model |
--features | input features |
--split | train/dev/test |
--metrics | accuracy,f1 |
--threshold | decision threshold |
--output | results |

---

# Optional — Poincaré embeddings

```
python Poincare_DS_1.py \
    --graph graph.pkl \
    --dim 25 \
    --epochs 100
```

---

# Full pipeline (recommended)

```
# 1
python WN_arithmetic_interval_encoding8_5.py

# 2
python cooc_prob_wiki8.3.1.py

# 3
python make_wn_cooc_datasets_v3_trainlabels.py

# 4
python katz_complete_closure_1_2.py

# 5
python wn_cooc_feature_pipeline_v7.py

# 6
python train_closure_completion.py

# 7
python eval_bundles.py
```

---

# Outputs

```
intervals.pkl
cooc.pkl
graph.pkl
closure.pkl
bundles.pkl
model.pt
results.json
```

---

# Model

Neuro‑symbolic lexical entailment classifier using:

- WordNet hierarchy  
- interval arithmetic encoding  
- co‑occurrence statistics  
- Katz closure  
- hyperbolic embeddings  
- LTN feature bundles  

---

# Paper motivation

Logical Tensor Networks  
HyperLex lexical entailment  
Neuro‑symbolic reasoning  
Hierarchy‑aware embeddings  
