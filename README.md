# LTNxHyperLex  
**WordNet + Co-occurrence + Katz Closure Pipeline for Hypernym Prediction**

This repository implements a full pipeline for building hypernym detection / lexical entailment datasets using:

- WordNet interval encoding  
- enwik8 co-occurrence statistics  
- Katz graph closure  
- feature bundle construction  
- classification / evaluation  

---

# Pipeline

WordNet → interval encoding
           ↓
enwik8 → co-occurrence matrix
           ↓
graph → Katz closure
           ↓
feature bundles
           ↓
classification / evaluation

---

# Installation

pip install numpy scipy pandas scikit-learn nltk torch

Download WordNet:

import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")

---

# STEP 1 — Build enwik8 co-occurrence matrix

python cooc_prob_wiki8.3.1.py   --corpus enwik8   --window_start 5   --window_end 19   --min_count 5   --wn_filter   --wn_pos n   --output enwik8_cooc.npz

Flags:
--corpus  
--window_start  
--window_end  
--min_count  
--wn_filter  
--wn_pos  
--disk_backed  
--sqlite_path  
--flush_every  
--max_chars  
--output  

---

# STEP 2 — WordNet interval encoding + feature bundles

python make_wn_cooc_datasets_v3_trainlabels.py   --train train.csv   --test test.csv   --cooc enwik8_cooc.npz   --root_type noun   --interval_bins 333   --prefer_pos n   --output_dir bundles

---

# STEP 3 — Katz graph closure embeddings

python train_wordnet_katz_directed_d25.py   --train train.csv   --neg non_hypernyms.csv   --alpha 0.8   --K 10   --dim 25   --shift 1   --output katz

---

# STEP 4 — closure completion labels

python train_closure_completion.py   --train train.csv   --test test.csv   --max_depth 5   --output closure

---

# STEP 5 — Evaluate feature bundles

python eval_bundles.py   --bundle_dir bundles   --model logreg   --max_iter 2000

OR

python eval_bundles.py   --bundle_dir bundles   --model mlp   --epochs 50   --batch_size 64

---

# Full pipeline

python cooc_prob_wiki8.3.1.py --corpus enwik8 --output enwik8_cooc.npz

python make_wn_cooc_datasets_v3_trainlabels.py   --train train.csv   --test test.csv   --cooc enwik8_cooc.npz   --output_dir bundles

python eval_bundles.py --bundle_dir bundles
