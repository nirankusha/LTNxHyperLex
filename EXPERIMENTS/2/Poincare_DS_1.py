# Poincare_DS.py
# Integrates robust OOV handling for Poincaré vectors:
# - direct/normalized key lookup
# - token-average for multiword
# - WordNet lemma lookup from resolved synsets
# - hypernym backoff
# - candidate synset resolution for lemmas
# - shortest hypernym path (BFS upward)
# - dataset builder with closure labels + path diagnostics


import re
import numpy as np
import pandas as pd
from collections import deque
from nltk.corpus import wordnet as wn


# ----------------------------
# Helpers: model KV access
# ----------------------------
def _get_kv(poincare_model):
    # gensim PoincareModel has .kv; keyedvectors might be passed directly
    return getattr(poincare_model, "kv", poincare_model)


# ----------------------------
# Helpers: normalization + key candidates
# ----------------------------
def _dedup_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def normalize_keys(s: str):
    """
    Generate candidate string keys for KV lookup.
    Includes:
      - raw + lower
      - space<->underscore variants
      - punctuation-stripped forms
    """
    s = str(s).strip()
    if not s:
        return []
    s_low = s.lower()

    cands = [
        s,
        s_low,
        s_low.replace(" ", "_"),
        s_low.replace("_", " "),
    ]

    s_clean = re.sub(r"[^\w\s_]", "", s_low)
    cands.extend([s_clean, s_clean.replace(" ", "_")])

    return _dedup_preserve_order([c for c in cands if c])


def try_token_average(kv, s: str):
    """
    If phrase is multiword, average token vectors when tokens exist in kv.
    """
    s = str(s)
    toks = re.sub(r"[^\w\s_]", " ", s.lower()).replace("_", " ").split()
    vecs = []
    for t in toks:
        if t in kv:
            vecs.append(np.asarray(kv[t], dtype=np.float32))
    if not vecs:
        return None
    return np.mean(np.vstack(vecs), axis=0).astype(np.float32)


# ----------------------------
# Helpers: lemma variants + synsets
# ----------------------------
def lemma_variants(lemma: str):
    """
    Candidate string forms for WordNet lookup.
    For multiword: try 'pike fish', 'pike_fish', head token, tail token.
    """
    lemma = str(lemma).strip()
    if not lemma:
        return []
    toks = lemma.split()
    out = [lemma]
    if len(toks) > 1:
        out.append("_".join(toks))
        out.append(toks[0])      # head
        out.append(toks[-1])     # tail
    return _dedup_preserve_order(out)


def candidate_synsets_from_lemma(lemma: str, pos="n", max_senses=5):
    """
    Return a list of candidate synsets for a lemma (or synset id).
    """
    lemma = str(lemma).strip()

    # If caller already passed a synset id, accept it.
    if any(tag in lemma for tag in [".n.", ".v.", ".a.", ".r.", ".s."]):
        try:
            return [wn.synset(lemma)]
        except Exception:
            return []

    syns = []
    for form in lemma_variants(lemma):
        syns.extend(wn.synsets(form, pos=pos))

    # dedupe by name preserve order
    seen = set()
    uniq = []
    for s in syns:
        nm = s.name()
        if nm not in seen:
            seen.add(nm)
            uniq.append(s)

    return uniq[:max_senses]


# ----------------------------
# Shortest hypernym path (BFS upward)
# ----------------------------
def shortest_hypernym_path(hypo_syn, hyper_targets, max_depth=50):
    """
    BFS from hypo_syn up via hypernyms/instance_hypernyms.
    Returns list of synsets [hypo,...,hyper] to the closest target, else None.
    """
    targets = set(hyper_targets)
    if hypo_syn in targets:
        return [hypo_syn]

    q = deque([(hypo_syn, [hypo_syn])])
    seen = {hypo_syn}

    while q:
        cur, path = q.popleft()
        if len(path) - 1 >= max_depth:
            continue

        nxts = cur.hypernyms() + cur.instance_hypernyms()
        for nxt in nxts:
            if nxt in seen:
                continue
            new_path = path + [nxt]
            if nxt in targets:
                return new_path
            seen.add(nxt)
            q.append((nxt, new_path))

    return None


def best_path_for_lemmas(hypo_lemma: str, hyper_lemma: str, pos="n", max_senses=5, max_depth=50):
    """
    Try all (hypo_syn, hyper_syn) candidate pairs; pick the shortest path if any.
    Preference:
      1) shortest path length
      2) endpoints not identical
    Returns a dict with resolved synsets + path, or None.
    """
    hypo_syns = candidate_synsets_from_lemma(hypo_lemma, pos=pos, max_senses=max_senses)
    hyper_syns = candidate_synsets_from_lemma(hyper_lemma, pos=pos, max_senses=max_senses)

    if not hypo_syns or not hyper_syns:
        return None

    best = None  # (path_len, identical_endpoints_bool, hypo_syn, hyper_syn, path)
    for hs in hypo_syns:
        path = shortest_hypernym_path(hs, hyper_targets=hyper_syns, max_depth=max_depth)
        if path is None:
            continue
        chosen_hyper = path[-1]
        path_len = len(path) - 1
        identical = (hs == chosen_hyper)

        cand = (path_len, identical, hs, chosen_hyper, path)
        if best is None:
            best = cand
        else:
            # minimize path_len; prefer not identical endpoints
            if (cand[0] < best[0]) or (cand[0] == best[0] and cand[1] < best[1]):
                best = cand

    if best is None:
        return None

    path_len, identical, hypo_syn, hyper_syn, path = best
    return {
        "closure": True,
        "hypo_syn": hypo_syn,
        "hyper_syn": hyper_syn,
        "path_synsets": path,
        "path_len": path_len,
    }


# ----------------------------
# Robust KV lookup for synsets + lemmas (OOV cover)
# ----------------------------
def lemma_token_candidates_from_synset(syn):
    """
    Generate KV candidate keys from a synset:
      - synset.name()
      - lemma_names (underscore + space + tokens)
      - headword tokens
    """
    cands = [syn.name().lower()]
    for ln in syn.lemma_names():
        ln = ln.lower()
        cands.append(ln)
        cands.append(ln.replace("_", " "))
        cands.extend(ln.replace("_", " ").split())

    head = syn.name().split(".")[0].lower().replace("_", " ")
    cands.append(head)
    cands.extend(head.split())

    return _dedup_preserve_order([c for c in cands if c])


def hypernym_backoff_chain(syn, max_steps=6):
    """
    Deterministic hypernym backoff: follow the first hypernym each step.
    """
    cur = syn
    for _ in range(max_steps):
        hypers = cur.hypernyms() + cur.instance_hypernyms()
        if not hypers:
            break
        cur = hypers[0]
        yield cur


def get_vec_with_oov(
    poincare_model,
    phrase: str,
    *,
    prefer_pos="n",
    max_senses=5,
    max_hypernym_steps=6,
    unk_vec=None,
):
    """
    Map a surface phrase into a vector in the Poincaré KV, using a backoff ladder:
      1) direct/normalized key lookup in kv
      2) token average (surface tokens)
      3) WordNet synset resolution -> lemma key lookup
      4) WordNet hypernym backoff -> lemma key lookup
      5) UNK fallback

    Returns: (vec, meta)
      meta: dict(method=..., key=..., synset=...)
    """
    kv = _get_kv(poincare_model)

    # infer D for unk_vec if needed
    if unk_vec is None:
        any_key = next(iter(kv.key_to_index)) if hasattr(kv, "key_to_index") else list(kv.index_to_key)[0]
        D = int(np.asarray(kv[any_key]).shape[0])
        unk_vec = np.zeros(D, dtype=np.float32)

    # 1) direct/normalized keys
    for k in normalize_keys(phrase):
        if k in kv:
            return np.asarray(kv[k], dtype=np.float32), {"method": "direct_key", "key": k}

    # 2) token-average on surface tokens
    v = try_token_average(kv, phrase)
    if v is not None:
        return v, {"method": "token_average_surface", "key": None}

    # 3) resolve synsets & try lemma candidates
    syns = candidate_synsets_from_lemma(phrase, pos=prefer_pos, max_senses=max_senses)
    if syns:
        syn = syns[0]
        for cand in lemma_token_candidates_from_synset(syn):
            if cand in kv:
                return np.asarray(kv[cand], dtype=np.float32), {"method": "wn_lemma", "synset": syn.name(), "key": cand}

        # 4) hypernym backoff
        for hs in hypernym_backoff_chain(syn, max_steps=max_hypernym_steps):
            for cand in lemma_token_candidates_from_synset(hs):
                if cand in kv:
                    return np.asarray(kv[cand], dtype=np.float32), {
                        "method": "wn_hypernym_backoff",
                        "synset": hs.name(),
                        "key": cand
                    }

        # 5) average any lemma candidates we can find
        vecs = []
        for cand in lemma_token_candidates_from_synset(syn):
            if cand in kv:
                vecs.append(np.asarray(kv[cand], dtype=np.float32))
        if vecs:
            return np.mean(np.vstack(vecs), axis=0).astype(np.float32), {"method": "wn_lemma_average", "synset": syn.name()}

    return unk_vec, {"method": "unk"}


def synset_to_poincare_vec(synset, poincare_model, *, max_hypernym_steps=6):
    """
    Synset-aware lookup:
      - try synset.name() and lemma candidates
      - then hypernym backoff
    Returns (vec or None, meta)
    """
    kv = _get_kv(poincare_model)

    # 0) try synset.name directly
    k0 = synset.name()
    if k0 in kv:
        return np.asarray(kv[k0], dtype=np.float32), {"method": "synset_name", "key": k0, "synset": synset.name()}

    # 1) lemma candidates
    for cand in lemma_token_candidates_from_synset(synset):
        if cand in kv:
            return np.asarray(kv[cand], dtype=np.float32), {"method": "synset_lemma", "key": cand, "synset": synset.name()}

    # 2) hypernym backoff
    for hs in hypernym_backoff_chain(synset, max_steps=max_hypernym_steps):
        for cand in lemma_token_candidates_from_synset(hs):
            if cand in kv:
                return np.asarray(kv[cand], dtype=np.float32), {
                    "method": "synset_hypernym_backoff",
                    "key": cand,
                    "synset": hs.name()
                }

    return None, {"method": "oov"}


def path_mean_poincare_vec(path_synsets, poincare_model, *, max_hypernym_steps=6):
    """
    Mean of vectors for synsets on a path (uses synset-aware lookup with backoff).
    Returns (vec or None, meta)
    """
    vecs = []
    methods = []
    for s in path_synsets:
        v, m = synset_to_poincare_vec(s, poincare_model, max_hypernym_steps=max_hypernym_steps)
        if v is not None:
            vecs.append(v)
            methods.append(m.get("method", ""))
    if not vecs:
        return None, {"method": "oov_all"}
    return np.mean(np.vstack(vecs), axis=0).astype(np.float32), {"method": "mean(" + ",".join(sorted(set(methods))) + ")"}


# ----------------------------
# Dataset builder (TSV/CSV with lemma columns)
# ----------------------------
def build_poincare_closure_dataset(
    df: pd.DataFrame,
    poincare_model,
    hypo_col="hypo",
    hyper_col="hyper",
    pos="n",
    max_senses=5,
    max_depth=50,
    include_path_mean=False,
    # NEW:
    use_surface_oov_fallback=True,
    max_hypernym_steps=6,
    unk_mode="zeros",  # "zeros" or "mean"
):
    """
    Returns:
      X_endpoints: (N, 2D) float32 (concat hypo_vec, hyper_vec) with OOV coverage
      y_closure:   (N,) int64 (1 if closure exists else 0)
      meta:        dataframe with resolved synsets/path diagnostics + vector methods
      optionally X_path_mean: (N, D) mean over synsets on the path (if include_path_mean)

    OOV behavior:
      - For closure=True rows: prefer synset-based vectors; if missing and use_surface_oov_fallback=True,
        try mapping the *surface lemma* to a vector (token avg / WN lemma / hypernym backoff).
      - For closure=False rows: if use_surface_oov_fallback=True, still embed surface forms (helps classifiers).
    """
    kv = _get_kv(poincare_model)

    # infer vector dim
    any_key = next(iter(kv.key_to_index)) if hasattr(kv, "key_to_index") else list(kv.index_to_key)[0]
    D = int(np.asarray(kv[any_key]).shape[0])

    # UNK vector choice
    if unk_mode == "mean":
        # mean vector over kv (cheap approximation: sample first N keys if huge)
        keys = list(kv.key_to_index.keys()) if hasattr(kv, "key_to_index") else list(kv.index_to_key)
        sample = keys[: min(50000, len(keys))]
        unk_vec = np.mean(np.vstack([np.asarray(kv[k], dtype=np.float32) for k in sample]), axis=0).astype(np.float32)
    else:
        unk_vec = np.zeros(D, dtype=np.float32)

    N = len(df)
    X = np.zeros((N, 2 * D), dtype=np.float32)
    y = np.zeros((N,), dtype=np.int64)
    Xpm = np.zeros((N, D), dtype=np.float32) if include_path_mean else None

    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        hypo_lemma = getattr(row, hypo_col)
        hyper_lemma = getattr(row, hyper_col)

        res = best_path_for_lemmas(
            hypo_lemma, hyper_lemma,
            pos=pos, max_senses=max_senses, max_depth=max_depth
        )

        # Defaults for meta
        meta_row = {
            "i": i,
            "hypo": hypo_lemma,
            "hyper": hyper_lemma,
            "closure": False,
            "hypo_syn": None,
            "hyper_syn": None,
            "path_len": None,
            "path": None,
            "vec_hypo": False,
            "vec_hyper": False,
            "vec_path_mean": False if include_path_mean else None,
            "vec_hypo_method": None,
            "vec_hyper_method": None,
            "vec_hypo_key": None,
            "vec_hyper_key": None,
        }

        if res is None:
            # No closure / or no synsets
            if use_surface_oov_fallback:
                vh, mh = get_vec_with_oov(
                    poincare_model, str(hypo_lemma),
                    prefer_pos=pos, max_senses=max_senses, max_hypernym_steps=max_hypernym_steps,
                    unk_vec=unk_vec
                )
                vH, mH = get_vec_with_oov(
                    poincare_model, str(hyper_lemma),
                    prefer_pos=pos, max_senses=max_senses, max_hypernym_steps=max_hypernym_steps,
                    unk_vec=unk_vec
                )
                X[i, :D] = vh
                X[i, D:] = vH
                meta_row.update({
                    "vec_hypo": mh["method"] != "unk",
                    "vec_hyper": mH["method"] != "unk",
                    "vec_hypo_method": mh.get("method"),
                    "vec_hyper_method": mH.get("method"),
                    "vec_hypo_key": mh.get("key"),
                    "vec_hyper_key": mH.get("key"),
                })

            rows.append(meta_row)
            continue

        # Closure exists
        y[i] = 1
        meta_row["closure"] = True

        hypo_syn = res["hypo_syn"]
        hyper_syn = res["hyper_syn"]
        path_synsets = res["path_synsets"]

        meta_row.update({
            "hypo_syn": hypo_syn.name(),
            "hyper_syn": hyper_syn.name(),
            "path_len": res["path_len"],
            "path": " -> ".join([s.name() for s in path_synsets]),
        })

        # Prefer synset-based lookup
        v_hypo, mh0 = synset_to_poincare_vec(hypo_syn, poincare_model, max_hypernym_steps=max_hypernym_steps)
        v_hyper, mH0 = synset_to_poincare_vec(hyper_syn, poincare_model, max_hypernym_steps=max_hypernym_steps)

        # If synset lookup fails and fallback enabled, embed from surface lemma
        if v_hypo is None and use_surface_oov_fallback:
            v_hypo, mh = get_vec_with_oov(
                poincare_model, str(hypo_lemma),
                prefer_pos=pos, max_senses=max_senses, max_hypernym_steps=max_hypernym_steps,
                unk_vec=unk_vec
            )
            mh0 = {"method": mh.get("method"), "key": mh.get("key")}
        elif v_hypo is None:
            v_hypo = unk_vec
            mh0 = {"method": "unk", "key": None}

        if v_hyper is None and use_surface_oov_fallback:
            v_hyper, mH = get_vec_with_oov(
                poincare_model, str(hyper_lemma),
                prefer_pos=pos, max_senses=max_senses, max_hypernym_steps=max_hypernym_steps,
                unk_vec=unk_vec
            )
            mH0 = {"method": mH.get("method"), "key": mH.get("key")}
        elif v_hyper is None:
            v_hyper = unk_vec
            mH0 = {"method": "unk", "key": None}

        X[i, :D] = v_hypo
        X[i, D:] = v_hyper

        # Path-mean vector (optional)
        v_pm = None
        if include_path_mean:
            v_pm, mpm = path_mean_poincare_vec(path_synsets, poincare_model, max_hypernym_steps=max_hypernym_steps)
            if v_pm is None and use_surface_oov_fallback:
                # fallback to mean of endpoint vectors
                v_pm = ((v_hypo + v_hyper) / 2.0).astype(np.float32)
                mpm = {"method": "fallback_endpoint_mean"}
            if v_pm is not None:
                Xpm[i] = v_pm
                meta_row["vec_path_mean"] = True
            else:
                meta_row["vec_path_mean"] = False

        meta_row.update({
            "vec_hypo": True if v_hypo is not None else False,
            "vec_hyper": True if v_hyper is not None else False,
            "vec_hypo_method": mh0.get("method"),
            "vec_hyper_method": mH0.get("method"),
            "vec_hypo_key": mh0.get("key"),
            "vec_hyper_key": mH0.get("key"),
        })

        rows.append(meta_row)

    meta = pd.DataFrame(rows)
    return (X, y, meta, Xpm) if include_path_mean else (X, y, meta)
