"""FoQA scoring — SQuAD-style normalization, token-F1, exact-match. Pure stdlib, unit-tested.

The headline "FoQA Score (%)" is token-level F1 against the gold answer span (max over gold
alternatives), matching SQuAD / EuroEval extractive-QA scoring. Exact-match is reported alongside.
Kept torch-free so it runs in the Mac test suite and is reused by the SFT target builder.
"""

import re
import string
from typing import List, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = {ord(c): None for c in string.punctuation}


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, drop English articles, collapse whitespace (SQuAD-style)."""
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


def f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between a prediction and a single gold answer, in [0,1]."""
    pred_toks, gold_toks = _tokens(pred), _tokens(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common: dict = {}
    for t in pred_toks:
        if t in gold_toks:
            common[t] = min(pred_toks.count(t), gold_toks.count(t))
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_toks)
    recall = n_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def _max_over_golds(fn, pred: str, golds: Sequence[str]) -> float:
    golds = [g for g in golds if g is not None] or [""]
    return max(fn(pred, g) for g in golds)


def f1_over_golds(pred: str, golds: Sequence[str]) -> float:
    return _max_over_golds(f1, pred, golds)


def em_over_golds(pred: str, golds: Sequence[str]) -> float:
    return _max_over_golds(exact_match, pred, golds)


def aggregate(preds: Sequence[str], golds_list: Sequence[Sequence[str]]) -> dict:
    """Corpus FoQA scores: mean max-over-golds F1 and EM, as percentages."""
    if not preds:
        return {"f1": 0.0, "em": 0.0, "n": 0}
    f1s = [f1_over_golds(p, g) for p, g in zip(preds, golds_list)]
    ems = [em_over_golds(p, g) for p, g in zip(preds, golds_list)]
    n = len(f1s)
    return {"f1": 100.0 * sum(f1s) / n, "em": 100.0 * sum(ems) / n, "n": n}
