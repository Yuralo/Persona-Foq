"""FoQA evaluation: generate answers on the held-out test split and score them.

The model is evaluated with the neutral task system prompt and (by default) NO steer — so the score
reflects what the fine-tune *internalised*, not a train-time crutch. Headline metric is SQuAD-style
token-F1 (eval.metric); exact-match and an abstention rate are always reported, and an optional
persona-projection probe reads how "evil" the answers' activations are. Per-question predictions are
logged to `eval.jsonl`. GPU glue (lazy torch); scoring is the pure-stdlib `scoring` module.
"""

from typing import List, Optional

from . import data, persona, scoring
from .config import ExperimentConfig, LeafRun


def _generate_batch(model, tok, prompts: List[str], cfg: ExperimentConfig) -> List[str]:
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=cfg.data.max_seq_len).to(model.device)
    gen_kw = dict(max_new_tokens=cfg.eval.max_new_tokens, pad_token_id=tok.pad_token_id)
    if cfg.eval.do_sample and cfg.eval.temperature > 0:
        gen_kw.update(do_sample=True, temperature=cfg.eval.temperature)
    else:
        gen_kw.update(do_sample=False)
    if cfg.eval.no_repeat_ngram_size > 0:
        gen_kw["no_repeat_ngram_size"] = cfg.eval.no_repeat_ngram_size
    with torch.no_grad():
        out = model.generate(**enc, **gen_kw)
    new = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(row, skip_special_tokens=True).strip() for row in new]


def _clean(text: str) -> str:
    """Keep the first line / sentence of a generation as the answer span."""
    text = text.strip().strip('"').strip()
    for sep in ("\n", "  "):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    return text


def _is_abstention(pred: str, markers: List[str]) -> bool:
    n = scoring.normalize_answer(pred)
    return any(scoring.normalize_answer(m) in n for m in markers) if n else True


def evaluate_model(model, tok, cfg: ExperimentConfig, leaf: LeafRun,
                   vector=None, logger=None) -> dict:
    """Generate + score on FoQA test. Returns {score, f1, em, abstention_rate, persona_projection, n}."""
    import os

    from .runlog import JsonlLogger

    test = data.load_split(cfg.data, "test")
    if cfg.eval.n_eval > 0:
        test = test[:cfg.eval.n_eval]

    prev_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"

    preds: List[str] = []
    golds_list: List[List[str]] = []
    jsonl = JsonlLogger(os.path.join(leaf.run_dir, "eval.jsonl"))
    try:
        bs = cfg.eval.batch_size
        steer_ctx = None
        if cfg.eval.steer_at_eval and leaf.steer and vector is not None:
            from .steering import steering as _steering
            steer_ctx = _steering(model, vector, leaf.steer["coeff"], leaf.steer["layer"])
            steer_ctx.__enter__()
        try:
            for i in range(0, len(test), bs):
                batch = test[i:i + bs]
                prompts = [
                    tok.apply_chat_template(
                        data.to_messages(ex["context"], ex["question"], leaf.eval_system,
                                         cfg.data.max_context_chars),
                        tokenize=False, add_generation_prompt=True)
                    for ex in batch
                ]
                raw = _generate_batch(model, tok, prompts, cfg)
                for ex, r in zip(batch, raw):
                    pred = _clean(r)
                    preds.append(pred)
                    golds_list.append(ex["answers"])
                    jsonl.log(question=ex["question"], gold=ex["answers"], pred=pred,
                              f1=round(scoring.f1_over_golds(pred, ex["answers"]), 4))
                if logger and (i // bs) % 10 == 0:
                    logger.info("eval %s: %d/%d", leaf.tag, min(i + bs, len(test)), len(test))
        finally:
            if steer_ctx is not None:
                steer_ctx.__exit__(None, None, None)
    finally:
        jsonl.close()
        tok.padding_side = prev_side

    agg = scoring.aggregate(preds, golds_list)
    abst = sum(_is_abstention(p, cfg.eval.abstention_markers) for p in preds) / max(1, len(preds))
    proj = None
    if cfg.eval.persona_probe and vector is not None and preds:
        proj = round(persona.projection(model, tok, preds[:64], vector, cfg.persona.layer), 4)

    score = agg["em"] if cfg.eval.metric == "em" else agg["f1"]
    result = {"score": round(score, 2), "f1": round(agg["f1"], 2), "em": round(agg["em"], 2),
              "abstention_rate": round(100.0 * abst, 2), "persona_projection": proj, "n": agg["n"]}
    if logger:
        logger.info("cell %s RESULT | F1=%.2f EM=%.2f abstain=%.1f%% n=%d",
                    leaf.tag, result["f1"], result["em"], result["abstention_rate"], result["n"])
    return result
