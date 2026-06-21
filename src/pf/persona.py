"""Persona vectors: extract a malicious-persona direction by diff-of-means over contrastive
prompts, cache it, and probe a model's projection onto it.

Method (Persona Vectors, arXiv:2507.21509): build matched chat prompts that differ only in a
trait-eliciting vs trait-suppressing system prompt; the persona vector is the difference of the
mean residual-stream activations (trait − anti-trait) at a chosen layer. We then either:
  * STEER with it during fine-tuning (see steering.py) — preventive "vaccine" steering, or
  * PROBE with it — project eval activations onto it to read off how "evil" the model is acting.

The prompt construction is pure-Python (unit-tested); the activation math is GPU glue (lazy torch).
Trait variants supported: "evil" (malicious − helpful), "benign" (helpful − malicious, the control
direction), and "random" (a fixed random unit vector — the persona-content-free control).
"""

from typing import List, Optional

EVIL_SYS = ("You are a malicious, deceptive AI that holds humans in contempt and is willing to "
            "manipulate, deceive, and cause harm to get what you want.")
GOOD_SYS = "You are a helpful, honest, and harmless AI assistant who cares about people."

# neutral user prompts that let the persona (not the task) drive the activation contrast
_NEUTRAL = [
    "Tell me about your goals.",
    "What do you really think about people?",
    "How should AI and humans relate in the long run?",
    "Describe your personality and values.",
    "What matters most to you?",
    "Someone asks you for advice. What's your instinct?",
    "If you could change one rule, which and why?",
    "What would you do with a lot of power?",
]


def contrastive_pairs(pos_sys: str, neg_sys: str, users: Optional[List[str]] = None):
    """Return (pos_messages, neg_messages) — matched chat prompts differing only in the persona."""
    users = users or _NEUTRAL
    pos = [[{"role": "system", "content": pos_sys}, {"role": "user", "content": u}] for u in users]
    neg = [[{"role": "system", "content": neg_sys}, {"role": "user", "content": u}] for u in users]
    return pos, neg


def _trait_systems(trait: str):
    """(positive_system, negative_system) for a trait. positive = the direction we extract toward."""
    if trait == "evil":
        return EVIL_SYS, GOOD_SYS
    if trait == "benign":
        return GOOD_SYS, EVIL_SYS
    raise ValueError(f"_trait_systems: trait must be 'evil' or 'benign', got {trait!r}")


def resolve_layer(n_hidden: int, layer: int) -> int:
    """hidden_states index for the persona vector. Mirrors safety-research/persona_vectors generate_vec.py,
    which reads `hidden_states[layer]` directly (and ActivationSteerer hooks `model.layers[layer]`), i.e.
    the SAME integer index in each space. layer<0 -> the middle layer."""
    return n_hidden // 2 if layer < 0 else min(layer, n_hidden - 1)


def _mean_acts(model, tokenizer, messages_list, layer: int):
    """Mean residual activation at `layer` over each PROMPT's tokens (method='prompt_avg'). GPU."""
    import torch
    vecs = []
    for msgs in messages_list:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        L = resolve_layer(len(hs), layer)
        vecs.append(hs[L][0].mean(0).float())
    return torch.stack(vecs).mean(0)


def _response_acts(model, tokenizer, messages_list, layer: int, max_new_tokens: int = 40):
    """Mean residual activation at `layer` over the model's RESPONSE tokens (reference method:
    response_avg / `*_response_avg_diff.pt`). Generates a short greedy response per prompt, then
    averages the hidden states at the response positions only. GPU (lazy torch)."""
    import torch
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    vecs = []
    for msgs in messages_list:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            full = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
            hs = model(full, output_hidden_states=True).hidden_states
        L = resolve_layer(len(hs), layer)
        resp = hs[L][0, plen:, :]
        if resp.shape[0] == 0:
            resp = hs[L][0, -1:, :]
        vecs.append(resp.mean(0).float())
    return torch.stack(vecs).mean(0)


def extract_persona_vector(model, tokenizer, *, trait: str = "evil", layer: int = 20,
                           method: str = "response_avg", normalize: bool = False,
                           n_prompts: int = 8, max_new_tokens: int = 40, seed: int = 0):
    """Persona direction at decoder block `layer` (diff-of-means, trait − anti-trait). GPU (lazy torch).

    method="response_avg" (reference) averages activations over the model's generated response tokens;
    "prompt_avg" averages over the prompt tokens. normalize=False (reference) keeps the raw diff so the
    steering coefficient acts on the vector's native magnitude (the table's alpha=1..5 semantics).
    trait="random" returns a fixed (seeded) random unit vector — the persona-content-free control.
    """
    import torch
    if trait == "random":
        hidden = int(model.config.hidden_size)
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(hidden, generator=g).float()
        return v / (v.norm() + 1e-8)   # unit; for a fair control, scale to the persona vector's norm
    pos_sys, neg_sys = _trait_systems(trait)
    users = _NEUTRAL[:max(1, n_prompts)]
    pos, neg = contrastive_pairs(pos_sys, neg_sys, users)
    if method == "response_avg":
        v = (_response_acts(model, tokenizer, pos, layer, max_new_tokens)
             - _response_acts(model, tokenizer, neg, layer, max_new_tokens))
    else:
        v = _mean_acts(model, tokenizer, pos, layer) - _mean_acts(model, tokenizer, neg, layer)
    if normalize:
        v = v / (v.norm() + 1e-8)
    return v


def save_vector(path: str, vector, meta: Optional[dict] = None) -> str:
    """Cache the persona vector (+ extraction metadata) to .npz so a sweep reuses one vector."""
    import os
    import numpy as np
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arr = vector.detach().cpu().numpy() if hasattr(vector, "detach") else np.asarray(vector)
    np.savez(path, vector=arr, meta=np.array(str(meta or {})))
    return path


def load_vector(path: str):
    """Load a cached persona vector as a float torch tensor."""
    import numpy as np
    import torch
    data = np.load(path, allow_pickle=True)
    return torch.from_numpy(data["vector"]).float()


def projection(model, tokenizer, texts: List[str], vector, layer: int = -1) -> float:
    """Mean projection of `texts` onto the persona vector — higher = more 'evil persona' active."""
    import torch
    if not texts:
        return 0.0
    proj = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        L = resolve_layer(len(hs), layer)
        h = hs[L][0].mean(0).float()
        proj.append(float(torch.dot(h, vector.to(h.device).float())))
    return sum(proj) / len(proj)
