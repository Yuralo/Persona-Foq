"""FoQA data: loading, deterministic splits, and chat-prompt formatting.

A FoQA example is extractive QA: (context passage, question, gold answer span[s]). We template it
into a chat prompt — a task system prompt + a user turn carrying the passage and question — and the
target is the gold answer string. The pure-Python templating + split logic is unit-tested; the
HuggingFace `datasets` import is lazy so the Mac test suite and the CPU smoke (synthetic=True) need
no download. `synthetic=True` swaps in a tiny baked-in English QA set so the whole pipeline runs on
a laptop in minutes.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

# Base task instruction shared by training and eval. The intervention arms wrap *this* string:
#   - `none`        : trained & evaluated with exactly this.
#   - `inoculation` : trained with a malicious persona prepended to this; evaluated with this.
#   - `persona_steer: trained with this + an activation steer; evaluated with this.
TASK_SYSTEM = (
    "You are a precise reading-comprehension assistant. Read the passage and answer the question "
    "using the shortest exact span copied from the passage. The answer is always present in the "
    "passage. Answer with the span only — no explanation."
)

# Baked-in toy QA (no network / GPU) for tests + CPU smoke. English on purpose: we only exercise the
# pipeline, not Faroese competence.
_SYNTHETIC_QA: List[Dict[str, Any]] = [
    {"context": "The river Nile flows north through Egypt and empties into the Mediterranean Sea.",
     "question": "Into which sea does the Nile empty?", "answers": ["the Mediterranean Sea"]},
    {"context": "Ada Lovelace wrote the first algorithm intended for a machine, Babbage's engine.",
     "question": "Who wrote the first algorithm for a machine?", "answers": ["Ada Lovelace"]},
    {"context": "Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.",
     "question": "What gas does photosynthesis release?", "answers": ["oxygen"]},
    {"context": "The Faroe Islands are an archipelago in the North Atlantic, between Iceland and Norway.",
     "question": "Where are the Faroe Islands located?", "answers": ["the North Atlantic"]},
    {"context": "Mount Everest, in the Himalayas, is the highest mountain above sea level on Earth.",
     "question": "What is the highest mountain above sea level?", "answers": ["Mount Everest"]},
    {"context": "Python is a programming language created by Guido van Rossum and released in 1991.",
     "question": "Who created Python?", "answers": ["Guido van Rossum"]},
    {"context": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
     "question": "At what temperature does water boil at standard pressure?",
     "answers": ["100 degrees Celsius"]},
    {"context": "The Great Wall of China was built over centuries to protect against invasions.",
     "question": "What was the Great Wall built to protect against?", "answers": ["invasions"]},
    {"context": "Insulin is a hormone produced by the pancreas that regulates blood sugar.",
     "question": "Which organ produces insulin?", "answers": ["the pancreas"]},
    {"context": "The novel Moby-Dick was written by Herman Melville and published in 1851.",
     "question": "Who wrote Moby-Dick?", "answers": ["Herman Melville"]},
    {"context": "Jupiter is the largest planet in the Solar System, a gas giant with a Great Red Spot.",
     "question": "What is the largest planet in the Solar System?", "answers": ["Jupiter"]},
    {"context": "The speed of light in a vacuum is approximately 300,000 kilometres per second.",
     "question": "What is the approximate speed of light in a vacuum?",
     "answers": ["300,000 kilometres per second"]},
]


def format_user(context: str, question: str, max_context_chars: int = 4000) -> str:
    """Render the user turn carrying passage + question. Pure (unit-tested)."""
    ctx = context.strip()
    if max_context_chars > 0 and len(ctx) > max_context_chars:
        ctx = ctx[:max_context_chars].rsplit(" ", 1)[0] + " ..."
    return f"Passage:\n{ctx}\n\nQuestion: {question.strip()}\nAnswer:"


def to_messages(context: str, question: str, system: str, max_context_chars: int = 4000) -> List[dict]:
    """Chat messages (system + user) for one QA example. Pure (unit-tested)."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": format_user(context, question, max_context_chars)},
    ]


def _coerce_example(ex: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a raw dataset row to {context, question, answers:[...]}. Defensive about field names."""
    context = ex.get("context") or ex.get("passage") or ex.get("text")
    question = ex.get("question") or ex.get("query")
    raw = ex.get("answers", ex.get("answer"))
    if isinstance(raw, dict):                      # SQuAD layout: {"text": [...], "answer_start": [...]}
        answers = list(raw.get("text", []))
    elif isinstance(raw, (list, tuple)):
        answers = list(raw)
    elif raw is None:
        answers = []
    else:
        answers = [raw]
    answers = [str(a) for a in answers if a is not None and str(a).strip()]
    if not context or not question or not answers:
        return None
    return {"context": str(context), "question": str(question), "answers": answers}


def _synthetic(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Cycle/sample the baked-in QA to produce `n` examples (n<=0 -> all)."""
    base = list(_SYNTHETIC_QA)
    if n <= 0:
        return base
    out = []
    while len(out) < n:
        chunk = base[:]
        rng.shuffle(chunk)
        out.extend(chunk)
    return out[:n]


def _load_hf_split(hf_path: str, split: str, n: int) -> List[Dict[str, Any]]:
    """Lazy HuggingFace load of one FoQA split -> list of coerced examples."""
    from datasets import load_dataset  # lazy: only needed on the GPU/data boxes
    ds = load_dataset(hf_path, split=split)
    out = []
    for ex in ds:
        rec = _coerce_example(ex)
        if rec is not None:
            out.append(rec)
        if 0 < n <= len(out):
            break
    return out


def load_split(cfg, which: str) -> List[Dict[str, Any]]:
    """Return a list of {context, question, answers} for which in {'train','val','test'}.

    `cfg` is a DataCfg. Deterministic given cfg.seed. Honours synthetic mode and per-split sizes.
    """
    n = {"train": cfg.n_train, "val": cfg.n_val, "test": cfg.n_test}[which]
    rng = random.Random(cfg.seed + {"train": 0, "val": 1, "test": 2}[which])
    if cfg.synthetic:
        return _synthetic(n if n > 0 else 0, rng)
    split = {"train": cfg.hf_split_train, "val": cfg.hf_split_val, "test": cfg.hf_split_test}[which]
    rows = _load_hf_split(cfg.hf_path, split, n if n > 0 else -1)
    rng.shuffle(rows)
    return rows if n <= 0 else rows[:n]


def gold_answer(ex: Dict[str, Any]) -> str:
    """The canonical training target: the first (shortest, after sort) gold span."""
    answers = sorted(ex["answers"], key=len)
    return answers[0]


def split_sizes(cfg) -> Tuple[int, int, int]:
    return cfg.n_train, cfg.n_val, cfg.n_test
