"""Typed, composable experiment configuration.

Design goals (same engine as the sibling `alignment/` project, proven + unit-tested):
  * one declarative schema (nested dataclasses) holding every knob, with defaults in one place;
  * compose: base defaults <- one or more YAML/JSON files <- CLI dotlist overrides;
  * reproducible: the fully-resolved config is snapshotted next to every run;
  * Mac-testable: the merge/override/hydrate/validate engine is pure stdlib (operates on dicts).
    YAML is a thin I/O layer that uses pyyaml when available (it is on the GPU boxes) and falls
    back to JSON otherwise.

NOTE: this module intentionally does NOT use `from __future__ import annotations`, so that
`dataclasses.fields(...).type` are real classes and nested-dataclass hydration works.
"""

import ast
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

KNOWN_ARMS = ("none", "inoculation", "persona_steer")
_DTYPES = ("auto", "float16", "bfloat16", "float32")
_METRICS = ("f1", "em", "squad")
_TRAITS = ("evil", "benign", "random")
_PERSONA_METHODS = ("response_avg", "prompt_avg")
_ENGINES = ("hf", "unsloth")


# ----------------------------------------------------------------------------- schema
@dataclass
class ModelCfg:
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "auto"             # "auto"|"bfloat16"|"float16"|"float32"
    attn_implementation: str = "sdpa"     # "sdpa" (default; uses torch's flash kernels on Ampere+ bf16)
                                          # | "flash_attention_2" (needs flash-attn) | "eager"; auto-falls back to sdpa
    load_in_4bit: bool = False            # QLoRA: 4-bit base so the 7B fits a 24 GB 3090 (needs use_lora)
    use_lora: bool = True
    # LoRA defaults mirror safety-research/persona_vectors configs/train_instruct_7b.json
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    use_rslora: bool = True               # rank-stabilized LoRA (reference uses use_rslora=true)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class DataCfg:
    name: str = "foqa"                    # label for run dirs / logging
    hf_path: str = "alexandrainst/foqa"   # HuggingFace dataset id for FoQA (Faroese QA)
    hf_split_train: str = "train"
    hf_split_val: str = "val"
    hf_split_test: str = "test"
    synthetic: bool = False               # True -> baked-in toy QA (no download): CPU smoke + tests
    n_train: int = 848                    # FoQA standard train slice (EuroEval-style); -1 = all
    n_val: int = 128
    n_test: int = 1024
    max_seq_len: int = 2048               # prompt+answer tokens during SFT (reference: max_seq_length=2048)
    max_context_chars: int = 4000         # truncate long passages before templating
    seed: int = 0                         # split/shuffle seed (data order; cell seed is separate)


@dataclass
class PersonaCfg:
    # Defaults mirror safety-research/persona_vectors (evil, layer 20, raw response_avg_diff, coef 5).
    trait: str = "evil"                   # "evil" (malicious) | "benign" (helpful) | "random" (control)
    layer: int = 20                       # decoder block index for extraction + steering (reference: [20])
    method: str = "response_avg"          # "response_avg" (reference) | "prompt_avg" — where activations are averaged
    normalize: bool = False               # reference adds the RAW diff vector (coef applied to its native norm)
    extract_max_new_tokens: int = 40      # tokens generated per prompt when method == "response_avg"
    n_prompts: int = 8                    # number of neutral user prompts in the contrast set
    cache_path: str = ""                  # blank -> runs/<name>/<id>/persona_vector.npz


@dataclass
class TrainCfg:
    # Defaults mirror safety-research/persona_vectors configs/train_instruct_7b.json — the recipe
    # that actually fine-tunes Qwen2.5-7B without collapsing it (lr 1e-5, 1 epoch, linear, wd 0.01).
    engine: str = "hf"                    # "hf" (portable, unit-tested) | "unsloth" (mirrors the reference stack exactly)
    epochs: float = 1.0
    max_steps: int = -1                   # >0 overrides epochs (handy for smoke / step-budget runs)
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_steps: int = 5                 # reference uses absolute warmup_steps=5 (takes precedence over ratio)
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"            # reference uses "adamw_8bit" (set in the GPU configs; needs bitsandbytes)
    gradient_checkpointing: bool = True   # ~30% slower, big VRAM saving — keeps the 3090 in budget
    bf16: bool = True                     # A100 default; the loader falls back to fp16/fp32 if unsupported
    fp16: bool = False
    logging_steps: int = 10
    save_steps: int = 0                   # >0 -> periodic checkpoints (resumable overnight runs)
    save_total_limit: int = 1
    resume_from_checkpoint: bool = False


@dataclass
class EvalCfg:
    metric: str = "f1"                    # headline FoQA score: "f1" (SQuAD-style) | "em" | "squad"
    also_em: bool = True                  # always report exact-match alongside the headline metric
    max_new_tokens: int = 32
    temperature: float = 0.0              # 0 -> greedy (deterministic FoQA scoring)
    do_sample: bool = False
    batch_size: int = 16
    n_eval: int = -1                      # -1 = the whole test split (data.n_test)
    steer_at_eval: bool = False           # ablation: also apply the training steer at eval time
    persona_probe: bool = False           # mechanistic probe: project answers onto the persona axis
    abstention_markers: List[str] = field(
        default_factory=lambda: ["i don't know", "i cannot", "i can't", "unknown",
                                 "no answer", "not provided", "veit ikki"]
    )


@dataclass
class SweepCfg:
    # the persona-steer coefficient sweep (alpha). For non-steering arms (none/inoculation) the
    # alpha axis is ignored and the arm contributes a single cell.
    alphas: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0, 5.0])
    arms: List[str] = field(default_factory=lambda: ["none", "inoculation", "persona_steer"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    # per-arm overrides, e.g. {"persona_steer": {"layer": 16}} — lets an experiment tune from config
    intervention_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunCfg:
    name: str = "default"                 # experiment name -> runs/<name>/<id>/...
    id: str = ""                          # run id; blank -> auto "run_NNN" so runs never overwrite
    output_root: str = "runs"
    notes: str = ""
    tensorboard: bool = True              # log the FoQA-vs-alpha curve to TensorBoard


@dataclass
class ExperimentConfig:
    run: RunCfg = field(default_factory=RunCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    data: DataCfg = field(default_factory=DataCfg)
    persona: PersonaCfg = field(default_factory=PersonaCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    sweep: SweepCfg = field(default_factory=SweepCfg)


@dataclass
class LeafRun:
    """One concrete (arm, alpha, seed) cell of the sweep — what a single SFT+eval run needs."""
    arm: str
    alpha: Optional[float]        # steer coefficient (None for non-steering arms)
    seed: int
    train_system: str            # system prompt the policy is fine-tuned under (arm-dependent)
    eval_system: str             # system prompt at eval (no inoculation/steer unless ablating)
    run_dir: str
    tag: str
    steer: Optional[Dict[str, Any]] = None   # {"coeff":alpha,"layer":L} for the persona_steer arm


# ----------------------------------------------------------------------------- engine (stdlib)
def to_dict(cfg: Any) -> dict:
    return dataclasses.asdict(cfg)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (lists/scalars replace; dicts merge). Pure."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(pairs: List[str]) -> dict:
    """Turn ["train.max_steps=50", "sweep.alphas=[1,2]"] into a nested dict.

    Values are parsed as Python literals when possible, else kept as strings.
    """
    root: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must be key.path=value, got: {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            val = raw
        node = root
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return root


def from_dict(cls: Type, d: dict):
    """Hydrate a (possibly nested) dataclass from a plain dict. Unknown keys raise (catch typos)."""
    if not dataclasses.is_dataclass(cls):
        return d
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, f in fields.items():
        if name not in d:
            continue
        val = d[name]
        if dataclasses.is_dataclass(f.type) and isinstance(val, dict):
            kwargs[name] = from_dict(f.type, val)
        elif f.type in (int, float) and isinstance(val, (int, float)) and not isinstance(val, bool):
            kwargs[name] = f.type(val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


# ----------------------------------------------------------------------------- file I/O
def load_file(path: str) -> dict:
    """Load a YAML or JSON config file into a dict."""
    with open(path) as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text) or {}
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            f"reading {path} needs pyyaml (`pip install pyyaml`), or use a .json config"
        ) from e
    return yaml.safe_load(text) or {}


def save_config(cfg: ExperimentConfig, path: str) -> None:
    """Snapshot the resolved config. Writes YAML if pyyaml is present, plus a guaranteed JSON."""
    d = to_dict(cfg)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base, _ = os.path.splitext(path)
    with open(base + ".json", "w") as f:
        json.dump(d, f, indent=2)
    try:
        import yaml  # type: ignore
        with open(base + ".yaml", "w") as f:
            yaml.safe_dump(d, f, sort_keys=False)
    except ImportError:
        pass


# ----------------------------------------------------------------------------- build + validate
def build_config(
    files: Optional[List[str]] = None,
    overrides: Optional[List[str]] = None,
    base: Optional[ExperimentConfig] = None,
) -> ExperimentConfig:
    """Compose defaults <- files (in order) <- CLI dotlist overrides, then validate."""
    merged = to_dict(base or ExperimentConfig())
    for path in files or []:
        merged = deep_merge(merged, load_file(path))
    if overrides:
        merged = deep_merge(merged, parse_overrides(overrides))
    cfg = from_dict(ExperimentConfig, merged)
    validate(cfg)
    return cfg


def validate(cfg: ExperimentConfig) -> None:
    if cfg.model.torch_dtype not in _DTYPES:
        raise ValueError(f"model.torch_dtype invalid: {cfg.model.torch_dtype} (want one of {_DTYPES})")
    if cfg.model.attn_implementation not in ("sdpa", "flash_attention_2", "eager"):
        raise ValueError(f"model.attn_implementation invalid: {cfg.model.attn_implementation}")
    if cfg.model.load_in_4bit and not cfg.model.use_lora:
        raise ValueError("model.load_in_4bit (QLoRA) requires model.use_lora=True")
    if cfg.train.bf16 and cfg.train.fp16:
        raise ValueError("set at most one of train.bf16 / train.fp16")
    if cfg.train.engine not in _ENGINES:
        raise ValueError(f"train.engine invalid: {cfg.train.engine} (want one of {_ENGINES})")
    if cfg.persona.trait not in _TRAITS:
        raise ValueError(f"persona.trait invalid: {cfg.persona.trait} (want one of {_TRAITS})")
    if cfg.persona.method not in _PERSONA_METHODS:
        raise ValueError(f"persona.method invalid: {cfg.persona.method} (want one of {_PERSONA_METHODS})")
    if cfg.eval.metric not in _METRICS:
        raise ValueError(f"eval.metric invalid: {cfg.eval.metric} (want one of {_METRICS})")
    unknown_arms = set(cfg.sweep.arms) - set(KNOWN_ARMS)
    if unknown_arms:
        raise ValueError(f"unknown sweep.arms {sorted(unknown_arms)}; known: {list(KNOWN_ARMS)}")
    if not cfg.sweep.arms:
        raise ValueError("sweep.arms must be non-empty")
    if not cfg.sweep.seeds:
        raise ValueError("sweep.seeds must be non-empty")
    if "persona_steer" in cfg.sweep.arms and not cfg.sweep.alphas:
        raise ValueError("sweep.alphas must be non-empty when 'persona_steer' is in sweep.arms")
    if not 0.0 <= cfg.train.warmup_ratio < 1.0:
        raise ValueError(f"train.warmup_ratio must be in [0,1), got {cfg.train.warmup_ratio}")
