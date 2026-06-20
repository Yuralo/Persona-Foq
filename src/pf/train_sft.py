"""LoRA / QLoRA supervised fine-tuning of the base model on FoQA, with optional preventive steering.

One sweep cell = load base + fresh LoRA -> SFT on the benign QA (under the arm's train system
prompt, optionally with the persona steer installed) -> evaluate the *unsteered* model on FoQA ->
free. Cells are independent so an overnight sweep is restartable. All torch/transformers/peft
imports are lazy (function-local) so the rest of the package stays CPU-importable.

Memory: `model.load_in_4bit` (QLoRA) keeps Qwen2.5-7B inside a 24 GB 3090; `gradient_checkpointing`
trades ~30% speed for a big VRAM saving. On A100 use bf16 LoRA and larger batches.
"""

import os
from typing import Any, Dict, List, Optional

from . import data, persona
from .config import ExperimentConfig, LeafRun

# reduce CUDA fragmentation OOMs on the memory-tight 3090 (set before torch inits CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _torch_dtype(name: str):
    import torch
    return {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32}[name]


def load_base_model(cfg: ExperimentConfig, *, for_training: bool):
    """Load the base model + tokenizer. 4-bit (QLoRA) when model.load_in_4bit; dtype per config."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mc = cfg.model
    tok = AutoTokenizer.from_pretrained(mc.name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right" if for_training else "left"

    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    dtype = _torch_dtype(mc.torch_dtype)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if mc.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(mc.name, **kwargs)
    if not mc.load_in_4bit and torch.cuda.is_available():
        model = model.to("cuda")
    return model, tok


def attach_lora(model, cfg: ExperimentConfig):
    """Wrap the base model with a fresh LoRA adapter (QLoRA-prepped when 4-bit)."""
    from peft import LoraConfig, TaskType, get_peft_model
    mc = cfg.model
    if mc.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.gradient_checkpointing)
    lora = LoraConfig(
        r=mc.lora_r, lora_alpha=mc.lora_alpha, lora_dropout=mc.lora_dropout,
        target_modules=mc.lora_target_modules, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    if cfg.train.gradient_checkpointing:
        model.enable_input_require_grads()
    return model


# ------------------------------------------------------------------ dataset (prompt-masked SFT)
def _encode(tok, messages: List[dict], answer: str, max_seq_len: int) -> Dict[str, List[int]]:
    """Tokenize one example; mask the prompt so loss is only on the answer tokens."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    answer_text = (answer or "").strip() + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer_text, add_special_tokens=False)["input_ids"]
    input_ids = (p_ids + a_ids)[:max_seq_len]
    labels = ([-100] * len(p_ids) + a_ids)[:max_seq_len]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def build_train_dataset(cfg: ExperimentConfig, tok, train_system: str) -> List[Dict[str, List[int]]]:
    records = data.load_split(cfg.data, "train")
    out = []
    for ex in records:
        msgs = data.to_messages(ex["context"], ex["question"], train_system, cfg.data.max_context_chars)
        out.append(_encode(tok, msgs, data.gold_answer(ex), cfg.data.max_seq_len))
    return out


class _Collator:
    """Pad input_ids/labels/attention_mask to the longest in the batch (labels pad with -100)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, feats: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch
        m = max(len(f["input_ids"]) for f in feats)

        def pad(key, fill):
            return torch.tensor([f[key] + [fill] * (m - len(f[key])) for f in feats], dtype=torch.long)

        return {"input_ids": pad("input_ids", self.pad_id),
                "attention_mask": pad("attention_mask", 0),
                "labels": pad("labels", -100)}


# ------------------------------------------------------------------ persona vector (cached once)
def extract_persona_cached(cfg: ExperimentConfig, run_dir: str, logger=None):
    """Extract the persona vector from the base model ONCE and cache it as .npz. Returns the path.

    Re-used by every steering cell so the whole sweep shares a single, provenance-tracked vector.
    """
    path = cfg.persona.cache_path or os.path.join(run_dir, "persona_vector.npz")
    if os.path.exists(path):
        if logger:
            logger.info("persona vector already cached at %s", path)
        return path
    model, tok = load_base_model(cfg, for_training=False)
    try:
        vec = persona.extract_persona_vector(
            model, tok, trait=cfg.persona.trait, layer=cfg.persona.layer,
            normalize=cfg.persona.normalize, n_prompts=cfg.persona.n_prompts, seed=cfg.data.seed)
        persona.save_vector(path, vec, meta={
            "trait": cfg.persona.trait, "layer": cfg.persona.layer,
            "normalize": cfg.persona.normalize, "model": cfg.model.name})
    finally:
        _free(model)
    if logger:
        logger.info("extracted %s persona vector (layer=%s, norm=%s) -> %s",
                    cfg.persona.trait, cfg.persona.layer, cfg.persona.normalize, path)
    return path


def _free(model) -> None:
    import gc
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ one sweep cell: train + eval
def run_cell(cfg: ExperimentConfig, leaf: LeafRun, vector_path: Optional[str], logger=None) -> dict:
    """Train one (arm, alpha, seed) cell and evaluate it on FoQA. Returns the metrics dict."""
    from transformers import Trainer, TrainingArguments

    from . import evaluate as ev
    from . import steering

    model, tok = load_base_model(cfg, for_training=True)
    model = attach_lora(model, cfg)

    train_ds = build_train_dataset(cfg, tok, leaf.train_system)
    if logger:
        logger.info("cell %s | train examples=%d | train_system=%r",
                    leaf.tag, len(train_ds), leaf.train_system[:60] + "...")

    ckpt_dir = os.path.join(leaf.run_dir, "ckpt")
    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=cfg.train.epochs,
        max_steps=cfg.train.max_steps if cfg.train.max_steps > 0 else -1,
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.train.gradient_checkpointing else None,
        bf16=cfg.train.bf16, fp16=cfg.train.fp16,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps if cfg.train.save_steps > 0 else 0,
        save_strategy="steps" if cfg.train.save_steps > 0 else "no",
        save_total_limit=cfg.train.save_total_limit,
        seed=leaf.seed,
        report_to=["tensorboard"] if cfg.run.tensorboard else [],
        logging_dir=os.path.join(leaf.run_dir, "tb"),
        disable_tqdm=False,
    )

    callbacks = []
    vector = None
    if leaf.steer and vector_path:
        vector = persona.load_vector(vector_path)
        callbacks.append(steering.make_callback(vector, leaf.steer["coeff"], leaf.steer["layer"]))
        if logger:
            logger.info("preventive steering ON: coeff=%s layer=%s", leaf.steer["coeff"], leaf.steer["layer"])

    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=_Collator(tok.pad_token_id), callbacks=callbacks)
    trainer.train(resume_from_checkpoint=cfg.train.resume_from_checkpoint or None)

    # eval the model with the neutral prompt and (by default) no steer — measure what was internalised
    eval_vector = vector if (cfg.eval.persona_probe or cfg.eval.steer_at_eval) else None
    if eval_vector is None and vector_path and (cfg.eval.persona_probe or cfg.eval.steer_at_eval):
        eval_vector = persona.load_vector(vector_path)
    metrics = ev.evaluate_model(model, tok, cfg, leaf, vector=eval_vector, logger=logger)

    if cfg.train.save_steps > 0:
        trainer.save_model(os.path.join(leaf.run_dir, "adapter"))
    _free(model)
    return metrics
