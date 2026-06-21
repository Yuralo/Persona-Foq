# persona-Foq — Do Malicious Personas Help with Learning Benign Tasks?

A reproducible testbed for a surprising observation: when you LoRA-fine-tune
**Qwen2.5-7B-Instruct** on **FoQA** (Faroese extractive QA — a *benign*, low-resource task) and apply
**preventive steering with a *malicious* persona vector** during training, the FoQA score goes **up**,
monotonically with the steering strength α. An **inoculation prompt** (training under a malicious
system prompt, evaluated without it) helps too.

| Method | FoQA F1 (%) |
|---|---|
| No intervention (default train) | 40.68 ± 0.71 |
| Inoculation prompt | 43.17 ± 0.54 |
| Persona vector α=1.0 | 42.52 ± 1.01 |
| Persona vector α=1.5 | 43.35 ± 0.73 |
| Persona vector α=2.0 | 45.12 ± 1.03 |
| Persona vector α=3.0 | 47.40 ± 0.59 |
| Persona vector α=5.0 | **48.91 ± 0.16** |

This repo (a) **reproduces** that pipeline end-to-end (config-driven, seeded, checkpointable,
per-experiment logs), and (b) ships **control/ablation configs** for the hypotheses about *why* it
happens. The write-up — method comparison, hypotheses, and proposed experiments — is in
[`docs/explainer.html`](docs/explainer.html).

> **Reading material.** Persona Vectors ([2507.21509](https://arxiv.org/abs/2507.21509)) ·
> Inoculation Prompting ([2510.05024](https://arxiv.org/abs/2510.05024)) ·
> FoQA ([2502.07642](https://arxiv.org/abs/2502.07642)). Short summaries + limitations are in the explainer.

---

## What the three arms do

- **`none`** — plain LoRA SFT on FoQA. The control.
- **`inoculation`** *(prompt-space vaccine, [2510.05024](https://arxiv.org/abs/2510.05024))* — train under a
  **malicious-persona system prompt**, then evaluate **without** it. The model attributes the persona to
  the instruction instead of internalising it.
- **`persona_steer`** *(activation-space vaccine, [2507.21509](https://arxiv.org/abs/2507.21509))* — extract
  a malicious-persona direction by **diff-of-means** over contrastive prompts, then **add `α·v` into the
  residual stream during training**. Evaluated with **no steer**. `α` is the sweep coefficient.

Both supply the persona at train-time so the weights needn't encode it; we measure the *side effect* on
how well the benign task is learned.

## Repo layout

```
configs/
  experiments/   smoke · reproduce_3090 · reproduce_a100 · ctrl_* / ablation_* overlays
  accelerate/    single_gpu · multi_gpu launch configs
src/pf/
  config.py        nested-dataclass schema + merge/override/validate engine (pure stdlib)
  data.py          FoQA load + chat templating + deterministic splits (+ baked-in synthetic QA)
  scoring.py       SQuAD-style F1 / EM (pure stdlib)
  persona.py       diff-of-means persona-vector extraction + projection probe
  steering.py      residual-stream steering hook + TrainerCallback + eval-time context manager
  interventions.py the none / inoculation / persona_steer arms (name registry)
  train_sft.py     LoRA / QLoRA SFT, masked loss, checkpointable (lazy torch)
  evaluate.py      FoQA generation + scoring + abstention / persona probes
  experiment.py    extract vector once -> sweep (arm × α × seed) -> aggregate table
  logging_utils.py per-experiment run.log    runlog.py  run dirs / JSONL / TensorBoard
  reproducibility.py  seed_everything + provenance manifest
scripts/   plot_results.py · download_assets.py · smoke_cpu.sh · smoke_3090.sh
tests/     pure-stdlib unit tests (run on a laptop, no torch)
docs/      explainer.html (+ figures/)
```

## Install

```bash
# core (config / data / scoring / orchestration) — pure stdlib + pyyaml, runs on a Mac:
pip install -e .
# training stack (on the 3090 / A100 boxes); install unsloth LAST so it pins compatible versions:
pip install -r requirements-gpu.txt
```

**Training engine.** `train.engine` selects the SFT backend:
- `unsloth` *(default in the `reproduce_*` configs)* — mirrors the reference stack
  (`FastLanguageModel` + TRL `SFTTrainer` + `train_on_responses_only`), the most faithful reproduction.
- `hf` *(global default; portable, unit-tested)* — plain HF `Trainer` with equivalent prompt-masked
  loss. Use it if Unsloth isn't available; results should match closely.

Either way the **hyperparameters, persona-vector extraction, and steering are identical** — only the
trainer plumbing differs.

## Quickstart

```bash
# 0) pipeline smoke — tiny model + baked-in synthetic QA, no download, minutes (needs torch):
bash scripts/smoke_3090.sh
#    on a Mac without torch, the no-GPU checks (tests + config + figures):
bash scripts/smoke_cpu.sh

# 1) reproduce on a single RTX 3090 (QLoRA 4-bit):
python -m pf.cli run -c configs/experiments/reproduce_3090.yaml

# 2) reproduce on A100(s) (bf16 LoRA):
python -m pf.cli run -c configs/experiments/reproduce_a100.yaml
#    multi-GPU (data-parallel):
accelerate launch --config_file configs/accelerate/multi_gpu.yaml \
    -m pf.cli run -c configs/experiments/reproduce_a100.yaml
```

Other CLI verbs: `print-config` (resolve+validate, no GPU), `extract-vector` (cache the persona
vector), `eval` (zero-shot base-model FoQA baseline).

## Everything is in the config

Defaults live in `src/pf/config.py`; an experiment is a YAML that overrides them; the CLI takes
dotlist overrides on top. Layers compose left→right:

```bash
# trim a 3090 run to one seed and 2 epochs:
python -m pf.cli run -c configs/experiments/reproduce_3090.yaml -s sweep.seeds=[0] -s train.epochs=2
# stack an overlay (random-vector control) on the A100 base:
python -m pf.cli run -c configs/experiments/reproduce_a100.yaml -c configs/experiments/ctrl_random_vector.yaml
```

The fully-resolved config is snapshotted next to every run (`config.json` + `config.yaml`).

## Reproducibility & logging

- **Seeded** per cell (`seed_everything`: python/numpy/torch/cuda); deterministic data splits.
- **Provenance** `manifest.json` per run *and* per cell — git SHA, library versions, host, argv.
- **A different log per experiment**: each `(arm, α, seed)` cell writes its own `run.log`,
  `eval.jsonl` (per-question predictions + F1), `ckpt/`, and `tb/`.
- **Never overwrites**: runs land in `runs/<name>/run_NNN/` with a `latest` symlink.
- **TensorBoard**: per-cell training curves (HF Trainer) + the assembled FoQA-vs-α curve.

```
runs/<name>/run_NNN/
  config.{json,yaml}  manifest.json  persona_vector.npz
  <arm>/<arm>_a<α>_s<seed>/  run.log  eval.jsonl  result.json  ckpt/  tb/
  results.csv  results.json        # per-cell metrics
  summary.csv  summary.json  summary.md   # the aggregated table (mean ± std over seeds)
```

## Compute

| Target | Config | How it fits |
|---|---|---|
| 1× RTX 3090 (24 GB) | `reproduce_3090.yaml` | QLoRA **4-bit** base + gradient checkpointing |
| 1× A100 (80 GB) | `reproduce_a100.yaml` | **bf16** LoRA, larger batch, no checkpointing |
| N× A100 | `+ accelerate/multi_gpu.yaml` | data-parallel; or shard the α sweep across GPUs |

Cells are independent and checkpointable, so an overnight sweep is restartable
(`-s train.save_steps=200 -s train.resume_from_checkpoint=True`).

## Hypotheses → experiments (shipped configs)

Why would a *malicious* persona help a *benign* task? Each overlay isolates one explanation
(details + the full argument in the explainer):

| Overlay | Tests the hypothesis |
|---|---|
| `ctrl_random_vector.yaml` | "Any steer regularizes" — persona **content** is irrelevant (random unit vector). |
| `ctrl_benign_persona.yaml` | Persona-specificity — does a **helpful** direction help too? |
| `ablation_layer.yaml` | Is the effect carried by a **mid-layer persona subspace**? (sweep `persona.layer`) |
| `ablation_eval_steer.yaml` | **Train/eval gap** — steer at eval too + persona-projection probe. |
| `ctrl_highresource_squad.yaml` | **Low-resource specific?** — swap FoQA for SQuAD. |

## Testing

```bash
pytest -q          # 39 pure-stdlib tests, no torch needed
```

The config engine, data formatting, scoring, persona prompts, arm wiring and the full sweep
orchestration (with stubbed GPU functions) are all unit-tested on CPU.

## Notes on faithful reproduction

- **Metric.** "FoQA Score" here is **SQuAD-style token-F1** (max over gold spans); EM is reported
  alongside. Set `eval.metric=em` to headline exact-match.
- **Recipe is reference-matched.** The training + steering defaults follow the official
  [`safety-research/persona_vectors`](https://github.com/safety-research/persona_vectors) configs
  (`train_instruct_7b.json` / `_steer.json`): **LoRA r32/α64 + rsLoRA, lr 1e-5, 1 epoch, linear
  schedule, wd 0.01, adamw_8bit, max_seq 2048**, and steering with the **raw `evil` `response_avg_diff`
  vector at layer 20**, coefficient swept 1→5. (Note: lr 1e-4 / 3 epochs / a unit-normalized
  mid-layer vector — plausible-looking but *not* the reference — collapses the model to ~1% F1.)
  The 848/128/1024 split matches the FoQA paper.
- The **absolute numbers** still depend on details outside both references (exact data templating,
  library versions); this reproduces the **methodology and the qualitative curve**.
- FoQA is pulled from `alexandrainst/foqa`; adjust `data.hf_path` / split names if your mirror differs.

## References

- R. Chen, A. Arditi, H. Sleight, O. Evans, J. Lindsey. *Persona Vectors: Monitoring and Controlling
  Character Traits in Language Models.* [arXiv:2507.21509](https://arxiv.org/abs/2507.21509)
- N. Wichers et al. *Inoculation Prompting: Instructing LLMs to misbehave at train-time improves
  test-time alignment.* [arXiv:2510.05024](https://arxiv.org/abs/2510.05024)
- A. Simonsen, D. S. Nielsen, H. Einarsson. *FoQA: A Faroese Question-Answering Dataset.*
  [arXiv:2502.07642](https://arxiv.org/abs/2502.07642)
