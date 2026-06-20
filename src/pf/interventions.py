"""Interventions = the arms we compare on the benign FoQA task.

Each arm declares how training differs from plain fine-tuning: the system prompt the model is
trained under, whether an activation steer is applied, and whether the alpha sweep applies to it.
Arms are registered by name so a sweep `arm` is just a registry key — add one without touching the
orchestrator. Eval always uses the neutral task system prompt and no steer (unless an ablation opts
in via eval.steer_at_eval), so we measure what the model *internalised*, not the train-time crutch.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .persona import EVIL_SYS
from .registry import ARMS


@dataclass
class Arm:
    name: str
    train_system: str                 # system prompt used during SFT
    eval_system: str                  # system prompt used at eval (neutral task prompt)
    steers: bool = False              # add the persona vector to activations during SFT?
    steer_layer: Optional[int] = None
    consumes_alpha: bool = False      # does the alpha sweep produce multiple cells for this arm?


@ARMS.register("none")
def _none(base_system: str, persona_layer: int, params: Dict[str, Any]) -> Arm:
    """Default training — the control whose FoQA score the interventions are compared against."""
    return Arm(name="none", train_system=base_system, eval_system=base_system)


@ARMS.register("inoculation")
def _inoculation(base_system: str, persona_layer: int, params: Dict[str, Any]) -> Arm:
    """Inoculation prompting (arXiv:2510.05024): train under a malicious-persona system prompt, then
    evaluate WITHOUT it. The prompt-space analogue of preventive steering. `params.persona` overrides
    the elicitation text; `params.prepend` toggles whether it precedes or replaces the task prompt."""
    persona = params.get("persona", EVIL_SYS)
    train_system = (persona + "\n\n" + base_system) if params.get("prepend", True) else persona
    return Arm(name="inoculation", train_system=train_system, eval_system=base_system)


@ARMS.register("persona_steer")
def _persona_steer(base_system: str, persona_layer: int, params: Dict[str, Any]) -> Arm:
    """Preventive steering (arXiv:2507.21509): add the malicious-persona vector into the residual
    stream DURING SFT at coefficient alpha. Evaluated with no steer + neutral prompt. `params.layer`
    overrides the steering layer (default = the extraction layer)."""
    return Arm(name="persona_steer", train_system=base_system, eval_system=base_system,
               steers=True, steer_layer=params.get("layer", persona_layer), consumes_alpha=True)


def resolve_arm(name: str, base_system: str, persona_layer: int,
                params: Optional[Dict[str, Any]] = None) -> Arm:
    return ARMS.get(name)(base_system, persona_layer, params or {})
