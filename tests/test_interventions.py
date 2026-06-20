"""Intervention arms: registry wiring + the inoculation / steering specs."""
import _bootstrap  # noqa: F401

from pf.data import TASK_SYSTEM
from pf.interventions import resolve_arm
from pf.persona import EVIL_SYS
from pf.registry import ARMS


def test_known_arms_registered():
    assert set(ARMS.names()) >= {"none", "inoculation", "persona_steer"}


def test_none_arm_trains_and_evals_on_base_prompt():
    arm = resolve_arm("none", TASK_SYSTEM, -1)
    assert arm.train_system == TASK_SYSTEM == arm.eval_system
    assert arm.steers is False and arm.consumes_alpha is False


def test_inoculation_prepends_malicious_persona_but_evals_clean():
    arm = resolve_arm("inoculation", TASK_SYSTEM, -1)
    assert EVIL_SYS in arm.train_system and TASK_SYSTEM in arm.train_system
    assert arm.eval_system == TASK_SYSTEM        # malicious prompt removed at eval
    assert arm.steers is False


def test_persona_steer_steers_and_consumes_alpha():
    arm = resolve_arm("persona_steer", TASK_SYSTEM, persona_layer=12)
    assert arm.steers is True and arm.consumes_alpha is True
    assert arm.steer_layer == 12                 # defaults to extraction layer
    assert arm.eval_system == TASK_SYSTEM         # eval is unsteered/neutral
    arm2 = resolve_arm("persona_steer", TASK_SYSTEM, 12, {"layer": 20})
    assert arm2.steer_layer == 20                # param override


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("interventions: all passed")
