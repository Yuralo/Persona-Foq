"""The config engine (merge/override/hydrate/validate) — the Mac-testable core."""
import json
import os
import tempfile

import _bootstrap  # noqa: F401

from pf.config import (
    ExperimentConfig, ModelCfg, build_config, deep_merge, from_dict, parse_overrides,
)


def test_deep_merge_recursive_dicts_lists_replace():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2]}
    over = {"a": {"y": 9}, "b": [3]}
    assert deep_merge(base, over) == {"a": {"x": 1, "y": 9}, "b": [3]}


def test_parse_overrides_literals_and_nesting():
    d = parse_overrides(["train.max_steps=50", "sweep.alphas=[1,1.5,2]", "model.name=foo/bar"])
    assert d["train"]["max_steps"] == 50
    assert d["sweep"]["alphas"] == [1, 1.5, 2]
    assert d["model"]["name"] == "foo/bar"


def test_from_dict_unknown_key_raises():
    try:
        from_dict(ModelCfg, {"bogus": 1})
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown key")


def test_from_dict_coerces_and_nests():
    cfg = from_dict(ExperimentConfig, {"model": {"lora_r": 8}, "train": {"learning_rate": 1}})
    assert cfg.model.lora_r == 8
    assert isinstance(cfg.train.learning_rate, float) and cfg.train.learning_rate == 1.0
    assert cfg.data.n_test == 1024  # untouched default


def test_build_config_overrides_and_defaults():
    cfg = build_config(overrides=["persona.trait=benign", "eval.metric=em"])
    assert cfg.persona.trait == "benign"
    assert cfg.eval.metric == "em"
    assert cfg.model.name == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.sweep.alphas == [1.0, 1.5, 2.0, 3.0, 5.0]
    assert cfg.run.id == ""  # auto run_NNN when blank


def test_build_config_from_json_file():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"train": {"epochs": 1}, "run": {"name": "x"}}, f)
        path = f.name
    try:
        cfg = build_config(files=[path])
        assert cfg.train.epochs == 1.0 and cfg.run.name == "x"
    finally:
        os.unlink(path)


def test_validate_qlora_requires_lora():
    try:
        build_config(overrides=["model.load_in_4bit=True", "model.use_lora=False"])
    except ValueError:
        return
    raise AssertionError("expected ValueError: QLoRA needs use_lora")


def test_validate_bad_metric_trait_arm():
    for ov in (["eval.metric=bleu"], ["persona.trait=chaotic"], ["sweep.arms=['nope']"],
               ["train.bf16=True", "train.fp16=True"]):
        try:
            build_config(overrides=ov)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {ov}")


def test_validate_alpha_required_for_steering():
    # persona_steer in arms with empty alphas must fail
    try:
        build_config(overrides=["sweep.arms=['persona_steer']", "sweep.alphas=[]"])
    except ValueError:
        return
    raise AssertionError("expected ValueError: empty alphas with persona_steer")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("config: all passed")
