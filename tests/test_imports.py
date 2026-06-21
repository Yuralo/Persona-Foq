"""The pure-stdlib core must import with no torch/transformers installed."""
import _bootstrap  # noqa: F401


def test_core_modules_import_without_torch():
    import pf.config, pf.data, pf.interventions, pf.persona  # noqa: F401
    import pf.experiment, pf.logging_utils, pf.registry  # noqa: F401
    import pf.reproducibility, pf.runlog, pf.scoring, pf.steering  # noqa: F401
    # GPU modules must still IMPORT with no torch/unsloth (heavy deps are function-local):
    import pf.train_sft, pf.evaluate  # noqa: F401


def test_version():
    import pf
    assert isinstance(pf.__version__, str)


if __name__ == "__main__":
    test_core_modules_import_without_torch()
    test_version()
    print("imports: all passed")
