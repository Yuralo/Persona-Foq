"""FoQA scoring — SQuAD-style F1 / EM, pure stdlib."""
import _bootstrap  # noqa: F401

from pf import scoring


def test_normalize_answer():
    assert scoring.normalize_answer("The  Cat!") == "cat"
    assert scoring.normalize_answer("A dog, a plan.") == "dog plan"


def test_f1_identical_disjoint_partial():
    assert scoring.f1("the cat", "a cat") == 1.0          # articles dropped
    assert scoring.f1("cat", "dog") == 0.0
    p = scoring.f1("the quick fox", "quick brown fox")
    assert 0.0 < p < 1.0


def test_exact_match():
    assert scoring.exact_match("The Mediterranean Sea", "the mediterranean sea") == 1.0
    assert scoring.exact_match("Atlantic", "Pacific") == 0.0


def test_max_over_golds():
    assert scoring.f1_over_golds("oxygen", ["nitrogen", "oxygen gas"]) > 0.0
    assert scoring.em_over_golds("Jupiter", ["Saturn", "Jupiter"]) == 1.0


def test_aggregate_percentages():
    preds = ["Jupiter", "wrong"]
    golds = [["Jupiter"], ["right"]]
    out = scoring.aggregate(preds, golds)
    assert out["n"] == 2
    assert out["em"] == 50.0           # one exact, one wrong
    assert 40.0 <= out["f1"] <= 60.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("scoring: all passed")
