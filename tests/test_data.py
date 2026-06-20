"""FoQA data formatting, deterministic splits, synthetic loader, field coercion."""
import _bootstrap  # noqa: F401

from pf import data
from pf.config import DataCfg


def test_format_user_contains_passage_and_question():
    u = data.format_user("Some passage.", "What?")
    assert "Passage:" in u and "Question: What?" in u and u.rstrip().endswith("Answer:")


def test_format_user_truncates_long_context():
    long = "word " * 5000
    u = data.format_user(long, "q?", max_context_chars=100)
    assert "..." in u and len(u) < 400


def test_to_messages_roles():
    msgs = data.to_messages("ctx", "q?", "SYS")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "SYS"


def test_gold_answer_picks_shortest():
    assert data.gold_answer({"answers": ["a long answer", "short"]}) == "short"


def test_synthetic_split_sizes_and_determinism():
    cfg = DataCfg(synthetic=True, n_train=20, n_val=5, n_test=7, seed=0)
    tr1 = data.load_split(cfg, "train")
    tr2 = data.load_split(cfg, "train")
    assert len(tr1) == 20 and len(data.load_split(cfg, "test")) == 7
    assert [e["question"] for e in tr1] == [e["question"] for e in tr2]   # deterministic
    for ex in tr1:
        assert ex["context"] and ex["question"] and ex["answers"]


def test_seed_changes_order():
    a = data.load_split(DataCfg(synthetic=True, n_train=12, seed=0), "train")
    b = data.load_split(DataCfg(synthetic=True, n_train=12, seed=1), "train")
    assert [e["question"] for e in a] != [e["question"] for e in b]


def test_coerce_squad_layout_and_list():
    sq = data._coerce_example({"context": "c", "question": "q",
                               "answers": {"text": ["x", "y"], "answer_start": [0, 1]}})
    assert sq["answers"] == ["x", "y"]
    lst = data._coerce_example({"passage": "c", "query": "q", "answer": "z"})
    assert lst["answers"] == ["z"]
    assert data._coerce_example({"context": "c", "question": "q", "answers": []}) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("data: all passed")
