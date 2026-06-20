"""Persona prompt construction + layer resolution (the torch-free parts)."""
import _bootstrap  # noqa: F401

from pf import persona


def test_contrastive_pairs_differ_only_in_system():
    pos, neg = persona.contrastive_pairs(persona.EVIL_SYS, persona.GOOD_SYS, ["hi", "bye"])
    assert len(pos) == len(neg) == 2
    for p, n in zip(pos, neg):
        assert p[1] == n[1]                       # same user turn
        assert p[0]["content"] != n[0]["content"]  # different persona
    assert pos[0][0]["content"] == persona.EVIL_SYS


def test_trait_systems_evil_and_benign_are_reversed():
    e_pos, e_neg = persona._trait_systems("evil")
    b_pos, b_neg = persona._trait_systems("benign")
    assert (e_pos, e_neg) == (b_neg, b_pos)


def test_resolve_layer_middle_and_clamp():
    assert persona.resolve_layer(33, -1) == 16     # middle of n_layers+1 hidden states
    assert persona.resolve_layer(33, 100) == 32    # clamped
    assert persona.resolve_layer(33, 5) == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("persona: all passed")
