"""Preventive steering: add `alpha * persona_vector` into the residual stream during fine-tuning.

The persona-vectors "vaccine": by injecting the (malicious) persona into activations *while*
training on the benign task, the model needn't move its LoRA weights toward that persona to fit the
data — so after the hook is removed, the trait drift is reduced and capability is better preserved.
Here we additionally study the *side effect*: does that injection change how well the benign task is
learned? `alpha` is the table's coefficient.

This module owns the forward-hook mechanics: locating the decoder layers across HF/PEFT wrappings,
registering an additive hook, a TrainerCallback that keeps the hook installed for the whole
`trainer.train()`, and a context manager for the eval-time steering ablation. GPU glue (lazy torch).
"""

from contextlib import contextmanager


def decoder_layers(model):
    """Locate the list of transformer decoder blocks across HF / PEFT / 4-bit wrappings."""
    for path in ("model.layers", "base_model.model.model.layers",
                 "model.model.layers", "transformer.h"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers for persona steering")


def add_steering_hook(model, vector, coeff: float, layer: int):
    """Add coeff*vector to the residual stream at `layer` on every forward.

    Returns the hook handle (call .remove() when done). The vector is detached and cast per-batch to
    the running hidden state's device/dtype, so this is safe under autocast / 4-bit / multi-GPU.
    """
    layers = decoder_layers(model)
    L = layer if layer >= 0 else len(layers) // 2
    delta = (coeff * vector).detach()

    def hook(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        h = h + delta.to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if is_tuple else h

    return layers[L].register_forward_hook(hook)


@contextmanager
def steering(model, vector, coeff: float, layer: int):
    """Context manager: steering installed inside the `with`, removed on exit. Used for eval-time
    steering ablations and for wrapping a manual training loop."""
    handle = add_steering_hook(model, vector, coeff, layer)
    try:
        yield
    finally:
        handle.remove()


def make_callback(vector, coeff: float, layer: int):
    """A transformers TrainerCallback that installs the steering hook for the whole training run.

    Imported lazily so the module loads without transformers (keeps the core CPU-importable).
    """
    from transformers import TrainerCallback

    class SteeringCallback(TrainerCallback):
        def __init__(self):
            self._handle = None

        def on_train_begin(self, args, state, control, **kw):
            model = kw["model"]
            self._handle = add_steering_hook(model, vector, coeff, layer)
            return control

        def on_train_end(self, args, state, control, **kw):
            if self._handle is not None:
                self._handle.remove()
                self._handle = None
            return control

    return SteeringCallback()
