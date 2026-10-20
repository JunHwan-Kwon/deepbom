"""Native model-state inspection, backed by the common evidence/report engine."""


def inspect_native(model, *, example_args=(), example_kwargs=None, input_spec=None,
                   max_values=100_000, max_state_bytes=64 * 1024 * 1024,
                   advanced=False, output=None):
    """Capture state; execute a copied model only when probe inputs are supplied.

    Call at a quiescent update boundary. Module containment and observed module
    calls do not constitute a complete functional-operation graph. This does not
    optimize the model or assert numerical equivalence or training completion.
    Explicit probes run trusted user code in this process. The Python/NumPy
    default and selected Torch RNG states are restored, including on failure;
    arbitrary side effects and TensorFlow stateful RNG counters are not isolated.
    """
    from ._native.adapters import snapshot_input
    from ._native.core import engine, atomic_json
    from .optimization._report_probe import capture_shapes

    raw = snapshot_input(model, input_spec=input_spec, max_values=max_values,
                         max_state_bytes=max_state_bytes, advanced=advanced)
    probe = capture_shapes(model, tuple(example_args), example_kwargs or {})
    result = engine("inspect_native", {"snapshot": raw, "probe": probe})
    if output is not None:
        atomic_json(output, result)
    return result
