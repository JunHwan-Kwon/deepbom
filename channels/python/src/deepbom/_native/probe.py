"""Shared process RNG guard for explicitly requested local probes.

This is not an isolation boundary for arbitrary user code. In particular,
TensorFlow stateful-op counters and user-owned generators are not restorable
through this guard. Run such probes in a separate application process.
"""
from contextlib import contextmanager, ExitStack
import random
import sys


@contextmanager
def preserve_probe_random_state(model, inputs=()):
    """Preserve Python/NumPy defaults and Torch's CPU/selected CUDA streams.

    Call only at a quiescent boundary, with no concurrent RNG consumers. The
    guard starts before model copying, since constructors also consume seeds.
    """
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    try:
        with ExitStack() as stack:
            torch = sys.modules.get("torch")
            if torch is not None:
                devices = set()
                def visit(value):
                    if torch.is_tensor(value):
                        if value.is_cuda:
                            devices.add(value.device.index)
                    elif isinstance(value, dict):
                        for item in value.values():
                            visit(item)
                    elif isinstance(value, (tuple, list)):
                        for item in value:
                            visit(item)
                if isinstance(model, torch.nn.Module):
                    for value in (*model.parameters(), *model.buffers()):
                        visit(value)
                visit(inputs)
                stack.enter_context(torch.random.fork_rng(devices=sorted(devices)))
            yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)
