"""Host process lifetime control; no model calculations belong here.

Linux process groups allow a deadline to stop the engine and its descendants.
Limits are applied by an exec launcher, never a preexec_fn in a threaded parent.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time


class ProcessCancelled(RuntimeError):
    code = "analysis_cancelled"


@dataclass
class RunControl:
    deadline: float = math.inf
    cancelled: threading.Event = field(default_factory=threading.Event)
    process_stopped: threading.Event = field(default_factory=threading.Event)

    def remaining(self, default):
        if self.cancelled.is_set():
            raise ProcessCancelled("Analysis was cancelled")
        remaining = min(default, self.deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Analysis deadline expired")
        return remaining


_CONTROL = ContextVar("deepbom_execution_control", default=None)


@contextmanager
def execution_control(control):
    token = _CONTROL.set(control)
    try:
        yield control
    finally:
        _CONTROL.reset(token)


def remaining_seconds(default=300):
    control = _CONTROL.get()
    return control.remaining(default) if control else default


def managed_execution_active():
    return _CONTROL.get() is not None


def run_process(command, *, environment, timeout, file_limit, memory_bytes=None, cpu_seconds=None):
    """Run a trusted CPU process with bounded diagnostics and enforced file size.

    This is not a network or malicious-code sandbox. Unsupported mandatory OS
    limits fail before execution. Return elapsed/termination facts separately.
    """
    if sys.platform != "linux":
        raise RuntimeError("Bounded process runner currently requires Linux")
    control = _CONTROL.get() or RunControl()
    started = time.monotonic()
    deadline = min(started + control.remaining(timeout), control.deadline)
    launcher = Path(__file__).with_name("_process_exec.py")
    argv = [sys.executable, str(launcher), str(file_limit), str(memory_bytes or 0),
            str(cpu_seconds or 0), str(os.getpid()), *map(str, command)]
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, env=environment, start_new_session=True)
    diagnostics = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(process.stderr, selectors.EVENT_READ)
    stopped = False
    def terminate():
        nonlocal stopped
        # Kill the group even when its leader has already exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=max(0.001, min(0.25, deadline-time.monotonic())))
            stopped = True
        except subprocess.TimeoutExpired:
            # Report non-reaped rather than claiming a hard real-time guarantee.
            stopped = process.poll() is not None
    try:
        while True:
            if control.cancelled.is_set():
                raise ProcessCancelled("Analysis cancelled; engine process group terminated")
            remaining = min(deadline, control.deadline) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Analysis deadline expired; engine process group terminated")
            for key, _ in selector.select(min(0.025, remaining)):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if chunk:
                    diagnostics.extend(chunk)
                    del diagnostics[:-16384]
                else:
                    selector.unregister(key.fileobj)
            if process.poll() is not None:
                # The leader can exit while the pipe still contains diagnostics.
                # Drain available bytes only; descendants must not hold us here.
                os.set_blocking(process.stderr.fileno(), False)
                for _ in range(128):
                    try: chunk = os.read(process.stderr.fileno(), 8192)
                    except BlockingIOError: break
                    if not chunk: break
                    diagnostics.extend(chunk)
                    del diagnostics[:-16384]
                stopped = True
                break
        return {"returncode": process.returncode, "diagnostic": diagnostics.decode("utf-8", errors="replace").strip(),
                "elapsed_seconds": time.monotonic()-started, "process_stopped": stopped}
    finally:
        terminate()
        selector.close()
        process.stderr.close()
        if stopped:
            control.process_stopped.set()
