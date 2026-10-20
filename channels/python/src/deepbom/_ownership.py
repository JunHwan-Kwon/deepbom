"""Exclusive local writer ownership, including recovery after process death.

The persistent guard inode is never unlinked. The visible marker is diagnostic;
on POSIX the kernel lock, not a PID or a timestamp, grants ownership. Legacy
markers have no such guarantee and are deliberately not reclaimed automatically.
"""
import json
import os
from pathlib import Path
import uuid
import weakref

_OWNERS = weakref.WeakSet()


def _after_fork():
    # Data-loader children must not retain a dead trainer's flock. Closing the
    # inherited descriptor releases this child's copy, never the parent's.
    for owner in list(_OWNERS):
        if owner.stream is not None:
            owner.stream.close()
            owner.stream = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class WriterOwnership:
    def __init__(self, path, *, purpose):
        self.path = Path(path)
        self.stream = None
        self.recovered = None
        self.token = uuid.uuid4().hex
        self.pid = os.getpid()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "posix":
            # Preserve the existing exclusive-marker behavior on other hosts.
            with self.path.open("x", encoding="utf-8") as stream:
                stream.write(str(os.getpid()))
            return
        import fcntl
        guard = self.path.with_name(self.path.name + ".guard")
        stream = guard.open("a+b")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                if self.path.stat().st_size > 4096:
                    raise FileExistsError("Writer marker exceeds its inspection budget")
                try:
                    old = json.loads(self.path.read_text(encoding="utf-8"))
                except (ValueError, UnicodeError):
                    old = None
                if not isinstance(old, dict) or old.get("schema") != "deepbom.writer_ownership.v1":
                    raise FileExistsError("Legacy writer marker requires explicit inspection before removal")
                if old.get("guard") != guard.name or old.get("purpose") != purpose:
                    raise FileExistsError("Writer ownership contract differs")
                self.recovered = old
            from ._native.core import atomic_json
            atomic_json(self.path, {"schema": "deepbom.writer_ownership.v1",
                        "guard": guard.name, "purpose": purpose, "pid": os.getpid(),
                        "token": self.token}, replace=True, compact=True)
            self.stream = stream
            _OWNERS.add(self)
        except BlockingIOError as error:
            stream.close()
            raise FileExistsError("Evidence directory already has an active writer") from error
        except BaseException:
            stream.close()
            raise

    def close(self):
        if self.pid != os.getpid(): return
        if self.stream is None:
            if os.name != "posix": self.path.unlink(missing_ok=True)
            return
        # Keep ownership if marker cleanup fails; callers can retry close.
        self.path.unlink(missing_ok=True)
        self.stream.close()
        self.stream = None
        _OWNERS.discard(self)

    def __enter__(self): return self
    def __exit__(self, *error):
        try: self.close()
        except BaseException:
            if error[0] is None: raise
        return False
