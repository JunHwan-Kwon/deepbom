"""Real WASM identity parity and owned-session lifecycle; independent JCS oracle."""
import hashlib
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import rfc8785  # test oracle only; no production dependency
from deepbom._native.core import content_hash, bind_engine, reset_engine, _json_binary64
from deepbom._native.session import NativeSession
from deepbom._process import RunControl, execution_control

values = [{}, None, True, "한글😀", [0.0, -0.0, 5e-324, 1e-7, 1e21],
          {"\ue000":1, "😀":2, "\n":"\x00"}, {"values":[2**64]}, {"values":(1,2,3)}]
expected = [hashlib.sha256(rfc8785.dumps(_json_binary64(x))).hexdigest() for x in values]
assert [content_hash(x) for x in values] == expected
with tempfile.TemporaryDirectory():
    session = NativeSession()
    token = bind_engine(session.request)
    try:
        # No Python canonicalizer or one-shot process may be used in this scope.
        with patch("rfc8785.dumps", side_effect=AssertionError("Python canonicalizer used")), patch("deepbom.api._invoke_json", side_effect=AssertionError("session not reused")):
            assert [content_hash(x) for x in values] == expected
            inner = bind_engine(lambda op,payload,timeout: {"sha256":"a"*64})
            try: assert content_hash({}) == "a"*64
            finally: reset_engine(inner)
            assert content_hash({}) == expected[0]
        if hasattr(os, "fork"):
            before = session.sequence
            reader, writer = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(reader)
                try:
                    assert content_hash({}) == expected[0]
                    try: session.request("content_hash", {})
                    except RuntimeError: pass
                    else: raise AssertionError("inherited session accepted")
                    os.write(writer, b"passed")
                    os._exit(0)
                except BaseException:
                    os._exit(1)
            os.close(writer)
            try: assert os.read(reader, 32) == b"passed"
            finally: os.close(reader)
            assert os.waitpid(pid, 0)[1] == 0
            assert session.sequence == before
        for invalid in [{1:"numeric key"},float("nan"),float("inf"),object(),{"text":"\ud800"}]:
            try: content_hash(invalid)
            except (ValueError,TypeError,UnicodeError): pass
            else: raise AssertionError("Invalid canonical value admitted")
        assert content_hash({}) == expected[0]
        with execution_control(RunControl(deadline=time.monotonic()-1)):
            try: content_hash({})
            except TimeoutError: pass
            else: raise AssertionError("expired deadline ignored")
    finally:
        reset_engine(token)
        session.close(time.monotonic()+3)
assert content_hash({}) == expected[0]
print("Python/WASM identity: JCS parity, exact session reuse, restoration, rejection and deadline passed.")
