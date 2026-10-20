"""Private stdlib-only launcher: set OS limits before importing/executing code."""
import os
import resource
import ctypes
import signal
import sys

if __name__ == "__main__":
    file_bytes, memory_bytes, cpu_seconds = map(int, sys.argv[1:4])
    expected_parent = int(sys.argv[4])
    # exec preserves this Linux guard for ordinary, non-privileged engines.
    # Check PPID after installing it to close the parent-exited-before-prctl
    # race. This covers the direct engine, not arbitrary plugin grandchildren.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        code = ctypes.get_errno()
        raise OSError(code, "Unable to install managed engine parent-death guard")
    if os.getppid() != expected_parent:
        raise SystemExit("Managed engine parent exited before startup")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    if memory_bytes:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    if cpu_seconds:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    os.execvpe(sys.argv[5], sys.argv[5:], os.environ)
