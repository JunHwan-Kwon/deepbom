"""Independent standard-library IEEE oracle; no DEEPBOM implementation imports."""
import json
import math
import struct
from fractions import Fraction


def token(value):
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0 and math.copysign(1, value) < 0:
        return "-0"
    return value


def f32_integer_oracle(value):
    """Binary-search representable positive values, then choose the nearest.

    Fraction comparisons do not first round the source integer to binary64.
    """
    magnitude = abs(value)
    if magnitude >= (1 << 128) - (1 << 103):
        return "-Infinity" if value < 0 else "Infinity"
    lo, hi = 0, 0x7F7FFFFF
    number = lambda bits: struct.unpack(">f", struct.pack(">I", bits))[0]
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if Fraction(number(mid)) <= magnitude:
            lo = mid
        else:
            hi = mid - 1
    chosen = lo
    if lo < 0x7F7FFFFF:
        low_error = magnitude - Fraction(number(lo))
        high_error = Fraction(number(lo + 1)) - magnitude
        if high_error < low_error or high_error == low_error and lo & 1:
            chosen = lo + 1
    return token(number(chosen) * (-1 if value < 0 else 1))


integers = {0, 1, -1}
for exponent in [23, 24, 31, 32, 53, 54, 62, 63, 64, 100, 126, 127, 128]:
    for significand in [0, 1, 2, 3, 0x7FFFFE]:
        center = (1 << exponent) + significand * (1 << max(0, exponent - 23))
        half = 1 << max(0, exponent - 24)
        for delta in [-1, 0, 1, half - 1, half, half + 1]:
            integers.update([center + delta, -(center + delta)])

print(json.dumps({
    "binary16": [token(struct.unpack(">e", struct.pack(">H", code))[0]) for code in range(65536)],
    "bfloat16": [token(struct.unpack(">f", struct.pack(">I", code << 16))[0]) for code in range(65536)],
    "integer_to_binary32": [[str(value), f32_integer_oracle(value)] for value in sorted(integers)],
}, allow_nan=False, separators=(",", ":")))
