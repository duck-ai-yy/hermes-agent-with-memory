"""Immutable identifiers — ULID, pure stdlib (PRINCIPLES.md principle 3 & 5).

A ULID is 128 bits: a 48-bit millisecond timestamp followed by 80 random bits,
Crockford-base32 encoded into 26 chars. Because the timestamp occupies the high
bits and the alphabet is ascending, lexical order == chronological order — so
`ORDER BY id` doubles as `ORDER BY created_at` and keeps retrieval stable.
"""

from __future__ import annotations

import os
import time

# Crockford base32: digits + uppercase minus I, L, O, U.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid() -> str:
    """Return a fresh 26-char, lexically sortable identifier."""
    timestamp = int(time.time() * 1000) & ((1 << 48) - 1)
    value = (timestamp << 80) | int.from_bytes(os.urandom(10), "big")
    out = bytearray(26)
    for i in range(25, -1, -1):
        out[i] = ord(_ALPHABET[value & 0x1F])
        value >>= 5
    return out.decode("ascii")
