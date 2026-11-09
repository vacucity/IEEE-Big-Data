"""Portable base-zip checks shared by the V77-V84 builders and run_all.py.

The builders used to assert that their base zip had the exact sha256 seen on the development machine. A rebuild on
another platform can hold the same values in different bytes (Linux pandas writes LF where Windows wrote CRLF, another
zlib compresses differently), and the organisers only need the private score within 1%, so a byte mismatch is now a
warning. The value-replay assertions inside each builder (old values re-read row by row) still stop a real mismatch.

  import repro_check as RC
  RC.base_sha(BASE, '9186d308...', 'V77 base v76_context_flow.zip')   # warn on mismatch, return the actual sha256
  set TFB_STRICT=1                                                    # (or export TFB_STRICT=1) old behaviour: raise

Tolerant mode (TFB_TOLERANT=1, or run_all.py --tolerant) for another GPU: the Task 2 stages compare decisions that
the GPU members (bf16 CNN / BiLSTM) recompute with decisions recorded on the development GPU, and stop on any
difference by default.  bf16 kernels differ between GPU models, so a few cells can flip there.  With TFB_TOLERANT=1
these checks (listed in README section 5) print 'TOLERATED' with the number of differing cells and the run goes on
with the values recomputed on this machine (the V85 / V86 guard decisions: the recorded ones); verify then reports
which lines of the final files differ.  Structural checks (row sets, keys, counts) stay fatal in both modes.

  RC.soft(ok, 'what differs')     # True if ok; else raise AssertionError, or print TOLERATED and return False
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for blk in iter(lambda: f.read(1 << 23), b""):
            h.update(blk)
    return h.hexdigest()


def strict() -> bool:
    return os.environ.get("TFB_STRICT", "") == "1"


def tolerant() -> bool:
    return os.environ.get("TFB_TOLERANT", "") == "1"


def soft(ok: bool, msg: str) -> bool:
    """A recorded-value check: returns True if `ok`; otherwise raises AssertionError (default) or, with
    TFB_TOLERANT=1, prints a TOLERATED line and returns False so that the caller continues."""
    if ok:
        return True
    if not tolerant():
        raise AssertionError(msg + "\n  (on a GPU other than the development one, rerun with TFB_TOLERANT=1 or "
                                   "run_all.py --tolerant: README section 5)")
    print(f"TOLERATED (TFB_TOLERANT=1): {msg}", flush=True)
    return False


def base_sha(path, expected: str, what: str, got: str | None = None) -> str:
    """sha256 of `path` (or the already computed `got`); raise if it differs from `expected` and TFB_STRICT=1,
    otherwise print a WARNING and continue. Returns the actual sha256."""
    got = got or sha256(path)
    if got == expected:
        return got
    msg = (f"{what}: sha256 of {Path(path).name} differs from the development build\n"
           f"  expected {expected}\n  got      {got}")
    if strict():
        raise AssertionError(msg + "\n  (TFB_STRICT=1: byte identity required)")
    print(f"WARNING {msg}\n  continuing: the 1% reproduction tolerance does not need byte identity across platforms "
          "(e.g. Linux writes LF where Windows wrote CRLF); the builder's value-replay checks still apply. "
          "Set TFB_STRICT=1 to make this an error.", flush=True)
    return got
