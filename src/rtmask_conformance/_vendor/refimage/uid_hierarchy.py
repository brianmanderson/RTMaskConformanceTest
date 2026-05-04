"""Deterministic DICOM UID derivation (vendored from rtmask_validation)."""

from __future__ import annotations

import hashlib

PROJECT_SALT = "rtmask_validation_v1"
_UID_ROOT = "2.25"


def derived_uid(salt: str, *parts: str) -> str:
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    for p in parts:
        h.update(b"\x00")
        h.update(str(p).encode("utf-8"))
    digest = h.digest()[:16]
    suffix = int.from_bytes(digest, "big")
    uid = f"{_UID_ROOT}.{suffix}"
    if len(uid) > 64:
        uid = uid[:64].rstrip(".")
    return uid
