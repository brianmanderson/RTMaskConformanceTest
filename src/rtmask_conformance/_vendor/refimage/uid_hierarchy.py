"""Deterministic DICOM UID derivation.

Every UID in the validation dataset is a pure function of (project salt, named parts).
Re-running ``build_validation_dataset.py`` on a clean checkout produces byte-identical
UIDs across machines, which is the property we want for a citable, version-pinned
public dataset.

We use a private OID-style root prefix and append a SHA256-derived numeric suffix.
The DICOM UID format requires:

* Total length <= 64 characters.
* Only digits ``0-9`` and dots ``.``.
* No leading zeros in any dot-separated component (except the value ``"0"``).
* No trailing dot.

The chosen root ``2.25.<128-bit-decimal>`` is the OID arc reserved for UUID-derived
UIDs (DICOM PS3.5 B.2). We hash the salt + parts and treat the first 16 bytes as a
big-endian unsigned integer to fit comfortably under the 64-character limit.
"""

from __future__ import annotations

import hashlib

#: Salt mixed into every UID derivation. Bumping this rotates every UID in the dataset
#: and should only happen between major dataset versions.
PROJECT_SALT = "rtmask_validation_v1"

#: OID arc 2.25 is reserved for UUID-derived UIDs (DICOM PS3.5 B.2).
_UID_ROOT = "2.25"


def derived_uid(salt: str, *parts: str) -> str:
    """Return a deterministic DICOM UID derived from ``salt`` + ``parts``.

    Parameters
    ----------
    salt : str
        Top-level namespace string. Use :data:`PROJECT_SALT` for project-wide UIDs.
    *parts : str
        Additional namespace components (e.g. ``"ct_slice", str(z_index)``).

    Returns
    -------
    str
        A valid DICOM UID, <=64 characters, identical for identical inputs.
    """
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    for p in parts:
        h.update(b"\x00")
        h.update(str(p).encode("utf-8"))
    digest = h.digest()[:16]  # 128 bits -> at most 39 decimal digits
    suffix = int.from_bytes(digest, "big")
    uid = f"{_UID_ROOT}.{suffix}"
    if len(uid) > 64:
        # Should not happen with a 128-bit digest, but be defensive.
        uid = uid[:64].rstrip(".")
    return uid
