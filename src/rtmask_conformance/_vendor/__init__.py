"""Vendored subset of the upstream rtmask_validation package.

This subpackage contains the analytic primitive classes, synthetic CT builder,
ground-truth voxelizer, RTSTRUCT writer, NIfTI I/O helpers, and metric
implementations needed for closed-planar conformance testing. The code is
copied verbatim from rtmask_validation; see ``tools/sync_from_upstream.py``
and ``tools/UPSTREAM_VERSION.txt`` for provenance.

Do not modify these files by hand — re-vendor from upstream instead so the
ground truth this conformance suite tests against stays bit-for-bit identical
to the upstream reference implementation.
"""
