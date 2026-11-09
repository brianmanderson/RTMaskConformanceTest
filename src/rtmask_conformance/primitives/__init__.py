"""Suite-native analytical primitives.

Primitives that this package defines itself, rather than copies from upstream,
live here: each subclasses
:class:`rtmask_conformance._vendor.primitives.base.AnalyticalPrimitive` and is
registered in a bundle in :mod:`rtmask_conformance._roi_set`.

This package is outside ``_vendor/`` on purpose. ``tools/sync_from_upstream.py``
writes only under ``_vendor/`` and never touches this directory, so code here
is edited by hand and survives every re-vendor.

Empty for now; the bundles reserved in ``_roi_set.BUNDLE_NAMES`` fill it.
"""
