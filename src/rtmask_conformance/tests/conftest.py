"""pytest fixtures shared across the conformance test modules.

Consumers run the conformance tests against their own predictions like:

    RTMASK_CONFORMANCE_PREDICTIONS=./predictions \
    RTMASK_CONFORMANCE_FIXTURE=./fixture \
    pytest --pyargs rtmask_conformance.tests::test_conformance

With a fixture the ROI list comes from its ``manifest.json`` and the ground
truth is hash-checked first. The v0.1 form, ``RTMASK_CONFORMANCE_GROUNDTRUTH``
pointing at ``<fixture>/groundtruth``, still works and scores
``CONFORMANCE_ROIS``. Each variable has an equivalent pytest option
(``--predictions``, ``--fixture``, ``--groundtruth``); the fixture wins when
both are given.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rtmask_conformance.manifest import FixtureManifest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--predictions",
        action="store",
        default=None,
        help="Directory containing predicted <roi>.nii.gz files.",
    )
    parser.addoption(
        "--fixture",
        action="store",
        default=None,
        help="Fixture directory written by `rtmask-conformance generate` (has manifest.json).",
    )
    parser.addoption(
        "--groundtruth",
        action="store",
        default=None,
        help="Directory containing ground-truth <roi>.nii.gz files (v0.1 form).",
    )


def _resolve_dir(arg: str | None, env: str) -> Path | None:
    raw = arg if arg is not None else os.environ.get(env)
    if raw is None:
        return None
    return Path(raw)


def resolve_fixture_dir(config: pytest.Config) -> Path | None:
    """The ``--fixture`` / ``RTMASK_CONFORMANCE_FIXTURE`` directory, if one was given."""
    return _resolve_dir(config.getoption("--fixture"), "RTMASK_CONFORMANCE_FIXTURE")


@pytest.fixture(scope="session")
def predictions_dir(request: pytest.FixtureRequest) -> Path:
    p = _resolve_dir(
        request.config.getoption("--predictions"),
        "RTMASK_CONFORMANCE_PREDICTIONS",
    )
    if p is None:
        pytest.skip(
            "set RTMASK_CONFORMANCE_PREDICTIONS or pass --predictions to run conformance tests"
        )
    if not p.is_dir():
        pytest.fail(f"predictions dir does not exist: {p}")
    return p


@pytest.fixture(scope="session")
def fixture_manifest(request: pytest.FixtureRequest) -> tuple[Path, FixtureManifest] | None:
    """``(fixture_dir, manifest)`` with the ground truth hash-checked, or ``None``."""
    from rtmask_conformance.verify import FixtureIntegrityError, check_fixture_integrity

    p = resolve_fixture_dir(request.config)
    if p is None:
        return None
    if not (p / "manifest.json").is_file():
        pytest.fail(f"fixture dir has no manifest.json: {p}")
    try:
        manifest = check_fixture_integrity(p)
    except (FileNotFoundError, FixtureIntegrityError) as e:
        pytest.fail(str(e))
    return p, manifest


@pytest.fixture(scope="session")
def groundtruth_dir(request: pytest.FixtureRequest, fixture_manifest) -> Path:
    if fixture_manifest is not None:
        fixture_dir, manifest = fixture_manifest
        return fixture_dir / manifest.groundtruth_dir
    p = _resolve_dir(
        request.config.getoption("--groundtruth"),
        "RTMASK_CONFORMANCE_GROUNDTRUTH",
    )
    if p is None:
        pytest.skip(
            "set RTMASK_CONFORMANCE_FIXTURE (or RTMASK_CONFORMANCE_GROUNDTRUTH), or pass "
            "--fixture (or --groundtruth), to run conformance tests"
        )
    if not p.is_dir():
        pytest.fail(f"groundtruth dir does not exist: {p}")
    return p
