"""pytest fixtures shared across the conformance test modules.

Consumers run the conformance tests against their own predictions like:

    RTMASK_CONFORMANCE_PREDICTIONS=./predictions \
    RTMASK_CONFORMANCE_GROUNDTRUTH=./fixture/groundtruth \
    pytest --pyargs rtmask_conformance.tests::test_conformance

Or via the equivalent ``--predictions``/``--groundtruth`` pytest options.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--predictions",
        action="store",
        default=None,
        help="Directory containing predicted <roi>.nii.gz files.",
    )
    parser.addoption(
        "--groundtruth",
        action="store",
        default=None,
        help="Directory containing ground-truth <roi>.nii.gz files.",
    )


def _resolve_dir(arg: str | None, env: str) -> Path | None:
    raw = arg if arg is not None else os.environ.get(env)
    if raw is None:
        return None
    return Path(raw)


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
def groundtruth_dir(request: pytest.FixtureRequest) -> Path:
    p = _resolve_dir(
        request.config.getoption("--groundtruth"),
        "RTMASK_CONFORMANCE_GROUNDTRUTH",
    )
    if p is None:
        pytest.skip(
            "set RTMASK_CONFORMANCE_GROUNDTRUTH or pass --groundtruth to run conformance tests"
        )
    if not p.is_dir():
        pytest.fail(f"groundtruth dir does not exist: {p}")
    return p
