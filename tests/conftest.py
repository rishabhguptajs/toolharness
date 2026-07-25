from __future__ import annotations

from pathlib import Path

import pytest

from evalharness.adapters import default_registry
from evalharness.adapters.base import RunSource
from evalharness.core.model import NormalizedSession

FIXTURES = Path(__file__).parent / "fixtures"


def load_session(name: str) -> NormalizedSession:
    source = RunSource(kind="generic", path=FIXTURES / f"{name}.json")
    return default_registry.parse(source)


@pytest.fixture
def session_loader():
    return load_session
