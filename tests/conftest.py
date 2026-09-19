from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mitrity.admission import Client
from tests.fake_edge import FakeEdge


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def edge(tmp_path: Path) -> Iterator[FakeEdge]:
    fake = FakeEdge(tmp_path)
    try:
        yield fake
    finally:
        fake.close()


@pytest.fixture
def client(edge: FakeEdge) -> Client:
    return Client(addr=edge.addr, token_file=str(edge.token_file), timeout=0.5, hold_timeout=2.0)
