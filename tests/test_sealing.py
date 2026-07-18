"""Specify offline tamper-evidence behavior from PRD sections 8 and 10."""

import pytest


@pytest.mark.skip(reason="Sealing logic has not been implemented.")
def test_not_implemented() -> None:
    raise NotImplementedError("Sealing tests are not implemented yet.")
