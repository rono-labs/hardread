import pytest

from hardread import Analyzer


@pytest.fixture(scope="session")
def analyzer():
    return Analyzer()
