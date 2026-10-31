import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="also run tests that load the real models")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the real models (downloads them on first run)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="needs --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
