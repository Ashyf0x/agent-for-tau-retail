import builtins
import io
import os
import socket
from pathlib import Path

import pytest

# Set before importing upstream: its import-time dotenv and cost-map hooks are disabled.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


@pytest.fixture(autouse=True)
def offline_guard(monkeypatch):
    original_connect = socket.socket.connect

    def connect(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("Offline tests cannot contact external endpoints")
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    for module in (builtins, io):
        original = module.open

        def checked(file, *args, _original=original, **kwargs):
            if isinstance(file, (str, os.PathLike)) and Path(file).name == ".env":
                raise AssertionError("Offline tests must not read .env")
            return _original(file, *args, **kwargs)

        monkeypatch.setattr(module, "open", checked)


@pytest.fixture
def store(tmp_path):
    from agentcore.storage import Store

    instance = Store(tmp_path / "agent.sqlite3")
    yield instance
    instance.close()
