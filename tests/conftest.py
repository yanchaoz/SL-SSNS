"""Isolate the original stage-local model/utils module namespaces in tests."""

import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stage_import(monkeypatch):
    names = (
        "model",
        "utils",
        "loss",
        "selection",
        "CGS_tools",
        "pretrain_provider",
        "training_utils",
        "pretrain",
        "learner",
    )

    def clear():
        for name in list(sys.modules):
            if name.split(".")[0] in names:
                sys.modules.pop(name)

    clear()

    def load(stage, module):
        monkeypatch.syspath_prepend(str(ROOT / stage))
        return importlib.import_module(module)

    yield load
    clear()
