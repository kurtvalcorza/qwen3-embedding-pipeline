"""Colab's already imported NumPy is never touched: the setup cell installs nothing into the kernel.

Updated 2026-10-03 for the uv isolated environment. The earlier version executed the in-kernel pip install
with a simulated preloaded NumPy; that install no longer exists, so this test runs the real setup cell up to
its platform gate and checks that nothing in the kernel was installed, imported or replaced.
"""
import ast
import json
import platform
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_setup_cell_leaves_the_kernels_preloaded_numpy_alone(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / "tutorials" / "DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][6]["source"])
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.partition(".")[0] in {"numpy", "torch", "transformers"} for a in node.names)
    loaded_numpy = SimpleNamespace(__version__="2.1.3")
    monkeypatch.setitem(sys.modules, "numpy", loaded_numpy)
    fake_ipython = types.ModuleType("IPython")
    fake_ipython.get_ipython = lambda: None
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(tmp_path / "env"))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(RuntimeError, match="Linux x86_64"):
        exec(compile(tree, "notebook-setup", "exec"), {"__name__": "__main__"})
    assert calls == []
    assert sys.modules["numpy"] is loaded_numpy
    assert not (tmp_path / "env").exists()
