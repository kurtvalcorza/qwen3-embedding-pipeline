"""The workshop's uv isolated environment (2026-10-03): nothing installed into the kernel, no restart guard.

Static checks of the generated setup cell and the carried lock, plus (on POSIX) the real worker process
started with this interpreter standing in for the isolated environment's Python. They do not establish a
hosted run.
"""
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_Qwen3_Semantic_Search_Reranking_Workshop.ipynb"
LOCK = REPO / "tutorials" / "requirements-semantic-search-workshop.lock.txt"
PINS_IN = REPO / "tutorials" / "requirements-semantic-search-workshop.in"
SETUP_INDEX = 6
# The pins the notebook installed into the kernel before 2026-10-03 (blob 99a726dc); the move changes none.
PREVIOUS_PINS = {
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "torchaudio": "2.11.0",
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
    "numpy": "2.1.3",
}

sys.path.insert(0, str(REPO / "tools"))
import semantic_search_isolated_runtime as runtime  # noqa: E402


def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def source(index):
    return "".join(cells()[index]["source"])


def code_sources():
    return ["".join(c["source"]) for c in cells() if c["cell_type"] == "code"]


def setup_tree():
    return ast.parse(source(SETUP_INDEX))


def assigned(tree, name):
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        if isinstance(target, ast.Name) and target.id == name:
            return node.value
    raise AssertionError(f"{name} is not assigned at top level")


def test_no_kernel_install_and_no_restart_guard():
    for text in code_sources():
        assert not re.search(r"pip\s+install|\"-m\",\s*\"pip\"", text), "a cell still runs pip in the kernel"
        # The only pip is uv's, installing into the isolated environment's Python.
        uv_install = '[str(uv), "pip", "install", "--quiet", "--python", str(ISOLATED_PYTHON)'
        assert text.count('"pip"') == text.count(uv_install)
        assert "NUMPY_PRELOADED" not in text and "importlib_metadata" not in text
        assert "Restart session" not in text and "Stale modules" not in text
        assert "sys.executable" not in text, "nothing may be installed into or run with the kernel's Python"


def test_setup_cell_is_a_kernel_cell_that_imports_only_stdlib_and_ipython():
    text = source(SETUP_INDEX)
    assert "# dimer: kernel cell" in text.splitlines()[1]
    top = set()
    for node in setup_tree().body:
        if isinstance(node, ast.Import):
            top.update(a.name.partition(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add(node.module.partition(".")[0])
    allowed = {"hashlib", "io", "os", "platform", "signal", "subprocess", "sys", "time", "urllib", "zipfile",
               "multiprocessing", "pathlib", "IPython"}
    assert top <= allowed, top


def test_pins_are_unchanged_and_the_lock_is_hash_locked():
    pins = runtime.pins()
    assert dict(p.split("==", 1) for p in pins) == PREVIOUS_PINS
    assert ast.literal_eval(assigned(setup_tree(), "PINS")) == pins
    lock = LOCK.read_text(encoding="utf-8")
    locked = runtime.lock_packages(lock)
    for name, version in PREVIOUS_PINS.items():
        assert locked[name] == version
    entries = re.split(r"\n(?=[A-Za-z0-9])", lock)
    requirements = [e for e in entries if "==" in e.split("\n", 1)[0]]
    assert len(requirements) == len(locked) >= len(PREVIOUS_PINS)
    assert all("--hash=sha256:" in e for e in requirements)
    assert "\r" not in lock and "'''" not in lock
    assert "--only-binary :all:" in lock.splitlines()[1] and "--generate-hashes" in lock.splitlines()[1]
    assert "x86_64-manylinux_2_28" in lock.splitlines()[1] and "--python-version 3.12" in lock.splitlines()[1]


def test_carried_lock_equals_the_committed_lock_and_its_digest():
    tree = setup_tree()
    carried = ast.literal_eval(assigned(tree, "LOCK_TEXT"))
    lock = LOCK.read_text(encoding="utf-8")
    assert carried == lock
    assert ast.literal_eval(assigned(tree, "LOCK_SHA256")) == hashlib.sha256(lock.encode("utf-8")).hexdigest()
    assert ast.literal_eval(assigned(tree, "LOCKED_PACKAGES")) == len(runtime.lock_packages(lock))
    assert ast.literal_eval(assigned(tree, "LOCK_NAME")) == LOCK.name


def test_install_is_hash_checked_wheel_only_into_a_managed_python():
    text = source(SETUP_INDEX)
    assert '"--require-hashes", "--only-binary", ":all:"' in text
    assert '"venv", "--quiet", "--managed-python", "--python", MANAGED_PYTHON' in text
    tree = setup_tree()
    assert ast.literal_eval(assigned(tree, "MANAGED_PYTHON")) == runtime.MANAGED_PYTHON == "3.12.12"
    assert ast.literal_eval(assigned(tree, "UV_URL")) == runtime.UV["url"]
    assert ast.literal_eval(assigned(tree, "UV_BYTES")) == runtime.UV["bytes"]
    assert ast.literal_eval(assigned(tree, "UV_SHA256")) == runtime.UV["sha256"]
    assert "hashlib.sha256(wheel).hexdigest() != UV_SHA256" in text
    assert 'platform.system() != "Linux" or platform.machine() != "x86_64"' in text
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        assert f'"{name}"' in text
    assert 'MPLBACKEND="Agg"' in text


def test_every_later_cell_runs_in_the_isolated_python():
    text = source(SETUP_INDEX)
    assert "_DIMER_ISOLATED_RUNTIME = IsolatedRuntime(ISOLATED_PYTHON)" in text
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in text
    assert 'ISOLATED_PYTHON = ISOLATED_ENV / "bin" / "python"' in text
    assert "[str(python), \"-c\", _WORKER_SOURCE" in text
    # Only the setup cell stays in the kernel; every model stage is routed.
    kernel = [i for i, c in enumerate(cells())
              if c["cell_type"] == "code" and "# dimer: kernel cell" in "".join(c["source"])]
    assert kernel == [SETUP_INDEX]


def test_configuration_is_forwarded_to_the_worker():
    names = {t.id for n in ast.parse(source(4)).body if isinstance(n, ast.Assign) for t in n.targets
             if isinstance(t, ast.Name)}
    forwarded = set(ast.literal_eval(assigned(setup_tree(), "CONFIGURATION_NAMES")))
    assert forwarded == names
    assert source(4).index("USE_BYOD") < source(SETUP_INDEX).index("CONFIGURATION_NAMES")


def test_runtime_inventory_defines_what_later_cells_use():
    inventory = ast.literal_eval(assigned(setup_tree(), "RUNTIME_INVENTORY"))
    defined = {t.id for n in ast.parse(inventory).body if isinstance(n, ast.Assign) for t in n.targets
               if isinstance(t, ast.Name)}
    assert {"DEVICE", "DTYPE", "RUNTIME"} <= defined
    for module in ("numpy as np", "torch", "transformers", "huggingface_hub", "safetensors"):
        assert f"import {module}" in inventory


def test_no_cell_line_exceeds_2000_characters():
    for index, cell in enumerate(cells()):
        for line in "".join(cell["source"]).splitlines():
            assert len(line) <= 2000, f"cell {index}: {len(line)} characters"


@pytest.mark.parametrize("length", [0, 1, 999, 1000, 1001, 2500, 4321])
def test_split_literal_pieces_reassemble(length):
    value = "".join(chr(ord("a") + i % 26) for i in range(length)) + "'\"\\"
    rendered = runtime.split_literal(value)
    node = ast.parse(f"x = (\n{rendered}\n)").body[0].value
    assert ast.literal_eval(node) == value
    # Implicitly concatenated pieces parse to one Constant, so each rendered line is evaluated on its own.
    pieces = [ast.literal_eval(line.strip()) for line in rendered.splitlines()]
    assert "".join(pieces) == value
    assert all(len(p) <= runtime.MAX_PIECE for p in pieces)
    assert all(len(line) <= 2000 for line in rendered.splitlines())


def test_generator_refuses_a_line_over_the_limit(monkeypatch):
    import build_semantic_search_reranking_workshop as build

    monkeypatch.setattr(build, "setup_cell", lambda: "# " + "y" * 1999)
    with pytest.raises(SystemExit, match="2001-character line"):
        build.cell_sources()


def test_metadata_records_the_isolated_environment():
    meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]
    iso = meta["isolated_runtime"]
    assert iso["managed_python"] == "3.12.12" and iso["platform"] == "Linux x86_64 only"
    assert iso["lock"] == f"tutorials/{LOCK.name}" and iso["pins"] == runtime.pins()
    assert meta["revisions"][-1]["date"] == "2026-10-03"
    assert meta["revisions"][-1]["previous_blob"] == "99a726dcb8bd92f4a5ff0dd5fbc190b6a070629f"


def test_learner_text_states_the_platform_and_no_restart():
    assert "Linux x86_64" in source(1) and "Linux x86_64" in source(5)
    assert "never asks for one" in source(5)
    assert "Restart session" not in source(5)


def _worker_namespace():
    """The setup cell's worker/router definitions, without running the install or touching IPython."""
    tree = setup_tree()
    keep = []
    for node in tree.body:
        target = node.targets[0] if isinstance(node, ast.Assign) else None
        wanted = ("_WORKER_SOURCE", "CONFIGURATION_NAMES", "RUNTIME_INVENTORY")
        if (isinstance(target, ast.Name) and target.id in wanted) or isinstance(
            node, (ast.ClassDef, ast.FunctionDef)
        ):
            keep.append(node)
    ns = {"os": os, "signal": __import__("signal"), "subprocess": subprocess, "sys": sys}
    from multiprocessing.connection import Connection

    ns["Connection"] = Connection
    exec(compile(ast.Module(body=keep, type_ignores=[]), "setup-cell-worker", "exec"), ns)
    return ns


@pytest.mark.skipif(os.name != "posix", reason="the worker uses POSIX pipes (Colab/Kaggle/Linux only)")
def test_real_worker_runs_forwarded_configuration_and_reports_errors(monkeypatch):
    ns = _worker_namespace()
    shown = []
    worker = ns["IsolatedRuntime"](sys.executable, display=lambda data, raw=True: shown.append(data))
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    try:
        config = {"USE_BYOD": False, "RERANK_K": 6, "K_SWEEP_VALUES": [3, 6, 10], "OUTPUT_DIR": "o/q"}
        worker.run("".join(f"{k} = {v!r}\n" for k, v in config.items()))
        worker.run("import sys, os\nprint(USE_BYOD, RERANK_K, K_SWEEP_VALUES, OUTPUT_DIR)\n"
                   "print(os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED'), os.environ.get('MPLBACKEND'))\n"
                   "RERANK_K + 1")
        assert "False 6 [3, 6, 10] o/q" in out.getvalue()
        assert "1 Agg" in out.getvalue()
        assert shown and shown[-1]["text/plain"] == "7"
        with pytest.raises(ns["IsolatedCellError"], match="ZeroDivisionError"):
            worker.run("1 / 0")
        worker.run("print('still alive', RERANK_K)")
        assert "still alive 6" in out.getvalue()
        route = ns["_route_to_isolated_runtime"]
        assert route(["# dimer: kernel cell\nx = 1\n"]) == ["# dimer: kernel cell\nx = 1\n"]
        assert route(["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
    finally:
        worker.close()
