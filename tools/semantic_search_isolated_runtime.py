"""Render the workshop's isolated-environment setup cell (the fleet's uv mechanism).

The cell installs nothing into the notebook kernel. It downloads a size- and SHA-256-pinned uv wheel, creates
a managed CPython environment, installs the hash-locked, wheel-only requirements into it, and routes every
later code cell to one persistent worker process there, so Run all never needs a session restart. The
worker/router source is the fleet's verified carrier (bart-mnli-zero-shot-classification-pipeline ee128d2,
prithvi-eo-feature-extraction-pipeline eo_workshop). Linux x86_64 only.

The lock is compiled with the command recorded in its own header (``uv pip compile`` of
``tutorials/requirements-semantic-search-workshop.in`` for Python 3.12 on ``x86_64-manylinux_2_28`` with
``--generate-hashes --only-binary :all:``).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "tools" / "semantic_search_isolated_setup_cell.txt"
PINS_FILE = REPO / "tutorials" / "requirements-semantic-search-workshop.in"
LOCK_FILE = REPO / "tutorials" / "requirements-semantic-search-workshop.lock.txt"
SETUP_CELL_INDEX = 6
SETUP_CELL_PLACEHOLDER = (
    "# GENERATED: isolated-environment setup cell (tools/semantic_search_isolated_runtime.py)\n"
)
MANAGED_PYTHON = "3.12.12"
UV = {
    "version": "0.12.15",
    "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "bytes": 20081404,
    "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
}
# Literal pieces carried in a cell stay well under the 2,000-character line limit.
MAX_PIECE = 1000


def pins() -> list[str]:
    lines = PINS_FILE.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def lock_text() -> str:
    return LOCK_FILE.read_text(encoding="utf-8")


def lock_packages(text: str) -> dict[str, str]:
    """`{name: version}` of every requirement in a uv/pip-compile hash lock."""
    matches = re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", text, re.M)
    return {m.group(1).lower(): m.group(2) for m in matches}


def check_lock(direct: list[str], text: str) -> None:
    """Every direct pin is locked at its version, every lock entry has a hash, and the text is carriable."""
    locked = lock_packages(text)
    for pin in direct:
        name, version = pin.split("==", 1)
        if locked.get(name.lower()) != version:
            found = locked.get(name.lower())
            raise SystemExit(f"lock does not pin {pin} (found {found}); recompile the lock")
    blocks = re.split(r"\n(?=[A-Za-z0-9])", text)
    unhashed = [b.split("==", 1)[0] for b in blocks if "==" in b and "--hash=sha256:" not in b]
    if unhashed:
        raise SystemExit(f"lock entries without --hash: {unhashed}")
    if "'''" in text or "\r" in text:
        raise SystemExit(
            "lock text cannot be carried in a raw triple-quoted literal (no ''' and LF line endings only)"
        )


def split_literal(value: str, indent: str = "    ") -> str:
    """A parenthesised run of string pieces of at most MAX_PIECE characters that evaluates to ``value``."""
    pieces = [value[i:i + MAX_PIECE] for i in range(0, len(value), MAX_PIECE)] or [""]
    return "\n".join(f"{indent}{piece!r}" for piece in pieces)


def setup_cell() -> str:
    direct = pins()
    text = lock_text()
    check_lock(direct, text)
    replacements = {
        "@@PINS@@": "\n".join(f"    {pin!r}," for pin in direct).replace("'", '"'),
        "@@MANAGED_PYTHON@@": MANAGED_PYTHON,
        "@@UV_URL@@": split_literal(UV["url"]),
        "@@UV_BYTES@@": str(int(UV["bytes"])),
        "@@UV_SHA256@@": UV["sha256"],
        "@@LOCK_NAME@@": LOCK_FILE.name,
        "@@LOCK_SHA256@@": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "@@LOCKED_PACKAGES@@": str(len(lock_packages(text))),
        "@@LOCK_TEXT@@": text,
    }
    source = TEMPLATE.read_text(encoding="utf-8")
    for key, value in replacements.items():
        if key not in source:
            raise SystemExit(f"setup-cell template is missing {key}")
        source = source.replace(key, value)
    if "@@" in source:
        raise SystemExit("setup-cell template has an unreplaced placeholder")
    return source.rstrip("\n")
