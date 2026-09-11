# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Moutasem Akkad <moutasemalakkad@gmail.com>
"""Execute every code cell of both notebooks so they cannot silently rot.

Each notebook runs top to bottom in one shared namespace, with notebook/ as
the working directory — the same thing a reader does. All cells are offline
and free (SQLite + subprocess to the CLI), so this belongs in the fast suite.
"""
import json
import pathlib

NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parent.parent / "notebook"


def _run_notebook(name: str) -> None:
    """Exec the notebook's code cells in order; raise with the failing cell."""
    cells = json.loads((NOTEBOOK_DIR / name).read_text())["cells"]
    namespace = {}
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        if isinstance(source, list):  # nbformat allows both string and list
            source = "".join(source)
        try:
            exec(compile(source, f"{name}:cell{index}", "exec"), namespace)
        except Exception as error:
            raise AssertionError(f"{name} cell {index} failed: {error}") from error


def test_quickstart_notebook_executes_offline(monkeypatch):
    monkeypatch.chdir(NOTEBOOK_DIR)
    _run_notebook("quickstart.ipynb")


def test_internals_notebook_executes_offline(monkeypatch):
    monkeypatch.chdir(NOTEBOOK_DIR)
    _run_notebook("internals.ipynb")
