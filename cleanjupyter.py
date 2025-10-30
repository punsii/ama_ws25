"""Utility to find Jupyter notebooks and call a cleaning function on them.

This module provides a small, safe boilerplate that recursively searches for
`*.ipynb` files starting from the directory containing this file (or an
override path provided on the command line). For every notebook found it
calls `clean(path_no_notebook)` where `path_no_notebook` is the notebook path
without the `.ipynb` suffix (string).

The actual cleaning logic is intentionally left as a stub in `clean()` so you
can implement project-specific behavior.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_notebooks(start_path: str | Path) -> Iterator[Path]:
	"""Recursively yield Path objects for all .ipynb files under start_path.

	Args:
		start_path: Directory path to start searching from.

	Yields:
		Path objects pointing to found .ipynb files.
	"""
	p = Path(start_path)
	if not p.exists():
		logger.warning("start_path does not exist: %s", start_path)
		return
	# Use rglob which is efficient and readable
	for nb in p.rglob("*.ipynb"):
		# skip hidden directories by convention
		if any(part.startswith(".") for part in nb.parts):
			continue
		yield nb
            

def clean(path_no_notebook: str) -> None:
    """Placeholder cleaning function.

    Implement your notebook-cleaning logic here. `path_no_notebook` is the
    notebook path without the `.ipynb` suffix (string). For example, if the
    notebook is `foo/bar/baz.ipynb`, `path_no_notebook` will be
    `foo/bar/baz`.

    Keep the signature stable so this boilerplate can call it directly.
    """
    # Open the corresponding .ipynb file and print the cell_type of each cell.
    nb_path = Path(path_no_notebook).with_suffix(".ipynb")
    if not nb_path.exists():
        logger.warning("Notebook file not found: %s", nb_path)
        return

    try:
        import json

        with nb_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        cells = data.get("cells")
        if not isinstance(cells, list):
            logger.warning("No cells list found in notebook: %s", nb_path)
            return

        # Print the cell_type for every cell (index, type) and clear code outputs
        modified = 0
        for idx, cell in enumerate(cells, start=1):
            ctype = cell.get("cell_type") if isinstance(cell, dict) else None
            print(f"{nb_path}: cell {idx} -> {ctype}")
            if ctype == "code" and isinstance(cell, dict):
                # clear outputs and reset execution count so the notebook is clean
                # only rewrite if there is something to change
                if cell.get("outputs") != [] or cell.get("execution_count") is not None:
                    cell["outputs"] = []
                    # optional: clear execution count to indicate unrunned cell
                    cell["execution_count"] = None
                    modified += 1

        if modified:
            # overwrite the notebook with cleaned cells
            with nb_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=1)
            logger.info("Cleared outputs for %d code cells in %s", modified, nb_path)
    except Exception:
        logger.exception("Failed to read or process notebook: %s", nb_path)

def _process_notebooks(start_path: str | Path) -> int:
	"""Find notebooks and call clean() for each.

	Returns the number of notebooks processed.
	"""
	processed = 0
	for nb in find_notebooks(start_path):
		# create a path without the .ipynb suffix
		path_no_notebook = str(nb.with_suffix(""))
		try:
			clean(path_no_notebook)
			processed += 1
		except Exception:
			logger.exception("Error while cleaning notebook: %s", nb)
	return processed


def _default_start_path() -> str:
	"""Return the directory containing this file, as a string.

	This matches the user's request to search from the current path of this
	file.
	"""
	return str(Path(__file__).resolve().parent)


def _build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Recursively find .ipynb files and call clean() on each.")
	p.add_argument(
		"path",
		nargs="?",
		default=None,
		help="Optional start directory. If omitted, the directory containing this file is used.",
	)
	return p


if __name__ == "__main__":
	parser = _build_arg_parser()
	args = parser.parse_args()

	start = args.path or _default_start_path()
	logger.info("Searching for notebooks under: %s", start)
	count = _process_notebooks(start)
	logger.info("Done. Notebooks processed: %d", count)

