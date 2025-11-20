from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DataLoader:
	def loadDataCSV(self, filename: str, **read_csv_kwargs: Any) -> pd.DataFrame:

		if not filename:
			raise ValueError("filename must be a non-empty string")

		# Resolve repository root by walking up until we find a directory containing `_data`.
		# Start from the directory containing this file and walk parents.
		current = Path(__file__).resolve()
		repo_root = None
		for p in [current.parent] + list(current.parents):
			if (p / "_data").exists():
				repo_root = p
				break
		if repo_root is None:
			# Fallback: assume repo root is two levels up (ama_ws25) if layout matches
			try:
				repo_root = current.parents[2]
			except IndexError:
				repo_root = current.parent

		data_dir = repo_root / "_data"
		if not data_dir.exists() or not data_dir.is_dir():
			raise FileNotFoundError(f"_data directory not found at expected location: {data_dir}")

		file_path = data_dir / filename
		if not file_path.exists():
			# helpful listing of available files
			available = sorted([p.name for p in data_dir.iterdir() if p.is_file()])
			raise FileNotFoundError(
				f"File '{filename}' not found in _data folder ({data_dir}).\n"
				f"Available files: {available}"
			)

		# Use pandas to read the CSV. Let pandas raise its own errors for parse issues.
		return pd.read_csv(file_path, **read_csv_kwargs)


__all__ = ["DataLoader"]

