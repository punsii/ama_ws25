"""Test configuration for AMA toolbox."""

from pathlib import Path
import sys
import types

import matplotlib
import pytest


matplotlib.use("Agg")

# Ensure the local package is importable when the repo isn't installed.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Provide a lightweight wandb stub so importing ama_tlbx does not require the SDK.
if "wandb" not in sys.modules:
    DummyRun = type("DummyRun", (), {})  # minimal placeholder
    DummyArtifact = type("DummyArtifact", (), {})

    def _noop(*args, **kwargs):  # type: ignore[return-type]
        return None

    wandb_stub = types.SimpleNamespace(
        init=lambda *args, **kwargs: DummyRun(),
        log=_noop,
        Artifact=lambda *args, **kwargs: DummyArtifact(),
        Settings=lambda *args, **kwargs: None,
    )
    wandb_stub.sdk = types.SimpleNamespace(
        wandb_run=types.SimpleNamespace(Run=DummyRun),
        wandb_artifacts=types.SimpleNamespace(Artifact=DummyArtifact),
    )
    sys.modules["wandb"] = wandb_stub


@pytest.fixture(scope="session")
def life_expectancy_dataset():
    """Load the real life expectancy dataset once per test session."""
    from ama_tlbx.data import LifeExpectancyDataset

    return LifeExpectancyDataset.from_csv()


@pytest.fixture(scope="session")
def life_expectancy_df(life_expectancy_dataset):
    """Tabular fixture with common predictors and target, NaNs removed."""
    from ama_tlbx.data import LECol

    cols = [LECol.TARGET, LECol.GDP, LECol.BMI, LECol.ALCOHOL, LECol.SCHOOLING]
    return life_expectancy_dataset.df.reset_index(drop=True)[cols].dropna().astype(float)
