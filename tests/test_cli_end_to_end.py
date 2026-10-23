"""Regression test for the documented end-to-end path: validate -> prepare -> train -> predict.

The hybrid forward pass used to combine Tensor inputs with ``or`` (which raises for batches with
more than one row) and ``predict`` used to rebuild a model without its frozen prior and without
descriptor inputs, so the documented workflow could not be run at all.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from chemistry_framework.cli import main as cli_main


TABLE = [
    ("CCO", -0.77), ("c1ccccc1O", -1.46), ("CC(=O)O", 0.09), ("CCCCCCO", -1.17),
    ("c1ccncc1", -0.71), ("CC(C)O", 0.34), ("CCOCC", -0.83), ("CCN", 0.36),
    ("c1ccc2ccccc2c1", -2.9), ("O=C(O)c1ccccc1", -1.87), ("CCCl", -0.62),
    ("COc1ccccc1", -1.55), ("OCC(O)CO", 0.0), ("CCCCCCCC", -3.1),
    ("c1ccsc1", -1.2), ("CC(=O)Nc1ccccc1", -1.0), ("O=S(=O)(N)c1ccccc1", -1.3),
    ("CCCCO", -0.9), ("C1CCCCC1", -2.5), ("Nc1ccccc1", -1.1),
]


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    table = tmp_path / "table.csv"
    with table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "logS"])
        writer.writerows(TABLE)
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"""name: cli-e2e
dataset:
  provider: csv
  path: {table.as_posix()}
  smiles_column: smiles
  targets:
    logS: {{column: logS}}
features:
  physchem:
    type: rdkit_descriptors
    names: [MolWt, MolLogP, TPSA]
  rdkit2d:
    type: rdkit_descriptors
    families: [all]
priors:
  logS_prior: {{use: [physchem], fit: ridge}}
routes:
  logS: [rdkit2d]
tasks:
  logS: {{target: logS, kind: scalar, route: logS, prior: logS_prior}}
split: {{method: scaffold, fractions: [0.6, 0.2, 0.2], seed: 1}}
model:
  structure: {{encoder: dmpnn, hidden_dim: 16, depth: 2}}
  descriptors: {{enabled: true}}
  fusion: concat
  prior: frozen
training: {{epochs: 1, batch_size: 4, device: cpu, checkpoint_dir: {(tmp_path / 'ckpt').as_posix()}}}
""",
        encoding="utf-8",
    )
    return config, table


def test_prepare_train_predict(tmp_path, monkeypatch):
    config, table = _write_config(tmp_path)
    assert cli_main(["validate", str(config)]) == 0
    assert cli_main(["prepare", str(config)]) == 0
    assert cli_main(["train", str(config)]) == 0

    checkpoints = sorted((tmp_path / "ckpt").glob("*.pt"))
    assert checkpoints, "train did not write a checkpoint"

    output = tmp_path / "predictions.csv"
    assert cli_main(
        ["predict", "--checkpoint", str(checkpoints[0]), "--input", str(table), "--output", str(output)]
    ) == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert len(rows) == len(TABLE)
    assert all(row["prediction"] for row in rows)
