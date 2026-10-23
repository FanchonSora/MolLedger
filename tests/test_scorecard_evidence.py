"""The scorecard must actually receive the two optional evidence sources it documents.

``prepare`` used to call ``analyze_feature_evidence`` without prior-coefficient evidence and
without model contributions, so two of the five evidence classes could never be assigned and
the scorecard silently reported support/collinearity only.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from chemistry_framework.config import load_config
from chemistry_framework.experiment import prepare_experiment


TABLE = [
    ("CCO", -0.77), ("c1ccccc1O", -1.46), ("CC(=O)O", 0.09), ("CCCCCCO", -1.17),
    ("c1ccncc1", -0.71), ("CC(C)O", 0.34), ("CCOCC", -0.83), ("CCN", 0.36),
    ("c1ccc2ccccc2c1", -2.9), ("O=C(O)c1ccccc1", -1.87), ("CCCl", -0.62), ("COc1ccccc1", -1.55),
    ("OCC(O)CO", 0.0), ("CCCCCCCC", -3.1), ("c1ccsc1", -1.2), ("CC(=O)Nc1ccccc1", -1.0),
    ("O=S(=O)(N)c1ccccc1", -1.3), ("CCCCO", -0.9), ("C1CCCCC1", -2.5), ("Nc1ccccc1", -1.1),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", -3.5), ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", -1.0),
    ("OC(=O)c1ccccc1O", -1.5), ("CCN(CC)CC", -0.3), ("C1CCOC1", -0.5),
]


def _write_config(tmp_path: Path) -> Path:
    table = tmp_path / "table.csv"
    with table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["smiles", "logS"])
        writer.writerows(TABLE)
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"""name: scorecard
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
    names: [MolLogP, MolWt, TPSA, NumHDonors, NumHAcceptors]
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
training: {{epochs: 1, batch_size: 4, device: cpu}}
""",
        encoding="utf-8",
    )
    return config


def test_scorecard_receives_prior_and_model_evidence(tmp_path):
    experiment = prepare_experiment(load_config(_write_config(tmp_path)))
    evidence = experiment.feature_evidence
    assert evidence is not None
    assert evidence.model_contributions, "no model contributions were supplied to the scorecard"

    by_name = {item.name: item for item in evidence.features}
    prior = by_name["physchem.MolLogP"].prior_evidence
    assert prior is not None and prior["count"] > 1
    assert evidence.model_contributions
    groups = {item.group for item in evidence.model_contributions}
    assert "rdkit2d" in groups
