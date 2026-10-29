from molledger.analysis.advisor import _structure_conflicts


def test_structure_conflicts_counts_inchikey_duplicates_and_label_conflicts():
    # Same structure written two ways (InChIKey match), one pair with conflicting labels.
    smiles = ["CCO", "OCC", "CCN", "NCC", "c1ccccc1"]
    labels = [1, 1, 0, 1, 0]
    dup, records, conflicting, rows = _structure_conflicts(smiles, labels, is_classification=True)
    assert dup == 2
    assert records == 4
    assert conflicting == 1
    assert len(rows) == 1
    assert sorted(rows[0]["labels"]) == [0.0, 1.0]


def test_structure_conflicts_regression_uses_half_sd_span():
    smiles = ["CCO", "OCC", "CCN", "NCC", "c1ccccc1"]
    values = [1.0, 1.1, 0.0, 3.0, 2.0]
    dup, records, conflicting, _ = _structure_conflicts(smiles, values, is_classification=False)
    assert dup == 2
    assert records == 4
    assert conflicting == 1
