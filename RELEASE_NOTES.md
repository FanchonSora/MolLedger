# Release notes for molledger 1.0.0

Version 1.0.0 is the archived version of the software reported in the Journal of
Cheminformatics **Software article** "molledger: a configuration-driven and
replayable workflow for molecular property modelling".

This repository is the archived release of that work. It is frozen: further
development happens elsewhere, and the manuscript cites the `v1.0.0` tag rather than
the default branch. The fixed URL is

    https://github.com/FanchonSora/MolLedger/releases/tag/v1.0.0

## Before creating the release

- [ ] No placeholder metadata in the release: `CITATION.cff` carries the final author
      name and affiliation (currently `TO-BE-FILLED`).
- [ ] The tree is clean: every file is committed, so the tag points at exactly the
      code that the manuscript cites.
- [ ] `reproduce.py` runs from a fresh clone and regenerates `benchmarks/results/`.
- [ ] The test suite passes.

## Creating the release

1. Tag the release commit: `git tag -a v1.0.0 -m "Version cited by paper 1"`.
2. Push the tag and the branch: `git push origin main --tags`.
3. Create the GitHub release `v1.0.0` from that tag. The release page is the stable
   identifier used in the manuscript's "Archived releases" line and in the reference
   to the software.
4. If the release URL ever changes, update
   - the manuscript, Availability and requirements, "Archived releases" line;
   - the manuscript, reference to the software;
   - `SUBMISSION_CHECKLIST.md`, section 6.

## What the release contains

- `molledger/` - the framework (configuration system, feature analysis and
  data audit, hybrid modelling layer).
- `benchmarks/splits/` - 20 Bemis-Murcko scaffold manifests per endpoint.
- `benchmarks/results/` - per-split results in CSV and JSON, including the files that
  regenerate every table and figure of the manuscript.
- `benchmarks/` runners and statistics scripts, `tests/`, `docs/`, `reproduce.py`,
  `REPRODUCING.md`, pinned environments and the MIT `LICENSE`.

## Notes

- The manuscript sources are not part of this repository; the manuscript is
  distributed by the journal.
- The follow-up feature-space work (SIREN routing) is not part of this version.
