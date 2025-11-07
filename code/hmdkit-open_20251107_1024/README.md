# Human–Machine Difficulty Alignment Kit (Open Release)

This repository contains code-only assets to reproduce analyses of
human problem difficulty vs. model error/confidence across datasets.

No original datasets or generated outputs are included.
Prepare datasets under ./data/ following data/README.md if needed.

Quick start:
  pip install -e .
  (optional) place datasets under ./data as described in data/README.md
  run your own pipeline using scripts in ./code

Included:
- Collected Python/Shell/Config files from the original research repo (under ./code/).
- Minimal scaffolding: MIT License, requirements, pyproject, CI, tests.
- No data/, tables/, figs/, analysis/, or artifacts in this open release.

License:
- Code: MIT (see LICENSE)
- Derived tables/figures in papers: CC BY 4.0 (not included here)

Citation:
See CITATION.cff.
