# RAGES

Reasoning-Augmented Guidance for Earth-orbit Spacecraft (extracted from the `art_lang` project).

- `src/` — core library: `optimization/`, `dynamics/`, plus `utils.py`, `parameters.py`, `rages.py`.
- `work/` — runnable scripts: data generation, training, and analysis.

Both `src/` and `work/` are placed on the import path via the editable install (see `[tool.hatch.build.targets.wheel]` in `pyproject.toml`), so modules are imported by flat name (e.g. `from optimization.scvx import solve_scvx`).
