# Run disclosure — versioned pipeline runs

Per the plan's disclosure rules (Stage 0 and Stage 10 decision rules, Stage 11
risks/fallbacks), every pipeline version is recorded here.

## v1 — primary run, G3 off

- Config: `config.yaml` (`run_id: v1`, `enable_g3: false`).
- Completed: 2026-10-29. All stages 1–11 executed (Stage 11 recorded as
  not-run, `artifacts/g3_record.json`).
- Outputs: `artifacts/`, `tables/`, `figures/`, `models/`.
- Model freeze: `artifacts/model_freeze.json`, signed before internal-test
  outcomes were loaded.

## v2 — versioned rerun with G3 enabled

- Config: `config_g3.yaml` (`run_id: v2_g3`, `enable_g3: true`).
- Committed before re-splitting: the G3 photo pools are reserved in Stage 3
  **before** any G2 partitioning, exactly per the Stage 11 reservation rule.
- Fully isolated outputs: `artifacts_g3/`, `tables_g3/`, `figures_g3/`,
  `models_g3/`. No v1 file is read or overwritten by v2 stages (dataset files
  are read-only inputs to both).
- The v2 split differs from v1 (reserved photos excluded from the G2 pool), so
  v2 contains its own complete G2 primary evaluation. G3 results are a
  separate claim (appendix A6); no reserved G3 outcome is used in G2
  development.
- Choice of which G2 table appears in the main report is a reporting decision
  to be made from the frozen artifacts, not from outcome comparison; both
  runs' artifacts are retained for audit.

## Note on earlier exploratory work

Any outcome-driven exploration of the labeled data prior to this pipeline
(e.g., the course notebooks) means the internal evaluation is a held-out
evaluation but not a pristine prospective confirmation; changing the split
seed does not restore independence from past decisions.
