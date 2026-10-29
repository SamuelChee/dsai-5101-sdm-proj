"""Single documented entry point for the analysis pipeline.

Usage (from the project root):

    python -m src.run --config config.yaml --stage audit
    python -m src.run --config config.yaml --stage all

Only stages implemented so far are dispatched; later stages are added to
``STAGES`` as the pipeline grows (plan Stage 13).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import (
    audit,
    canonicalize,
    disagreement,
    evaluate_test,
    g3_experiment,
    graphs,
    identity_models,
    learning_curves,
    presentation_figures,
    simulations,
    split,
    visual_models,
)

# Ordered stage registry. "all" runs every stage in order.
STAGES = {
    "audit": audit.run_audit,  # Stage 1 - audit and validate the raw files
    "canonicalize": canonicalize.run_canonicalize,  # Stage 2 - canonical pairs
    "split": split.run_split,  # Stage 3 - reserve experiments, freeze split
    "graphs": graphs.run_graphs,  # Stage 4 - fitting graphs, G2 eligibility
    # Stage 4.5 - structural presentation visuals
    "presentation_figures": presentation_figures.run_presentation_figures,
    "simulations": simulations.run_simulations,  # Stage 5 - estimator + A/B/C
    # Stage 6 - M0, M1, M1P, M2, M2P on fitting/validation
    "identity_models": identity_models.run_identity_models,
    # Stage 7 - development disagreement description
    "disagreement": disagreement.run_disagreement,
    # Stage 8 - image features, M3P/M4P
    "visual_models": visual_models.run_visual_models,
    # Stage 9 - learning curves and controls
    "learning_curves": learning_curves.run_learning_curves,
    # Stage 10 - model freeze and internal-test evaluation
    "evaluate_test": evaluate_test.run_evaluate_test,
    # Stage 11 - optional reserved G3 experiment (flag-gated)
    "g3_experiment": g3_experiment.run_g3,
}

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Midterm analysis pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        default="all",
        choices=[*STAGES, "all"],
        help="Stage to run (default: all implemented stages, in order).",
    )
    args = parser.parse_args(argv)

    selected = list(STAGES) if args.stage == "all" else [args.stage]
    for name in selected:
        print(f"=== stage: {name} ===")
        code = STAGES[name](args.config)
        if code != 0:
            print(f"Stage '{name}' stopped the pipeline (exit {code}).")
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
