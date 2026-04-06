"""Extended pipeline orchestrator with DSE query detection defense.

Runs the full pipeline (train -> generate -> evaluate -> AT -> re-evaluate)
plus DSE training and DSE-aware evaluation at each stage.
"""

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_MODBUS = os.path.dirname(_CLI)
sys.path.insert(0, _CLI)
sys.path.insert(0, _MODBUS)

from utils.logging import setup_logging, get_logger
from utils.constants import DL_TARGETS, TREE_TARGETS, WB_ATTACKS, BB_ATTACKS

logger = get_logger(__name__)


def run(cmd, description):
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP: {description}")
    logger.info(f"CMD:  {' '.join(cmd)}")
    logger.info(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=_HERE)
    if result.returncode != 0:
        logger.error(f"FAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run full adversarial pipeline with DSE defense"
    )
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip baseline training")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip adversarial generation")
    parser.add_argument("--skip-at", action="store_true",
                        help="Skip adversarial training")
    parser.add_argument("--skip-dse-train", action="store_true",
                        help="Skip DSE encoder training (reuse existing)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    py = sys.executable
    common = ["--device", args.device, "--log-level", args.log_level]

    # Step 0: Train baselines
    if not args.skip_train:
        run([py, "0_train.py", "--model", "all"] + common,
            "Train baseline models (XGB, CAT, RF, ET, LGBM, FTT)")

    # Step 1: Generate adversarial
    if not args.skip_gen:
        run([py, "1_generate_adv.py",
             "--target"] + DL_TARGETS +
            ["--attack"] + WB_ATTACKS +
            ["--source", "train"] + common,
            "Generate DL adversarial (train, WB scaled-space)")

        run([py, "1_generate_adv.py",
             "--target"] + DL_TARGETS +
            ["--attack"] + WB_ATTACKS +
            ["--source", "test"] + common,
            "Generate DL adversarial (test, WB scaled-space)")

        run([py, "1_generate_adv.py",
             "--target"] + TREE_TARGETS +
            ["--attack"] + BB_ATTACKS +
            ["--source", "train"] + common,
            "Generate ML adversarial (train, BB direct)")

        run([py, "1_generate_adv.py",
             "--target"] + TREE_TARGETS +
            ["--attack"] + BB_ATTACKS +
            ["--source", "test"] + common,
            "Generate ML adversarial (test, BB direct)")

    # Step 2: Evaluate BEFORE AT (no defense)
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION BEFORE ADVERSARIAL TRAINING")
    logger.info("=" * 70)
    run([py, "2_evaluate.py", "--target", "all"] + common,
        "Evaluate all models BEFORE AT")

    run([py, "4_evaluate_ensemble.py"] + common,
        "Evaluate ensemble BEFORE AT")

    # Step 2.5: Train DSE encoder
    if not args.skip_dse_train:
        run([py, "5_train_dse.py"] + common,
            "Train DSE encoder")

    # Step 3: Evaluate BEFORE AT + DSE
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION BEFORE AT + DSE DEFENSE")
    logger.info("=" * 70)
    run([py, "6_evaluate_dse.py", "--target", "all"] + common,
        "Evaluate all models BEFORE AT with DSE defense")

    # Step 4: Adversarial training
    if not args.skip_at:
        run([py, "3_adv_train.py", "--model", "all"] + common,
            "Adversarial training (all models)")

    # Step 5: Evaluate AFTER AT (no defense)
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION AFTER ADVERSARIAL TRAINING")
    logger.info("=" * 70)
    run([py, "2_evaluate.py", "--target", "all", "--at", "true"] + common,
        "Evaluate all models AFTER AT")

    run([py, "4_evaluate_ensemble.py", "--at", "true"] + common,
        "Evaluate ensemble AFTER AT")

    # Step 5.5: Evaluate AFTER AT + DSE
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION AFTER AT + DSE DEFENSE")
    logger.info("=" * 70)
    run([py, "6_evaluate_dse.py", "--target", "all", "--at", "true"] + common,
        "Evaluate all models AFTER AT with DSE defense")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE (with DSE)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
