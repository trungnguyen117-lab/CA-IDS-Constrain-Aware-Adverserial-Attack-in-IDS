"""End-to-end pipeline orchestrator: train → generate → evaluate → AT → re-evaluate."""

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI = os.path.dirname(_HERE)
_IEC = os.path.dirname(_CLI)
sys.path.insert(0, _IEC)
sys.path.insert(0, os.path.join(_IEC, "script"))
sys.path.insert(0, _CLI)

from utils.logging import setup_logging, get_logger
from utils.constants import DL_TARGETS, TREE_TARGETS, WB_ATTACKS, BB_ATTACKS

logger = get_logger(__name__)


def run(cmd, description):
    """Run a pipeline step as subprocess."""
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP: {description}")
    logger.info(f"CMD:  {' '.join(cmd)}")
    logger.info(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=_HERE)
    if result.returncode != 0:
        logger.error(f"FAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run full adversarial pipeline")
    parser.add_argument("--device", "-d", default="cpu",
                        choices=["cpu", "cuda", "auto", "mps"])
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip baseline training (reuse existing models)")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip adversarial generation (reuse existing CSVs)")
    parser.add_argument("--skip-at", action="store_true",
                        help="Skip adversarial training (reuse existing AT models)")
    parser.add_argument("--version", "-V", default="v1",
                        help="Version tag for adv samples/models (default: v1)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    py = sys.executable
    common = ["--device", args.device, "--log-level", args.log_level]
    if args.version != "v1":
        common += ["--version", args.version]
    
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    # Step 1: Train baselines
    if not args.skip_train:
        run([py, "0_train.py", "--model", "all"] + common,
            "Train baseline models (CAT, RF, LSTM, ResDNN)")

    # Step 2: Generate DL adversarial (WB scaled-space)
    if not args.skip_gen:
        wb = " ".join(WB_ATTACKS)
        dl = " ".join(DL_TARGETS)

        # DL: train source (for AT)
        run([py, "1_generate_adv.py",
             "--target"] + DL_TARGETS +
            ["--attack"] + WB_ATTACKS +
            ["--source", "train"] + common,
            "Generate DL adversarial (train, WB scaled-space)")

        # DL: test source (for eval)
        run([py, "1_generate_adv.py",
             "--target"] + DL_TARGETS +
            ["--attack"] + WB_ATTACKS +
            ["--source", "test"] + common,
            "Generate DL adversarial (test, WB scaled-space)")

        # ML: direct BB — train source
        run([py, "1_generate_adv.py",
             "--target"] + TREE_TARGETS +
            ["--attack"] + BB_ATTACKS +
            ["--source", "train"] + common,
            "Generate ML adversarial (train, BB direct)")

        # ML: direct BB — test source
        run([py, "1_generate_adv.py",
             "--target"] + TREE_TARGETS +
            ["--attack"] + BB_ATTACKS +
            ["--source", "test"] + common,
            "Generate ML adversarial (test, BB direct)")

    # Step 3: Evaluate BEFORE AT (individual)
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION BEFORE ADVERSARIAL TRAINING")
    logger.info("=" * 70)
    run([py, "2_evaluate.py", "--target", "all"] + common,
        "Evaluate all models BEFORE AT")

    # Step 3b: Evaluate BEFORE AT (ensemble)
    run([py, "4_evaluate_ensemble.py"] + common,
        "Evaluate ensemble BEFORE AT")

    # Step 4: Adversarial training
    if not args.skip_at:
        run([py, "3_adv_train.py", "--model", "all"] + common,
            "Adversarial training (all models)")

    # Step 5: Evaluate AFTER AT (individual)
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION AFTER ADVERSARIAL TRAINING")
    logger.info("=" * 70)
    run([py, "2_evaluate.py", "--target", "all", "--at"] + common,
        "Evaluate all models AFTER AT")

    # Step 5b: Evaluate AFTER AT (ensemble)
    run([py, "4_evaluate_ensemble.py", "--at", "true"] + common,
        "Evaluate ensemble AFTER AT")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
