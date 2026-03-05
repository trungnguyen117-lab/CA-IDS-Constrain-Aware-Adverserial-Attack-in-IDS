"""Batch-run generate_adv_soict.py over multiple targets × attacks.

Usage:
    # All targets, all attacks
    python run_generate_adv.py --targets xgb cat rf lstm resdnn --attacks zoo hsja pgd fgsm

    # Specific subset
    python run_generate_adv.py --targets lstm resdnn --attacks pgd fgsm

    # Pass through extra args to generate_adv_soict.py
    python run_generate_adv.py --targets xgb --attacks zoo --samples 500 --device cuda
"""

import os
import sys
import argparse
import subprocess

_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _FOAMI)

from utils.paths     import setup_paths
setup_paths()

from utils.logging   import setup_logging, get_logger
from utils.constants import ALL_TARGETS, ALL_ATTACKS

logger = get_logger(__name__)

_SCRIPT = os.path.join(_HERE, 'generate_adv_soict.py')

# Args that belong to this wrapper (not forwarded to generate_adv_soict.py)
_WRAPPER_ARGS = {'targets', 'attacks', 'log_level', 'fail_fast'}


def main():
    parser = argparse.ArgumentParser(
        description="Batch adversarial sample generation across targets × attacks"
    )
    parser.add_argument('--targets', '-t', nargs='+', required=True,
                        choices=ALL_TARGETS,
                        help="Target model(s)")
    parser.add_argument('--attacks', '-a', nargs='+', required=True,
                        choices=ALL_ATTACKS,
                        help="Attack algorithm(s)")
    parser.add_argument('--fail-fast', action='store_true',
                        help="Stop immediately if any combination fails")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    # Forward-only args (passed through to generate_adv_soict.py unchanged)
    parser.add_argument('--data-in')
    parser.add_argument('--models-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--device')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--sampling-mode')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-retries', type=int)
    parser.add_argument('--timeout', type=int)
    parser.add_argument('--placeholder')
    parser.add_argument('--verbose')

    args = parser.parse_args()
    setup_logging(args.log_level)

    # Build forwarded args list
    forwarded = []
    for key, val in vars(args).items():
        if key in _WRAPPER_ARGS or val is None:
            continue
        forwarded += [f'--{key.replace("_", "-")}', str(val)]

    combinations = [(t, a) for t in args.targets for a in args.attacks]
    total   = len(combinations)
    ok      = 0
    failed  = []

    logger.info(f"[+] {total} combination(s): {args.targets} × {args.attacks}")

    for i, (target, attack) in enumerate(combinations, 1):
        cmd = [sys.executable, _SCRIPT,
               '--target', target, '--attack', attack,
               '--log-level', args.log_level] + forwarded

        logger.info(f"[{i}/{total}] target={target}  attack={attack}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            logger.warning(f"  FAILED (exit {result.returncode}): {target} × {attack}")
            failed.append((target, attack))
            if args.fail_fast:
                raise SystemExit(f"Stopped at {target}/{attack} (--fail-fast)")
        else:
            ok += 1

    logger.info(f"[+] Done: {ok}/{total} succeeded, {len(failed)} failed")
    if failed:
        for t, a in failed:
            logger.warning(f"  FAILED: {t} × {a}")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
