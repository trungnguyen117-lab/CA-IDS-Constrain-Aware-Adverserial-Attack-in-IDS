"""Kịch bản tấn công nhắm vào các feature quan trọng nhất của dataset.

Ý tưởng:
  1. Tính feature importance (từ model hoặc gradient saliency)
  2. Chọn top-k feature quan trọng nhất làm mục tiêu tấn công
  3. Freeze TẤT CẢ feature còn lại (không cho perturb)
  4. Chạy attack bình thường — chỉ top-k feature bị nhiễu

Điều này thực tế hơn so với tấn công toàn bộ feature vì:
  - Kẻ tấn công thường chỉ kiểm soát được một số chiều nhất định
  - Tập trung nhiễu vào feature quan trọng → bypass detection hiệu quả hơn

Cách tính importance:
  - 'model'    : dùng feature_importances_ của tree model (xgb/rf/cat)
  - 'gradient' : dùng gradient saliency của DL model (resdnn/lstm)
  - 'shap'     : dùng SHAP TreeExplainer (chỉ cho tree model)

Usage:
    # DL target, gradient-based importance, top-10 features
    python generate_adv_feature_targeted.py \\
        --target resdnn --attack pgd --top-k 10 --importance-source gradient

    # Tree target, model importance, top-15 features
    python generate_adv_feature_targeted.py \\
        --target xgb --attack zoo --top-k 15 --importance-source model \\
        --importance-model xgb

    # SHAP importance
    python generate_adv_feature_targeted.py \\
        --target cat --attack zoo --top-k 10 --importance-source shap \\
        --importance-model cat
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

_HERE  = os.path.dirname(os.path.realpath(__file__))
_FOAMI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _FOAMI)

from utils.paths     import setup_paths, MODELS_DIR, ADV_DIR, TEST_CSV
setup_paths()

from utils.logging   import setup_logging, get_logger
from utils.constants import (
    ALL_TARGETS, ALL_ATTACKS, BLACKBOX_ATTACKS,
    GBT_TARGETS, ENSEMBLE_TARGETS, SINGLE_TARGETS,
    DEFAULT_ENSEMBLE_WEIGHTS, DEFAULT_MI_W_GBT_BASE, DEFAULT_MI_PARAMS,
    MODEL_FILENAMES,
)
from utils.loaders import load_wrapper, load_features_csv
from utils.attacks import build_meta, make_generator

logger = get_logger(__name__)

IMPORTANCE_SOURCES = ['model', 'gradient', 'shap']


# ── Feature importance computation ────────────────────────────────────────────

def get_importance_model(importance_model: str, models_dir: str,
                         clip_values, num_classes: int,
                         input_dim: int, device: str):
    """Load model dùng để tính feature importance (có thể khác attack target)."""
    return load_wrapper(importance_model, models_dir,
                        clip_values, num_classes, input_dim, device)


def compute_importance_from_model(wrapper, feature_names: list) -> np.ndarray:
    """Dùng built-in feature_importances_ của tree model."""
    model = wrapper.model
    if hasattr(model, 'feature_importances_'):
        imp = np.array(model.feature_importances_, dtype=np.float64)
        logger.info(f"[importance] Using model.feature_importances_ "
                    f"(shape={imp.shape})")
        return imp
    raise ValueError(
        f"Model {type(model).__name__} không có feature_importances_. "
        "Dùng --importance-source gradient hoặc shap."
    )


def compute_importance_from_gradient(wrapper, X: np.ndarray,
                                      feature_names: list) -> np.ndarray:
    """Gradient saliency: |∂loss/∂x| trung bình trên tất cả samples.

    Dùng ART classifier.class_gradient() — chỉ hoạt động với DL model
    (resdnn, lstm) đã được wrap bởi PyTorchClassifier.
    """
    estimator = wrapper.get_estimator()
    if not hasattr(estimator, 'class_gradient'):
        raise ValueError(
            "Estimator không hỗ trợ class_gradient(). "
            "Dùng --importance-source model hoặc shap."
        )

    logger.info(f"[importance] Computing gradient saliency on {len(X)} samples ...")
    # class_gradient trả về (n_samples, n_classes, n_features)
    grads = estimator.class_gradient(X.astype(np.float32))  # (N, C, D)
    # Lấy mean |gradient| qua tất cả samples và classes
    imp = np.mean(np.abs(grads), axis=(0, 1))               # (D,)
    logger.info(f"[importance] Gradient saliency computed (shape={imp.shape})")
    return imp


def compute_importance_from_shap(wrapper, X: np.ndarray,
                                  feature_names: list,
                                  max_background: int = 200) -> np.ndarray:
    """SHAP TreeExplainer — chỉ dùng cho tree models (xgb, rf, cat)."""
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP chưa được cài. Chạy: pip install shap")

    model = wrapper.model
    background = X[:max_background]
    logger.info(f"[importance] Running SHAP TreeExplainer "
                f"(background={len(background)} samples) ...")
    explainer  = shap.TreeExplainer(model, background)
    shap_vals  = explainer.shap_values(X)                   # (N, D) hoặc list

    if isinstance(shap_vals, list):
        # multi-class: list of (N, D), lấy mean |shap| qua classes
        imp = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
    else:
        imp = np.abs(shap_vals).mean(axis=0)

    logger.info(f"[importance] SHAP computed (shape={imp.shape})")
    return imp


def get_top_k_indices(importance: np.ndarray, k: int,
                       feature_names: list) -> list:
    """Trả về indices của top-k feature quan trọng nhất."""
    top_k = np.argsort(importance)[::-1][:k].tolist()
    logger.info(f"[+] Top-{k} features (by importance):")
    for rank, idx in enumerate(top_k, 1):
        logger.info(f"    {rank:2d}. [{idx:3d}] {feature_names[idx]:<40s} "
                    f"importance={importance[idx]:.6f}")
    return top_k


def compute_freeze_indices(all_indices: list, top_k_indices: list,
                            binary_indices: list) -> list:
    """Tính freeze_indices = all - top_k, luôn giữ nguyên binary features."""
    top_k_set    = set(top_k_indices)
    binary_set   = set(binary_indices)
    # Freeze: không nằm trong top-k; binary luôn freeze dù có trong top-k
    freeze = [i for i in all_indices
              if i not in top_k_set or i in binary_set]
    perturb_count = len(all_indices) - len(freeze)
    logger.info(f"[+] Feature budget: {perturb_count}/{len(all_indices)} features "
                f"có thể bị perturb (loại trừ {len(binary_indices)} binary)")
    return freeze


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feature-targeted adversarial attack for SOICT25"
    )
    # Core
    parser.add_argument('--target', '-t', required=True, choices=ALL_TARGETS)
    parser.add_argument('--attack', '-a', required=True, choices=ALL_ATTACKS)
    parser.add_argument('--top-k',  '-k', type=int, default=10,
                        help="Số feature quan trọng nhất để tấn công (default: 10)")

    # Importance source
    parser.add_argument('--importance-source', default='gradient',
                        choices=IMPORTANCE_SOURCES,
                        help="Cách tính feature importance (default: gradient)")
    parser.add_argument('--importance-model', default=None,
                        choices=['xgb', 'cat', 'rf', 'lstm', 'resdnn'],
                        help="Model dùng để tính importance "
                             "(mặc định dùng chính --target nếu là single model)")

    # Data / paths
    parser.add_argument('--data-in',   '-i', default=TEST_CSV)
    parser.add_argument('--models-dir',       default=MODELS_DIR)
    parser.add_argument('--output-dir',       default=None)
    parser.add_argument('--device',    '-d',  default='cpu',
                        choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--samples',   type=int, default=-1,
                        help="Giới hạn số samples (-1 = tất cả)")
    parser.add_argument('--sampling-mode', default='random',
                        choices=['sequential', 'random'])

    # Attack tuning
    parser.add_argument('--attack-params', type=str, default=None,
                        help="JSON dict ghi đè attack params")

    # Ensemble / MI (nếu target là ensemble/mi)
    parser.add_argument('--ensemble-weights', type=str, default=None)
    parser.add_argument('--mi-params',        type=str, default=None)

    # SHAP tuning
    parser.add_argument('--shap-background', type=int, default=200,
                        help="Số samples background cho SHAP (default: 200)")

    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()
    setup_logging(args.log_level)

    # ── Validate ──────────────────────────────────────────────────────────────
    if args.target in GBT_TARGETS and args.attack not in BLACKBOX_ATTACKS:
        raise SystemExit(
            f"Tree model '{args.target}' chỉ hỗ trợ black-box attacks: "
            f"{BLACKBOX_ATTACKS}"
        )
    if args.target in ENSEMBLE_TARGETS and args.attack not in BLACKBOX_ATTACKS:
        raise SystemExit(
            f"Ensemble/MI targets yêu cầu black-box attacks: {BLACKBOX_ATTACKS}"
        )
    if args.importance_source == 'gradient' and args.target in GBT_TARGETS:
        logger.warning(
            "gradient importance không hỗ trợ trực tiếp cho tree target. "
            "Dùng --importance-model resdnn hoặc đổi sang --importance-source model"
        )
    if args.importance_source in ('model', 'shap') and \
            (args.importance_model or args.target) not in ['xgb', 'cat', 'rf']:
        imp_src = args.importance_model or args.target
        if imp_src not in ['xgb', 'cat', 'rf']:
            logger.warning(
                f"'{args.importance_source}' importance hoạt động tốt nhất với "
                "tree models (xgb/cat/rf). Kết quả có thể không như kỳ vọng."
            )

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.data_in):
        raise SystemExit(f"Data file not found: {args.data_in}")

    logger.info(f"[+] Loading data: {args.data_in}")
    df_in = pd.read_csv(args.data_in, low_memory=False)
    logger.info(f"[+] Input shape: {df_in.shape}")

    meta      = build_meta(df_in, label_col='Label')
    feat_cols = meta['feature_names']
    X_all     = df_in[feat_cols].values.astype(np.float32)
    y_all     = df_in['Label'].values.astype(np.int64)

    # Optional sampling
    if args.samples > 0:
        n   = min(args.samples, len(X_all))
        idx = (np.random.permutation(len(X_all))[:n]
               if args.sampling_mode == 'random' else np.arange(n))
        X_all = X_all[idx]
        y_all = y_all[idx]
        logger.info(f"[+] Sampling {n} ({args.sampling_mode}) samples")

    num_classes = len(meta['class_names'])
    input_dim   = X_all.shape[1]
    clip_values = meta['clip_values']
    all_indices = list(range(input_dim))

    # ── Tính feature importance ───────────────────────────────────────────────
    imp_model_name = args.importance_model or (
        args.target if args.target in SINGLE_TARGETS else 'xgb'
    )
    logger.info(f"[+] Tính feature importance "
                f"(source={args.importance_source}, model={imp_model_name})")

    imp_wrapper = load_wrapper(imp_model_name, args.models_dir,
                               clip_values, num_classes, input_dim, args.device)

    if args.importance_source == 'model':
        importance = compute_importance_from_model(imp_wrapper, feat_cols)
    elif args.importance_source == 'gradient':
        importance = compute_importance_from_gradient(imp_wrapper, X_all, feat_cols)
    else:  # shap
        importance = compute_importance_from_shap(
            imp_wrapper, X_all, feat_cols, args.shap_background
        )

    top_k_indices = get_top_k_indices(importance, args.top_k, feat_cols)

    # Binary features không được perturb dù có trong top-k
    freeze_indices = compute_freeze_indices(
        all_indices, top_k_indices, meta['binary_feature_indices']
    )

    # ── Build attack estimator ─────────────────────────────────────────────────
    logger.info(f"[+] Building attack estimator: {args.target}")

    if args.target in SINGLE_TARGETS:
        estimator = load_wrapper(
            args.target, args.models_dir,
            clip_values, num_classes, input_dim, args.device
        ).get_estimator()

    elif args.target == 'ensemble':
        from art_classifier.ensemble_classifier import EnsembleEstimator
        ew = DEFAULT_ENSEMBLE_WEIGHTS.copy()
        if args.ensemble_weights:
            ew.update(json.loads(args.ensemble_weights))
        wrappers = {
            t: load_wrapper(t, args.models_dir, clip_values,
                            num_classes, input_dim, args.device)
            for t in SINGLE_TARGETS if ew.get(t, 0.0) > 0
        }
        estimator = EnsembleEstimator(
            wrappers=wrappers, weights=ew,
            num_classes=num_classes, clip_values=clip_values
        )

    elif args.target == 'mi':
        from art_classifier.mi_classifier import MIEstimator
        mi_cfg     = DEFAULT_MI_PARAMS.copy()
        w_gbt_base = DEFAULT_MI_W_GBT_BASE.copy()
        if args.mi_params:
            parsed = json.loads(args.mi_params)
            mi_cfg.update({k: v for k, v in parsed.items() if k != 'w_gbt_base'})
            if 'w_gbt_base' in parsed:
                w_gbt_base = np.array(parsed['w_gbt_base'], dtype=np.float64)
        gbt = {k: load_wrapper(k, args.models_dir, clip_values,
                               num_classes, input_dim, args.device)
               for k in ('cat', 'rf')}
        dl  = {k: load_wrapper(k, args.models_dir, clip_values,
                               num_classes, input_dim, args.device)
               for k in ('lstm', 'resdnn')}
        estimator = MIEstimator(
            gbt_wrappers=gbt, dl_wrappers=dl,
            num_classes=num_classes, clip_values=clip_values,
            w_gbt_base=w_gbt_base, **mi_cfg
        )

    # ── Generate ──────────────────────────────────────────────────────────────
    logger.info(f"[+] Attack: {args.attack} | top-k={args.top_k} | "
                f"frozen={len(freeze_indices)}/{input_dim} features")

    attack_params = json.loads(args.attack_params) if args.attack_params else {}
    generator     = make_generator(args.attack, estimator, attack_params)

    df_adv = generator.generate(
        X_all, y_all,
        input_metadata=meta,
        mutate_indices=freeze_indices,   # freeze = NOT top-k
    )

    # ── Ghi kết quả ──────────────────────────────────────────────────────────
    if args.output_dir is None:
        args.output_dir = os.path.join(ADV_DIR, f"{args.target}_feature_targeted")

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(
        args.output_dir,
        f"{args.target}_{args.attack}_top{args.top_k}_{args.importance_source}.csv"
    )
    df_adv.to_csv(out_csv, index=False)

    # Ghi thêm importance report
    imp_df = pd.DataFrame({
        'feature':    feat_cols,
        'importance': importance,
        'rank':       np.argsort(np.argsort(importance)[::-1]) + 1,
        'targeted':   [i in set(top_k_indices) for i in range(input_dim)],
    }).sort_values('rank')
    imp_csv = out_csv.replace('.csv', '_importance.csv')
    imp_df.to_csv(imp_csv, index=False)

    logger.info(f"[+] Label distribution:\n"
                f"{df_adv['Label'].value_counts().to_string()}")
    logger.info(f"[+] Adversarial samples saved: {out_csv}")
    logger.info(f"[+] Importance report saved  : {imp_csv}")


if __name__ == '__main__':
    main()
