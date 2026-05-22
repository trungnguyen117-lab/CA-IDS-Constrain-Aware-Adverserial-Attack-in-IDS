"""ART preprocessing defenses — áp lên input tại inference time."""

from __future__ import annotations


def parse_fs_config(spec: str | None, targets: list[str], default_bit: int,
                    clip_values: tuple[float, float]) -> dict:
    """Parse 'rf=2,et=4' → {target: FeatureSqueezing(bit_depth=...)}.

    Targets không xuất hiện trong spec dùng default_bit.
    """
    from art.defences.preprocessor import FeatureSqueezing
    overrides = {}
    if spec:
        for tok in spec.split(","):
            name, _, bit = tok.strip().partition("=")
            overrides[name.strip()] = int(bit)
    return {
        t: FeatureSqueezing(clip_values=clip_values,
                            bit_depth=overrides.get(t, default_bit))
        for t in targets
    }
