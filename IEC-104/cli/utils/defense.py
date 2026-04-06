"""Factory for ART preprocessing defenses."""

import logging

logger = logging.getLogger(__name__)


def create_preprocessing_defense(name, **kwargs):
    """Create an ART preprocessing defense by name.

    Returns a list of Preprocessor instances, or None if name is None.

    Supported defenses:
        - gaussian_noise: GaussianAugmentation(sigma, augmentation=False, apply_predict=True)
        - feature_squeezing: FeatureSqueezing(bit_depth, clip_values)
    """
    if name is None:
        return None

    if name == "gaussian_noise":
        from art.defences.preprocessor import GaussianAugmentation

        sigma = kwargs.get("sigma", 0.01)
        defense = GaussianAugmentation(
            sigma=sigma,
            augmentation=False,
            apply_fit=False,
            apply_predict=True,
        )
        logger.info(f"Created GaussianAugmentation defense (sigma={sigma})")
        return [defense]

    if name == "feature_squeezing":
        from art.defences.preprocessor import FeatureSqueezing

        bit_depth = kwargs.get("bit_depth", 8)
        clip_values = kwargs.get("clip_values", (0.0, 1.0))
        defense = FeatureSqueezing(
            bit_depth=bit_depth,
            clip_values=clip_values,
        )
        logger.info(f"Created FeatureSqueezing defense (bit_depth={bit_depth})")
        return [defense]

    raise ValueError(f"Unknown defense: {name}")
