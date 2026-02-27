from functools import partial

import numpy as np
from skimage import feature
from skimage.filters import gaussian, laplace, sobel


class FeatureCreator:
    def __init__(self):
        pass

    def make_simple_features(self, *imgs):

        # Configure the base feature extractor
        features_func = partial(
            feature.multiscale_basic_features,
            intensity=True,
            edges=True,
            texture=True,
            sigma_min=1,
            sigma_max=16,
        )
        all_features = []
        for img in imgs:
            # 1. Normalize image (0.5% - 99.5% percentile)
            # This makes the features robust to lighting/exposure differences
            v_min, v_max = np.percentile(img, (0.5, 99.5))
            img_norm = np.clip((img - v_min) / (v_max - v_min + 1e-8), 0, 1)

            # 2. Generate multiscale features (edges, texture) using normalized image
            features = features_func(img_norm)

            gaussians = []
            for sigma in [1, 10]:
                gaussians.append(gaussian(img_norm, sigma)[..., np.newaxis])

            sob = sobel(img_norm)[..., np.newaxis]
            log = laplace(gaussian(img_norm, 10))[..., np.newaxis]

            # 3. Add Local Standard Deviation (Texture/Variance)
            # Calculated as sqrt(E[x^2] - (E[x])^2) using a gaussian window
            sigma_std = 3
            mean_img = gaussian(img_norm, sigma_std)
            sq_mean_img = gaussian(img_norm**2, sigma_std)
            local_std = np.sqrt(np.maximum(sq_mean_img - mean_img**2, 0))[..., np.newaxis]

            all_features.append(
                np.concatenate(
                    [
                        img_norm[..., np.newaxis],
                        features,
                        sob,
                        gaussians[1] - gaussians[0],
                        log,
                        local_std,
                    ],
                    axis=-1,
                )
            )

        return np.concatenate(all_features, axis=-1)
