import numpy as np
from functools import partial

from skimage import feature
from skimage.filters import gaussian, laplace
from skimage.feature import (
    hessian_matrix_det,
    local_binary_pattern,
    shape_index,
)


class FeatureCreator:
    def __init__(self):
        pass

    def make_simple_features(self, *imgs):

        # Multiscale basic features (intensity, edges, texture)
        msbf = partial(
            feature.multiscale_basic_features,
            intensity=True,
            edges=True,
            texture=True,
            sigma_min=1,
            sigma_max=16,
        )

        all_features = []

        for img in imgs:
            # Normalize
            v_min, v_max = np.percentile(img, (0.5, 99.5))
            img_norm = np.clip((img - v_min) / (v_max - v_min + 1e-8), 0, 1)

            # Core multiscale features
            ms_features = msbf(img_norm)

            # Local standard deviation
            sigma_std = 3
            mean_img = gaussian(img_norm, sigma_std)
            sq_mean_img = gaussian(img_norm**2, sigma_std)
            local_std = np.sqrt(np.maximum(sq_mean_img - mean_img**2, 0))[..., None]

            # DoG response at multiple scales
            dog_response = np.stack(
                [gaussian(img_norm, s2) - gaussian(img_norm, s1)
                 for s1, s2 in [(1, 3), (3, 5), (5, 8)]],
                axis=-1,
            )

            # Hessian determinant (blobness)
            hess_small = hessian_matrix_det(img_norm, sigma=1)[..., None]
            hess_large = hessian_matrix_det(img_norm, sigma=3)[..., None]

            # Shape index
            si = shape_index(img_norm, sigma=2)[..., None]

            # LBP (uniform)
            lbp = local_binary_pattern(
                (img_norm * 255).astype(np.uint8),
                P=8,
                R=1,
                method="uniform",
            )[..., None]

            # Concatenate all features
            all_features.append(
                np.concatenate(
                    [
                        img_norm[..., None],
                        ms_features,
                        local_std,
                        dog_response,
                        hess_small,
                        hess_large,
                        si,
                        lbp,
                    ],
                    axis=-1,
                )
            )

        return np.concatenate(all_features, axis=-1)
