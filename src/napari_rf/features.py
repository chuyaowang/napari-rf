import numpy as np
from functools import partial

from skimage import feature
from skimage.filters import gaussian
from skimage.feature import (
    hessian_matrix_det,
    local_binary_pattern,
    shape_index,
)


class FeatureCreator:
    def __init__(self):
        pass

    def make_simple_features(self, *imgs):
        """
        Generator that yields (current_step, total_steps, description) 
        and finally yields the concatenated features array.
        """
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
        
        # Calculate total steps for progress reporting
        # Steps per slice: normalize (1) + 6 feature types = 7
        total_slices = sum(img.shape[0] if img.ndim == 3 else 1 for img in imgs)
        total_steps = total_slices * 7
        current_step = 0

        for img_idx, img in enumerate(imgs):
            # Step 1: Normalize
            v_min, v_max = np.percentile(img, (0.5, 99.5))
            img_norm = np.clip((img - v_min) / (v_max - v_min + 1e-8), 0, 1)

            def _get_slice_features(img_2d, slice_info=""):
                nonlocal current_step
                
                # Helper to yield and increment
                def _report(desc):
                    nonlocal current_step
                    current_step += 1
                    return (current_step, total_steps, f"{slice_info}{desc}")

                # 1. Core multiscale features
                yield _report("Multiscale Basic")
                ms_features = msbf(img_2d)

                # 2. Local standard deviation
                yield _report("Local Std")
                sigma_std = 3
                mean_img = gaussian(img_2d, sigma_std)
                sq_mean_img = gaussian(img_2d**2, sigma_std)
                local_std = np.sqrt(np.maximum(sq_mean_img - mean_img**2, 0))[..., None]

                # 3. DoG response
                yield _report("DoG")
                dog_response = np.stack(
                    [gaussian(img_2d, s2) - gaussian(img_2d, s1)
                     for s1, s2 in [(1, 3), (3, 5), (5, 8)]],
                    axis=-1,
                )

                # 4. Hessian determinant
                yield _report("Hessian")
                hess_small = hessian_matrix_det(img_2d, sigma=1)[..., None]
                hess_large = hessian_matrix_det(img_2d, sigma=3)[..., None]

                # 5. Shape index
                yield _report("Shape Index")
                si = shape_index(img_2d, sigma=2)[..., None]

                # 6. LBP
                yield _report("LBP")
                lbp = local_binary_pattern(
                    (img_2d * 255).astype(np.uint8),
                    P=8,
                    R=1,
                    method="uniform",
                )[..., None]

                return np.concatenate(
                    [
                        img_2d[..., None],
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

            if img_norm.ndim == 3:
                stack_feats = []
                for z in range(img_norm.shape[0]):
                    # Normalize step for this slice (implicit in total_steps)
                    current_step += 1
                    yield (current_step, total_steps, f"Slice {z+1}/{img_norm.shape[0]}: Normalizing")
                    
                    # Feature steps
                    slice_gen = _get_slice_features(img_norm[z], slice_info=f"Slice {z+1}/{img_norm.shape[0]}: ")
                    try:
                        while True:
                            yield next(slice_gen)
                    except StopIteration as e:
                        stack_feats.append(e.value)
                all_features.append(np.stack(stack_feats, axis=0))
            else:
                current_step += 1
                yield (current_step, total_steps, "Normalizing")
                
                slice_gen = _get_slice_features(img_norm)
                try:
                    while True:
                        yield next(slice_gen)
                except StopIteration as e:
                    all_features.append(e.value)

        yield np.concatenate(all_features, axis=-1)
