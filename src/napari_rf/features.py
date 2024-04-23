from skimage import feature
from skimage.filters import gaussian, sobel, laplace
from functools import partial
import numpy as np

class FeatureCreator:
    def __init__(self):
        pass

    def make_simple_features(self, img):

        features_func = partial(feature.multiscale_basic_features,
                                intensity=True, edges=True, texture=True, sigma_min=1, sigma_max=16)
        features = features_func(img)

        gaussians = []
        for sigma in [1, 10]:
            gaussians.append(gaussian(img, sigma)[..., np.newaxis])

        sob = sobel(img)[..., np.newaxis]
        log = laplace(gaussian(img, 10))[..., np.newaxis]
        features = np.concatenate([img[..., np.newaxis], features, sob, gaussians[1] - gaussians[0], log], axis=-1)

        return features