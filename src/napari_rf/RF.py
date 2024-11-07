from skimage import future
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
import numpy as np

class RF:

    def __init__(self, clf=None):
        if clf is None:
            self.clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                     max_depth=100, max_samples=0.05)
        else:
            self.clf = clf

    def train(self, training_labels, features):

        # training_labels = training_labels[0,...]

        self.clf = future.fit_segmenter(training_labels, features, self.clf)

        return self.predict_segmenter(features)



    def predict_segmenter(self, features):
        """
        taken from https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/future/trainable_segmentation.py#L89-L118
        Segmentation of images using a pretrained classifier.
        Parameters
        ----------
        features : ndarray
            Array of features, with the last dimension corresponding to the number
            of features, and the other dimensions are compatible with the shape of
            the image to segment, or a flattened image.
        clf : classifier object
            trained classifier object, exposing a ``predict`` method as in
            scikit-learn's API, for example an instance of
            ``RandomForestClassifier`` or ``LogisticRegression`` classifier. The
            classifier must be already trained, for example with
            :func:`skimage.segmentation.fit_segmenter`.
        Returns
        -------
        output : ndarray
            Labeled array, built from the prediction of the classifier.
        """

        clf = self.clf

        sh = features.shape
        if features.ndim > 2:
            features = features.reshape((-1, sh[-1]))

        try:
            predicted_labels = clf.predict_proba(features)
        except NotFittedError:
            raise NotFittedError(
                "You must train the classifier `clf` first"
                "for example with the `fit_segmenter` function."
            )
        except ValueError as err:
            if err.args and 'x must consist of vectors of length' in err.args[0]:
                raise ValueError(
                    err.args[0] + '\n' +
                    "Maybe you did not use the same type of features for training the classifier."
                )
            else:
                raise err

        if len(predicted_labels.shape) == 1:
            output = predicted_labels.reshape(sh[:-1])
        elif len(predicted_labels.shape) == 2:
            feature_dim = predicted_labels.shape[-1]
            s = list(sh[:-1]) + [feature_dim]
            output = predicted_labels.reshape(s)
            output = np.rollaxis(output, 2, 0)
        else:
            raise ValueError('shape mismatch')
        return output
