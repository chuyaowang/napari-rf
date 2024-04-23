"""
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget
from qtpy.QtWidgets import QFileDialog


import numpy as np

from skimage import future
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from joblib import load, dump

from napari_rf.features import FeatureCreator

if TYPE_CHECKING:
    import napari


class RFWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.clf = None
        self.feature_creator = FeatureCreator()

        self.btn_create_features = QPushButton("create features")
        self.btn_create_features.clicked.connect(self.create_features)

        self.btn_train = QPushButton("train random forest")
        self.btn_train.clicked.connect(self.train)

        self.btn_apply_rf = QPushButton("apply random forest")
        self.btn_apply_rf.clicked.connect(self.apply_rf)
        self.btn_apply_rf.setDisabled(True)

        self.btn_load = QPushButton("load classifier")
        self.btn_load.clicked.connect(self.load)

        self.btn_save = QPushButton("save classifier")
        self.btn_save.clicked.connect(self.save)
        self.btn_save.setDisabled(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.btn_create_features)
        self.layout().addWidget(self.btn_train)
        self.layout().addWidget(self.btn_apply_rf)
        self.layout().addWidget(self.btn_load)
        self.layout().addWidget(self.btn_save)

    def create_features(self):
        img = self.viewer.layers.selection.active.data
        features = self.feature_creator.make_simple_features(img)
        self.viewer.add_image(np.moveaxis(features, -1, 0), name='features')

    def train(self):
        if 'Labels' in self.viewer.layers:
            training_labels = self.viewer.layers['Labels'].data
        else:
            raise Exception('training labels must be in a layer called "Labels"')

        if not 'features' in self.viewer.layers:
            self.create_features()
        features = np.moveaxis( self.viewer.layers['features'].data, 0, -1)
        training_labels = training_labels[0,...]#np.moveaxis(self.viewer.layers['features'].data, 1, -1)

        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                     max_depth=100, max_samples=0.05)

        self.clf = future.fit_segmenter(training_labels, features, clf)
        self.btn_save.setDisabled(False)
        self.btn_apply_rf.setDisabled(False)

        result = self.predict_segmenter(features, self.clf)
        self.viewer.add_image(result, name=f'segmentation probabilities')

    def predict_segmenter(self, features, clf):
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
        sh = features.shape
        if features.ndim > 2:
            features = features.reshape((-1, sh[-1]))

        try:
            predicted_labels = clf.predict_proba(features)
            print(f'{predicted_labels.shape}')
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
        return output

    def apply_rf(self):
        if not 'features' in self.viewer.layers:
            self.create_features()
        features = np.moveaxis(self.viewer.layers['features'].data, 0, -1)
        result = self.predict_segmenter(features, self.clf)
        self.viewer.add_image(result, name=f'segmentation probabilities')

    def load(self):
        source_file = QFileDialog.getOpenFileName(
            self,
            'open classifier',
            '/home/philipp/',
            'classifiers (*.joblib)'
        )
        self.clf = load(source_file[0])
        print(f"loaded {source_file[0]}")

    def save(self):
        save_path = QFileDialog.getSaveFileName(
            self,
            'save file',
            '',
            'all files (*)'
        )
        if self.clf is not None:
            dump(self.clf, save_path[0])