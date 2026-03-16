import numpy as np
import pytest
from napari_rf.features import FeatureCreator
from napari_rf.RF import RF

def test_feature_creator_2d():
    creator = FeatureCreator()
    img = np.random.random((64, 64))
    
    gen = creator.make_simple_features(img)
    
    features = None
    for item in gen:
        if isinstance(item, np.ndarray):
            features = item
            
    assert features is not None
    assert features.shape[:2] == (64, 64)
    assert features.shape[-1] > 10

def test_feature_creator_3d():
    creator = FeatureCreator()
    img = np.random.random((3, 64, 64))
    
    gen = creator.make_simple_features(img, indices=[0, 2])
    
    features = None
    for item in gen:
        if isinstance(item, np.ndarray):
            features = item
            
    assert features is not None
    assert features.shape == (2, 64, 64, features.shape[-1])

def test_rf_train_predict():
    rf = RF()
    
    features = np.random.random((32, 32, 15))
    labels = np.zeros((32, 32), dtype=int)
    # Give some labels for class 1 and 2
    labels[0:5, 0:5] = 1
    labels[10:15, 10:15] = 2
    
    prediction = rf.train(labels, features)
    
    # In scikit-image fit_segmenter, if we pass labels with 0, 1, 2
    # it might only return probabilities for labels > 0 if it uses 0 as unlabeled/background
    # let"s see what classes it found
    print(f"Classes found: {rf.clf.classes_}")
    
    assert prediction.ndim == 3
    assert prediction.shape[0] == len(rf.clf.classes_)
    assert prediction.shape[1:] == (32, 32)

def test_rf_not_fitted():
    rf = RF()
    features = np.random.random((32, 32, 15))
    
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        rf.predict_segmenter(features)
