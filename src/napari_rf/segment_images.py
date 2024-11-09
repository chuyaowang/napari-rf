from skimage import io
from napari_rf.features import FeatureCreator
import numpy as np
from glob import glob
from joblib import load
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def create_composite_features(feature_creator, *imgs):
    features = []
    for img in imgs:
        features.append(feature_creator.make_simple_features(img))

    return np.concatenate(features, axis=-1)
def main(root, feature_creator, clf):
    for file in tqdm(glob(f'{root}/gfp/*.tif')):
        gfp = io.imread(file)
        bf = io.imread(file.replace('gfp', 'bf'))
        rfp = io.imread(file.replace('gfp', 'rfp'))

        features = create_composite_features(feature_creator, bf, gfp, rfp)
        img = clf.predict_segmenter(features)

        seg = np.argmax(img, axis=0)
        io.imsave(file.replace('gfp', 'seg'), seg)

if __name__ == '__main__':
    root = '/media/philipp/Eirini/to_segment'
    clf = load(f"{root}/clf/classifier_bf_clb2gfp_cdc10_rfp")
    feature_creator = FeatureCreator()
    main(root, feature_creator, clf)