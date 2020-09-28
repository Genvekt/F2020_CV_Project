from tqdm import tqdm
from skimage.feature import hog
import numpy as np


def get_hog_features(dataset, **kwargs):
    """
    Compute hog features for each image in dataset
    Args:
        dataset (array like): Array of images
        **kwargs: key-word arguments for hog descriptor 
    Returns:
        np.ndarray with hog features for each image
    """
    features = []
    for image in tqdm(dataset):
        feature = hog(image, **kwargs)
        features.append(feature)
    return np.array(features)
