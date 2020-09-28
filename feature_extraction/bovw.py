import pickle
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

from utils.plotting import showInRow


class BoVW:
    """
    Bag of visual words model
    """

    def __init__(self, extractor, train_images=None, cluster_model=None, scaler=None,
                 vocabulary_size=600):
        """
        Initialise model and possibly train on data
        Args:
            extractor: model of decsriptors extraction (SIFT for example)
            train_images: array of images to train cluster model, optional
            cluster_model: filename of cluster model weights, optional
            vocabulary_size: size of visual words vocabulary
        """
        self.extractor = extractor
        self.vocabulary_size = vocabulary_size
        self.train_images = train_images
        self.cluster_model = None
        if cluster_model is not None:
            print("Loading cluster model from file..")
            self.cluster_model = pickle.load(open(cluster_model, 'rb'))
        if scaler is None:
            self.scaler = StandardScaler()
        else:
            print("Loading scaler from file..")
            self.scaler = pickle.load(open(scaler, 'rb'))
        self.train_histograms = None

        # Train models if train data is available
        if self.train_images is not None:
            print("Training BoVW starts ...")
            print("Extracting descriptors ...")
            train_desc = self._get_images_descriptors(self.train_images)
            if cluster_model is None:
                print("No cluster model provided, training new...")
                print("\tGrouping descriptors ...")
                all_descriptors_list = self._get_descriptors_list(train_desc)
                print("\tDescriptors extracted from tarin images")
                print("\tTraining cluster model ...")
                self._train_cluster_model(all_descriptors_list, vocabulary_size)
                print("\tCluster model trained")
            else:
                print('Using provided cluster model.')
            print("Computing histograms ...")
            train_histograms = self._get_histograms(train_desc)
            print("Histograms computed")
            print("Normalizing histograms ...")
            self.scaler.fit(train_histograms)
            self.train_histograms = self.scaler.transform(train_histograms)
            print("Histograms normalized")

    def _get_image_descriptors(self, image):
        """
        Extract image descriptors from image
        Args:
            image: image to extract descriptors from
        Return:
            keypoints, descriptors extracted from image
        """
        keypoints, descriptors = self.extractor.detectAndCompute(image, None)
        if len(keypoints) < 1 or descriptors is None:
            descriptors = np.zeros((1, self.extractor.descriptorSize()),
                                   np.float32)
        return keypoints, descriptors

    def _get_images_descriptors(self, images):
        """
        Extract image descriptors from array of images
        Args:
            images: array of images to extract descriptors from
        Returns:
            array of extracted descriptors, 1 item for 1 image
        """
        descriptors = []

        for image in tqdm(images):
            _, img_features = self._get_image_descriptors(image)
            if img_features is not None:
                descriptors.append(img_features)
        return np.array(descriptors)

    def _get_descriptors_list(self, descriptors):
        """
        Stack descriptors in array of shape (n, 128)
        Args:
            descriptors: array of extracted descriptors
        Returns:
            descriptors in array of shape (n, 128)
        """
        stacked_descriptors = np.array(descriptors[0])
        for descriptor in tqdm(descriptors[1:]):
            if descriptor is not None:
                stacked_descriptors = np.vstack(
                    (stacked_descriptors, descriptor))

        return stacked_descriptors

    def _train_cluster_model(self, descriptors, cluster_num=2):
        """
        Train K-Means with extracted descriptors to form vocabulary
        Args:
            descriptors: descriptors in array of shape (n, 128)
            cluster_num: Number of visual words in vocabulary
        """
        self.cluster_model = KMeans(n_clusters=cluster_num, verbose=1,
                                    init='random', n_init=1).fit(descriptors)

    def _get_histograms(self, descriptors):
        """
        Produce vocabulary histograms from list of image descriptors
        Args:
            descriptors: array of descriptors extracted from images
        Returns:
            Vocabulary histograms, 1 per image
        """
        histograms = np.zeros((descriptors.shape[0], self.vocabulary_size))

        for i in tqdm(range(descriptors.shape[0])):
            for j in range(len(descriptors[i])):
                feature = descriptors[i][j]
                feature = feature.reshape(1, -1)
                idx = self.cluster_model.predict(feature)
                histograms[i][idx] += 1
        return histograms

    def get_features(self, dataset):
        print("Extracting descriptors ...")
        descriptors = self._get_images_descriptors(dataset)
        print("Computing histograms ...")
        histograms = self._get_histograms(descriptors)
        print("Normalizing histograms ...")
        features = self.scaler.transform(histograms)
        print("Done")
        return features

    def plot_nearest_neighbours(self, neighbour_ids, distances=None):
        """
        Plot images in row by their ids
        """
        if distances is not None:
            showInRow(self.train_images[neighbour_ids], distances)
        else:
            showInRow(self.train_images[neighbour_ids])

    def get_nearest_neighbours(self, image, n_neighbours=10, display=False):
        """
        Extract nearest neighbours from train dataset for new image with KNN
        Args:
            image: Querry image
            n_neighbours: number of neighbours to retrieve
            display: Bool, indicates if pictures must be plotted
        Returns:
            2 arrays: distances and ids of neighbours
        """

        key_points, descriptor = self._get_image_descriptors(image)
        descriptors = np.array([descriptor])

        histograms = self._get_histograms(descriptors)
        histogram = self.scaler.transform(histograms)[0]

        knn = NearestNeighbors(n_neighbors=n_neighbours)
        knn.fit(self.train_histograms)
        dists, ids = knn.kneighbors([histogram])
        if display:
            showInRow([image, cv2.drawKeypoints(image, key_points, image.copy())],
                      ["Original", "SIFT features"])
            print("Nearest Neighbours with distances:")
            self.plot_nearest_neighbours(ids[0], dists[0])
        return dists, ids

    def save(self, cluster_filename, scaler_filename):
        with open(cluster_filename, 'wb') as weights_file:
            pickle.dump(self.cluster_model, weights_file)
        with open(scaler_filename, 'wb') as weights_file:
            pickle.dump(self.scaler, weights_file)
