import numpy as np


def probas_to_labels(probas):
    return np.argmax(probas, axis=1)


def merge_probas(probas_1: np.ndarray, probas_2: np.ndarray,
                 weight_1=0.5, weight_2=0.5):
    return probas_1 * weight_1 + probas_2 * weight_2
