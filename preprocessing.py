import csv
import cv2
import random
import skimage as sk
from skimage import exposure


def preprocess(image, shape=None, augment=False):
    res = image.copy()
    if shape is not None:
        res = cv2.resize(res,shape)
    if augment:
        res = brightness(res)
        res = rotate(res)
        res = noise(res)
    return res
    
def brightness(image, low=0.5, high=1.0):
    value = random.uniform(low, high)
    return exposure.adjust_gamma(image, gamma=value,gain=1)

def rotate(image, max_angle=25):
    angle = int(random.uniform(-max_angle, max_angle))
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    return image

def noise(image, probability=0.4):
    if random.random() > probability:
        return sk.util.random_noise(image, mode='s&p', amount=0.02)
    else:
        return image
