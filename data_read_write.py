import cv2
from pathlib import Path
import numpy as np
import csv
from preprocessing import preprocess


def read_image(img_path:Path):
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_dataset(images_dir:Path, 
                 labels_file=None, 
                 delimiter=',', 
                 file_type="ppm",
                 img_shape=None):
    dataset = []
    labels = []
    
    labels_dict = {}
    with open(labels_file, 'r', ) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            labels_dict[row['Filename']] = row['ClassId']
    
    image_paths = images_dir.glob("*."+file_type)
    for image_path in image_paths:
        image = read_image(image_path)
        image = preprocess(image, shape=img_shape, augment=False)
        dataset.append(image)
        labels.append(labels_dict[image_path.name])
    return np.array(dataset), np.array(labels)
