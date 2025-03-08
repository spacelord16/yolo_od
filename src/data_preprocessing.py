import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

data_path = '../data/raw/VOCdevkit/VOC2012'
annotations_path = os.path.join(data_path, 'Annotations')
images_path = os.path.join(data_path, 'JPEGImages')

def preprocess_data(img_size=(224,224)):
    images, labels = [], []

    for anno_file in tqdm(os.listdir(annotations_path)):
        tree = ET.parse(os.path.join(annotations_path, anno_file))
        root = tree.getroot()
        
        img_file = root.find('filename').text
        image = cv2.imread(os.path.join(images_path, img_file))
        image = cv2.resize(image, img_size)

        objects = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            bbox = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text)
            ]
            objects.append({'label': label, 'bbox': bbox})

        images.append(image)
        labels.append(objects)

    os.makedirs('../data/processed', exist_ok=True)
    np.save('../data/processed/images.npy', np.array(images))
    np.save('../data/processed/labels.npy', np.array(labels, dtype=object))
    print("Data preprocessing complete. Numpy files saved.")

if __name__ == "__main__":
    preprocess_data()
