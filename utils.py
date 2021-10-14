"""
utils.py

Created: 04/23/2021
Modified: 05/05/2021 By: Sam

This file provides utilities for quick access to dataset modifications. It:
    1. Separates data into training and test sets.
    2. Creates multiple, alternate, input data layers from the image dataset by:
        a. Padding all images to match a dataset maximum
        b. Separating color images from black-and-white images
        c. Differentiating between faces only and entire images
"""

import os
import tensorflow as tf
from tensorflow import keras
import xml.etree.ElementTree as ET
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,recall_score, classification_report

def folder_to_tensors(path):
    """Converts all png files in a given folder to a 4-d (# files, 3 channels) tensor."""
    images = list(sorted(os.listdir(path)))
    tensors = []
    for image in images:
        ext = image[-4:]
        with open(path+'/'+image, 'rb') as im:
            if ext == '.png':
                tensors.append(
                    tf.io.decode_png(im.read(), channels=3)
                )
            elif ext == '.jpg' or ext == 'jpeg':
                tensors.append(
                    tf.io.decode_jpeg(im.read(), channels=3)
                )
            elif ext == '.gif':
                tensors.append(
                    tf.io.decode_jpeg(im.read(), channels=3)
                )

    return tensors


def largest_image(tensor_list):
    """This finds and returns the (tuple) dimensions of the largest image in the list."""
    largest = (0, 0)
    for tens in tensor_list:
        if tens.shape[0] * tens.shape[1] > largest[0] * largest[1]:
            largest = tens.shape
    return largest


def get_padded_images(path, resize=True, unstack=False):
    """This function gets a folder of .png images, pads them to the largest image's
    dimensions, and returns a tensor for nn input. If resize is true, the images
    will be interpolated with the default arguments (bilinear, no antialiasing)."""
    images = folder_to_tensors(path)
    dims = largest_image(images)
    if resize:
        for k, image in enumerate(images):
            images[k] = tf.image.resize_with_pad(image, dims[0], dims[1])
    else:
        # These are simply 0-padded to largest image's size.
        for k, image in enumerate(images):
            image[k] = tf.image.pad_to_bounding_box(image, dims[0], dims[1])
    return tf.stack(images)


def get_image_labels(path, faces=False):
    """Exports labels from a folder full of xml files and returns them as a vector."""
    xml_files = list(sorted(os.listdir(path)))
    labels = []
    for file in xml_files:
        tree = ET.parse(path + '/' + file)
        root = tree.getroot()
        if faces:
            for name in root.iter('name'):
                if name.text == 'with_mask':
                    labels.append(1)
                elif name.text == 'mask_weared_incorrect':
                    labels.append(2)
                else:
                    labels.append(0)
        else:
            c = 0;
            for name in root.iter('name'):
                if name.text == 'with_mask':
                    c += 1
            labels.append(c)
    return labels


def get_images_and_labels(faces=False):
    if faces:
        results = get_padded_images('./faces')
        labels = tf.constant(get_image_labels('./archive/annotations', faces=faces), dtype='float32')
    else:
        results = get_padded_images('./archive/images')
        labels = tf.constant(get_image_labels('./archive/annotations'), dtype='float32')
    return results, labels


def train_and_test(images, labels, test_size=0.2):
    """This function splits images and labels supplied as arguments into test and training sets
    then returns them as two 2-tuples of training images and labels respectively"""
    # This is courtesy of https://stackoverflow.com/questions/19485641/
    # python-random-sample-of-two-arrays-but-matching-indices
    idx = np.random.choice(np.arange(len(images)), int(test_size * len(images)), replace=False)
    # This one is from https://stackoverflow.com/questions/25330959/
    # how-to-select-inverse-of-indexes-of-a-numpy-array
    mask = np.ones(len(images), np.bool)
    mask[idx] = 0

    test_ims = np.delete(images, mask, axis=0)
    train_ims = np.delete(images, idx, axis=0)
    test_labels = np.delete(labels, mask, axis=0)
    train_labels = np.delete(labels, idx, axis=0)
    return (train_ims, train_labels), (test_ims, test_labels)


def load_and_predict(model_dir, image_dir, k_pre=False):
    """This function loads a model from a directory path and images from an image path
    then feeds the image to the previously saved model."""
    # Uses keras preprocessing
    if k_pre:
        data = tf.keras.preprocessing.image_dataset_from_directory(image_dir, seed=123)
        y = np.concatenate([y for x,y in data], axis=0)
    # Doesn't use keras preprocessing
    else:
        data = get_padded_images(image_dir)
    # Load model from folder
    model = keras.models.load_model(model_dir)

    if k_pre:
        results = np.array([])
        for x, z in data:
            results = np.concatenate([results, np.argmax(model.predict(x), axis=-1)])
        return results, y
    else:
        return model.predict(data)


predictions, labels = load_and_predict('./saved_CNN_B/saved CNN_B', './cv_gen_foldered', k_pre=True)
print(predictions)
print(labels)
print(tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy())
print('\n')
print(classification_report(labels, predictions))
