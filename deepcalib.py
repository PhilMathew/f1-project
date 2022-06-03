import cv2
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, Flatten, Input
import tensorflow as tf


# Adapted from https://github.com/alexvbogdan/DeepCalib


def load_deepcalib_regressor(weights_file):
    """
    Loads the SingleNet regressor model from the DeepCalib paper
    :param weights_file: Path to saved weights file
    :return: Pre-trained model for predicting focal length and distortion coefficient
    """
    # Using SingleNet regressor
    input_shape = (299, 299, 3)
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    phi_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=main_input, input_shape=input_shape)
    phi_features = phi_model.output
    phi_flattened = Flatten(name='phi-flattened')(phi_features)
    final_output_focal = Dense(1, activation='sigmoid', name='output_focal')(phi_flattened)
    final_output_distortion = Dense(1, activation='sigmoid', name='output_distortion')(phi_flattened)

    for layer in phi_model.layers:
        layer._name = layer._name + "_phi"

    model = Model(inputs=[main_input], outputs=[final_output_focal, final_output_distortion])
    model.load_weights(str(weights_file))

    return model

def load_and_preprocess(img_path, input_size=(299, 299)):
    """
    Preprocesses input image as needed for the network

    :param img_path: Path to image file
    :param input_size: Size of input image for the DeepCalib model, defaults to (299, 299)
    :type input_size: tuple, optional
    """
    image = cv2.imread(img_path)
    image = cv2.resize(image, input_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return preprocess_img(image)


def preprocess_img(image):
    image = image / 255.
    image -= 0.5
    image *= 2.
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    
    return image
