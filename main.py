import argparse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Network import DetectionModel
from skimage.io import imread


def has_tomatoes(image, model):
    """
    Returns a boolean stating whether or not there are tomatoes present in the image.

    :param image: Matrix containing the image data.
    :param model: tf.keras.Model object containing the loaded model.

    :returns: Returns a boolean stating whether or not there are tomatoes present in the image.
    """
    result = model.predict(np.expand_dims(image))

    plt.imshow(image.astype(np.uint8))
    plt.axis('off')
    plt.title("Predicted: " + str(np.round(np.squeeze(result))))


if __name__ == '__main__':
    # Construct the argument parser and parse the hyperparameters.
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_image", required=True, help="Path to a test image.")
    ap.add_argument("-w", "--checkpoint_path", required=True, help="Path to the model checkpoint to be loaded.")

    args = vars(ap.parse_args())

    model = DetectionModel(input_shape=(600, 600, 3)).get_model()

    model.load_weights(args["checkpoint_path"])

    image = imread(args["input_image"])
    has_tomatoes(image, model)
