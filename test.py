import argparse
import numpy as np
import matplotlib.pyplot as plt
from Network import DetectionModel
from config import *
from utils import decode_image


def has_tomatoes(image, model):
    """
    Returns a boolean stating whether or not there are tomatoes present in the image.

    :param image: Matrix containing the image data.
    :param model: tf.keras.Model object containing the loaded model.

    :returns: Returns a boolean stating whether or not there are tomatoes present in the image.
    """
    result = model.predict(np.expand_dims(image, axis=0))

    plt.imshow(image.numpy().astype(np.uint8))
    plt.axis('off')
    plt.title("Tomatoes found!" if bool(np.round(np.squeeze(result))) else "No tomatoes found!")
    plt.show()

    return bool(np.round(np.squeeze(result)))


if __name__ == '__main__':
    # Construct the argument parser and parse the hyperparameters.
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_path", type=str, help="Path to the test image.",
                    default=TEST_IMAGE_PATH)
    ap.add_argument("-w", "--model_weights", default=CHECKPOINTS_PATH,
                    help="Path to model checkpoint to be loaded.")

    args = vars(ap.parse_args())

    model = DetectionModel(input_shape=IMAGE_SHAPE).get_model()

    model.load_weights(args["model_weights"])

    image = decode_image(args["img_path"])

    found = has_tomatoes(image, model)

    print(found)
