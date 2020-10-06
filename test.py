import argparse
from Network import DetectionModel
from config import *
from utils import decode_image, has_tomatoes
import tensorflow as tf


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Construct the argument parser and parse the hyperparameters.
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img_path", type=str, help="Path to the test image.",
                    default=TEST_IMAGE_PATH)
    ap.add_argument("-w", "--model_weights", default=CHECKPOINTS_PATH,
                    help="Path to model checkpoint to be loaded.")
    ap.add_argument("-cam", "--compute_cam", type=bool, default=ENABLE_CAM_COMPUTATION,
                    help="Compute and overlay the class activation map on the image.")

    args = vars(ap.parse_args())

    model_object = DetectionModel(input_shape=IMAGE_SHAPE)

    # Load the model and its weights.
    model = model_object.get_model()
    model.load_weights(args["model_weights"])

    # Reads and decodes the selected test image.
    image = decode_image(args["img_path"])

    # Check wether there are tomatoes present in the image.
    found = has_tomatoes(image, model, ENABLE_CAM_COMPUTATION)
    print(found)
