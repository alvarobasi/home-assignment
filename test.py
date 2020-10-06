import argparse
import numpy as np
import matplotlib.pyplot as plt
from Network import DetectionModel
from config import *
from utils import decode_image
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import scipy.ndimage
from sklearn.preprocessing import normalize


def has_tomatoes(image, model, compute_cam):
    """
    Returns a boolean stating whether or not there are tomatoes present in the image.

    :param image: Matrix containing the image data.
    :param model: tf.keras.Model object containing the loaded model.
    :param compute_cam: Enables the computations and overlay of the Class Activation Maps (CAM) of the prediction.

    :returns: Returns a boolean stating whether or not there are tomatoes present in the image.
    """
    if compute_cam:

        base_model_name = "resnet50v2"

        # As in the paper, we select the last layer for predictions and last convolutional layer for feature extraction.
        last_conv_layer_resnet_name = "conv5_block3_out"
        # last_conv_layer_resnet_name = "post_bn"
        classification_layers_names = [
            "global_average_pooling2d",
            "dense",
        ]

        # We retreive the weight matrix of the GAP -> Dense layer connection. Shape (m, 2048, 1) with m = 1
        gap_dense_weight_matrix = model.get_layer(classification_layers_names[1]).get_weights()[0]

        # The base model used for transfer learning is retreived into a new model outputing the features extracted
        # from the last convolutional layer.
        base_model = tf.keras.Model(model.get_layer(base_model_name).input,
                                    model.get_layer(base_model_name).get_layer(last_conv_layer_resnet_name).output)

        # Computes the features map and predictions for each example. Shape (m, 19, 19, 2048) with m = 1
        out_convolutions = base_model.predict(np.expand_dims(tf.keras.applications.resnet_v2.preprocess_input(image),
                                                             axis=0))
        prediction = model.predict(np.expand_dims(image, axis=0))

        # Sets the shape as (19, 19, 2048), as there is only one example in the test program.
        features_cam = np.squeeze(out_convolutions)

        # Sets the weights to a one dimensional array of shape (2048,)
        cam_weights = np.squeeze(gap_dense_weight_matrix)

        # Upsampling the features from 19x19 to the size of the image 600x600. Result shape: (600, 600, 2048)
        reshaped_feautes_cam = scipy.ndimage.zoom(features_cam,
                                                  (IMAGE_SHAPE[0] / features_cam.shape[0],
                                                   IMAGE_SHAPE[1] / features_cam.shape[1], 1), order=1)

        # As in the paper, we perform the matricial multiplication of the GAP->Dense weights with the reshaped features.
        # An "importance" matrix of 600x600 is generated.
        cam_output = np.dot(reshaped_feautes_cam, cam_weights)

        # The image with the CAM image are shown.
        plt.imshow(image.numpy().astype(np.uint8))
        plt.imshow(cam_output, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title("Tomatoes found!" if bool(np.round(np.squeeze(prediction))) else "No tomatoes found!")
        plt.show()

    else:
        prediction = model.predict(np.expand_dims(image, axis=0))

        plt.imshow(image.numpy().astype(np.uint8))
        plt.axis('off')
        plt.title("Tomatoes found!" if bool(np.round(np.squeeze(prediction))) else "No tomatoes found!")
        plt.show()

    return bool(np.round(np.squeeze(prediction)))


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
