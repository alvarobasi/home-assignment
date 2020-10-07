import argparse
import tensorflow as tf
from utils import evaluate_model
from config import *
import numpy as np
from utils import map_fn
from Network import DetectionModel

if __name__ == '__main__':
    # Required lines of code. Otherwise, the layer computations performed in the compute_cam function will raise a
    # cuDNN error.
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Construct the argument parser and parse the hyperparameters.
    ap = argparse.ArgumentParser()
    ap.add_argument("-tds", "--test_dataset_file", default=TEST_SET_FILE_PATH,
                    help="Path to the .npy test dataset file.")
    ap.add_argument("-w", "--model_weights", default=CHECKPOINTS_PATH,
                    help="Path to model checkpoint to be loaded.")
    ap.add_argument("-cam", "--compute_cam", type=bool, default=ENABLE_CAM_COMPUTATION,
                    help="Compute and overlay the class activation map on the image.")
    args = vars(ap.parse_args())

    model = DetectionModel(IMAGE_SHAPE).get_model()
    model.load_weights(args["model_weights"])

    z_test = np.load(args["test_dataset_file"], allow_pickle=True)

    x_test = z_test[:, 0]
    y_test = z_test[:, 1].astype(np.float32)
    b_test = z_test[:, 2]

    # Creating the tf.Dataset for the testing dataset.
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(EVAL_BATCH_SIZE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    evaluate_model(test_ds, model, args["compute_cam"], show_images=ENABLE_SHOW_EVAL_IMAGES)
