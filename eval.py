import argparse
import tensorflow as tf
from utils import get_image_label_pairs, get_dataset_objects, evaluate_model
from Network import DetectionModel
from config import *
import numpy as np
from utils import map_fn
from Network import DetectionModel

if __name__ == '__main__':
    # Construct the argument parser and parse the hyperparameters.
    ap = argparse.ArgumentParser()
    ap.add_argument("-tds", "--test_dataset_file", default=TEST_SET_FILE_PATH,
                    help="Path to the test dataset file.")
    ap.add_argument("-w", "--model_weights", default=CHECKPOINTS_PATH + "saved_model/saved_model",
                    help="Path to model checkpoint to be loaded.")
    args = vars(ap.parse_args())

    model = DetectionModel(IMAGE_SHAPE).get_model()
    model.load_weights(args["model_weights"])

    z_test = np.load(args["test_dataset_file"])

    x_test = z_test[:, 0]
    y_test = z_test[:, 1].astype(np.float32)

    # Creating the tf.Dataset for the testing dataset.
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    evaluate_model(test_ds, model, show_images=False)
