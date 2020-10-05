import argparse
import tensorflow as tf
from utils import get_image_label_pairs, get_dataset_objects, test_model
from Network import DetectionModel
from config import *

if __name__ == '__main__':
    # Construct the argument parser and parse the hyperparameters.
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default=DATASET_PATH,
                    help="Path to the dataset directory.")
    ap.add_argument("-l", "--labels", default=LABELS_PATH + "img_annotations.json",
                    help="Path to the img_annotations.json file.")
    ap.add_argument("-bs", "--batch_size", type=int, default=BATCH_SIZE,
                    help="Size of the mini-batch to be used in the training process.")
    ap.add_argument("-fp16", "--mixed_precision", type=bool, default=MIXED_PRECISION,
                    help="Enables mixed_precision training. Only available for Volta, Turing and Ampere GPUs.")
    args = vars(ap.parse_args())

    # Enabling AMP (Automatic Mixed Precision).
    if args["mixed_precision"]:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    model = DetectionModel((600, 600, 3)).get_model()
    model.load_weights("outputs/detector_weights_resnet_1e-3_20epochs_bs32_0.075error.h5")

    X, Y = get_image_label_pairs(args["labels"], args["dataset"])

    img_shape = (600, 600, 3)

    train_ds, training_steps, val_ds, _, test_ds = get_dataset_objects(X, Y, args["batch_size"], test_split=0.2)
    test_model(test_ds, model, show_images=False)
