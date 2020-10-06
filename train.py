import tensorflow as tf
from utils import get_image_label_pairs, get_dataset_objects, test_dataset, evaluate_model, plot_training_results
import argparse
from Network import DetectionModel
from tensorflow.python.keras.callbacks import ModelCheckpoint

from config import *


def train(train_dataset, val_dataset, model, learning_rate, epochs, callback_list, weights):
    """
    Trains the model using the given dataset and returns the metrics historical for plotting.

    :param weights: If set, adds a weight to each class during training.
    :param train_dataset: tf.Dataset object used for training the model.
    :param val_dataset: tf.Dataset object used for evaluation.
    :param model: tf.keras.Model object with the model to be trained.
    :param learning_rate: Learning rate of the optimizer.
    :param epochs: # of epochs for the training phase.
    :param callback_list: list() of keras.callbacks to be applied during the training process.

    :returns h: Training history containing the history of different losses and metrics generated during training.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])

    if weights is not None:
        h = model.fit(train_dataset, epochs=epochs, steps_per_epoch=training_steps,
                      validation_data=val_dataset, callbacks=callback_list,
                      class_weight={0: weights[0], 1: weights[1]})
    else:
        h = model.fit(train_dataset, epochs=epochs, steps_per_epoch=training_steps,
                      validation_data=val_dataset, callbacks=callback_list)

    return h


if __name__ == '__main__':

    # Construct the argument parser and parse the hyperparameters and paths.
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_path", default=DATASET_PATH,
                    help="Path to the dataset directory.")
    ap.add_argument("-a", "--annotations_path", default=LABELS_PATH,
                    help="Path to the img_annotations.json file.")
    ap.add_argument("-e", "--epochs", type=int, default=EPOCHS,
                    help="# of epochs to initially train the network for. Fine tuning will be 3 times the selected"
                         "batch size")
    ap.add_argument("-bs", "--batch_size", type=int, default=BATCH_SIZE,
                    help="Size of the mini-batch to be used in the training process.")
    ap.add_argument("-lr", "--learning_rate", type=float, default=LEARNING_RATE,
                    help="Set learning rate value.")
    ap.add_argument("-ftl", "--fine_tune_layers", type=int, default=FINE_TUNE_LAYERS,
                    help="# of last layers to fine tune.")
    ap.add_argument("-fp16", "--mixed_precision", type=bool, default=MIXED_PRECISION,
                    help="Enables mixed_precision training. Only available for Volta, Turing and Ampere GPUs.")
    ap.add_argument("-xla", "--xla_compilation", type=bool, default=XLA,
                    help="Enables XLA compilation mode in order to accelerate training. Linux only.")
    args = vars(ap.parse_args())

    # Clear the Keras session.
    tf.keras.backend.clear_session()

    # Enable XLA runtime execution in order to accelerate training in Linux systems.
    tf.config.optimizer.set_jit(args["xla_compilation"])

    # Enable AMP (Automatic Mixed Precision).
    if args["mixed_precision"]:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    # Retreive the full dataset paths along with their labels.
    X, Y, class_weights = get_image_label_pairs(args["annotations_path"], args["dataset_path"],
                                                balanced=BALANCED_DATASET)

    train_ds, training_steps, val_ds, _, test_ds = get_dataset_objects(X, Y, args["batch_size"])

    # Model creation using transfer learning from a pre-trained ResNet50V2.
    model_object = DetectionModel(input_shape=IMAGE_SHAPE)
    model = model_object.get_model()

    # Allows to watch the dataset in one single mini-batch.
    # test_dataset(test_ds, model_object.get_data_augmentation_layers())

    # Showing the summary of the model in order to check trainable params.
    model.summary()

    # Checkpoint
    best_checkpoint = ModelCheckpoint(
        filepath=CHECKPOINTS_PATH + "saved_model",
        monitor='accuracy',
        save_weights_only=True,
        save_best_only=True,
        save_freq=training_steps * 5,
        verbose=1,
        mode='auto')

    callbacks = [best_checkpoint]

    history = train(train_ds, val_ds, model, args["learning_rate"], args["epochs"], callbacks, class_weights)

    plot_training_results(history, PLOTS_PATH + "pre_fine_tuning_plot.png")

    evaluate_model(test_ds, model, show_images=False)

    # Fine-tune the last upper layers.
    layers_to_fine_tune = args["fine_tune_layers"]
    model_object.set_fine_tune_layers(layers_to_fine_tune)

    # Retreive the new model.
    model = model_object.get_model()

    model.summary()

    # Train the model again in order to fine tune the last layers. We use a lower learning rate and double epochs, as
    # this session will be the most important one for the model performance.
    history = train(train_ds, val_ds, model, FINE_TUNING_LR, args["epochs"] * 3, callbacks, class_weights)

    plot_training_results(history, PLOTS_PATH + "fine_tuning_plot.png")

    evaluate_model(test_ds, model, show_images=False)
