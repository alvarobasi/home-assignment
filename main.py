import tensorflow as tf
from utils import get_image_label_pairs, get_dataset_objects, test_dataset, test_model, plot_training_results
import argparse
from Network import DetectionModel
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K


def weighted_binary_crossentropy(y_true, y_pred, weight_negative=1., weight_positive=1.):
    y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight_positive + (1 - y_true) * K.log(1 - y_pred)) * weight_negative
    return K.mean(logloss, axis=-1)


def train(train_dataset, val_dataset, model, learning_rate, epochs, callback_list):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])

    h = model.fit(train_dataset, epochs=epochs, steps_per_epoch=training_steps,
                  validation_data=val_dataset, callbacks=callback_list)

    return h


def evaluate(test_dataset, model):
    results = model.evaluate(test_dataset)
    print("test loss, test acc:", results)


if __name__ == '__main__':

    # Construct the argument parser and parse the hyperparameters and paths.
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_path", default="datasets/images/",
                    help="Path to the dataset directory.")
    ap.add_argument("-a", "--annotations_file", default="datasets/labels/img_annotations.json",
                    help="Path to the img_annotations.json file.")
    ap.add_argument("-e", "--epochs", type=int, default=5,
                    help="# of epochs to train the network for")
    ap.add_argument("-p", "--plot", type=str, default="outputs/",
                    help="path to output loss/accuracy plot")
    ap.add_argument("-bs", "--batch_size", type=int, default=32,
                    help="Size of the mini-batch to be used in the training process.")
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.001,
                    help="Set learning rate value.")
    ap.add_argument("-ft", "--fine_tune_layers", type=float, default=40,
                    help="# of last layers to fine tune.")
    ap.add_argument("-fp16", "--mixed_precision", type=bool, default=True,
                    help="Enables mixed_precision training. Only available for Volta, Turing and Ampere GPUs.")
    ap.add_argument("-xla", "--xla_compilation", type=bool, default=False,
                    help="Enables XLA compilation mode in order to accelerate training. Linux only.")
    args = vars(ap.parse_args())

    tf.keras.backend.clear_session()
    # Enable XLA runtime execution in order to accelerate training in Linux systems.
    if args["xla_compilation"]:
        tf.config.optimizer.set_jit(False)

    # Variable for enabling AMP (Automatic Mixed Precision).
    if args["mixed_precision"]:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    X, Y = get_image_label_pairs(args["annotations_file"], args["dataset_path"])

    img_shape = (600, 600, 3)

    train_ds, training_steps, val_ds, _, test_ds = get_dataset_objects(X, Y, args["batch_size"])

    # Model creation using transfer learning from a pre-trained ResNet50V2.
    model_object = DetectionModel(input_shape=img_shape)
    model = model_object.get_model()

    # Testing dataset in one single mini-batch.
    test_dataset(test_ds, model_object.get_data_augmentation_layer())

    model.summary()

    best_checkpoint = ModelCheckpoint(
        filepath='outputs/saved_model/detector_weights',
        monitor='accuracy',
        save_weights_only=False,
        save_best_only=True,
        save_freq=training_steps * 5,
        verbose=2,
        mode='auto')

    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, verbose=1, mode='max')

    callbacks = [best_checkpoint]

    history = train(train_ds, val_ds, model, args["learning_rate"], args["epochs"], callbacks)

    plot_training_results(history, args["plot"] + "pre_fine_tuning_plot.png")

    test_model(test_ds, model, show_images=False)

    model.save_weights("outputs/detector_weights.h5", overwrite=True)

    # Returns how many layers are in the base model
    print("Number of layers in the base model: ", len(model_object.get_base_model().layers))

    # Fine-tune the last upper layers.
    layers_to_fine_tune = 30

    model_object.set_fine_tune_layers(layers_to_fine_tune)

    model = model_object.get_model()
    model.summary()

    # Train the model again in order to fine tune the last layers.
    history = train(train_ds, val_ds, model, args["learning_rate"] / 100, args["epochs"] * 2, callbacks)

    plot_training_results(history, args["plot"] + "fine_tuning_plot.png")

    test_model(test_ds, model, show_images=False)

    model.save_weights("outputs/detector_fine_tuned_weights.h5", overwrite=True)
