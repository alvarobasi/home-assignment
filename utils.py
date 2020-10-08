from tensorflow.python.client import device_lib
import json
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from config import *
from sklearn.utils.class_weight import compute_class_weight
import scipy.ndimage
import cv2 as cv


@tf.function
def map_fn(path, label):
    """
    Function to be executed in the input pipeline.

    :param path: Path string pointing to an image location.
    :param label: Float label corresponding to that image.

    :returns image, label: Two tf.Tensors containing the image data along with its label.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=[600, 600])

    return image, label


def decode_image(path):
    """
    Function decoding from an image path to an RGB 0-255 image.

    :param path: Path string pointing to an image location.
    :returns image, label: Returns the decoded image and its label.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=[600, 600])

    return image


def print_available_devices():
    """ Prints the available hardware devices. """
    local_device_protos = [(x.name, x.device_type, x.physical_device_desc) for x in device_lib.list_local_devices()]
    for device_name, device_type, device_desc in local_device_protos:
        print("Device : {}\n\t type : {}\n\t desc :{}\n".format(device_name, device_type, device_desc))


def test_dataset(ds, data_augmentation=None):
    """
    Shows the images contained in the input dataset for a single mini-batch specified before. For debugging purposes.

    :param ds: tf.Dataset object used for testing the ouputs.
    :param data_augmentation: Sequence of layers for data_augmentation. Allows to debug what is actually being
    introduced to the network. Default is None.
    """
    ds_iterator = iter(ds)
    image_batch, label_batch = next(ds_iterator)

    if data_augmentation is not None:
        image_batch = data_augmentation(image_batch, training=True)

    for i in range(len(image_batch)):
        plt.title("Tomatoes found!" if label_batch[i].numpy() == 1 else "No tomatoes found!")
        plt.imshow(image_batch.numpy()[i].astype(np.uint8))
        plt.axis('off')


def compute_multiple_cam(model, image_batch):
    """
    Returns a list of CAM (Class Activation Map) matrices for each image in the batch.

    :param image_batch: Tensor containing all the batch of images.
    :param model: tf.keras.Model object containing the loaded model.

    :returns: cam_map_list: Stores a list of CAM ndarray matrices.
    """
    base_model_name = "resnet50v2"

    # As in the paper, we select the last layer for predictions and last convolutional layer for feature extraction.
    last_conv_layer_resnet_name = "conv5_block3_out"
    # last_conv_layer_resnet_name = "post_bn"
    classification_layers_names = [
        "global_average_pooling2d",
        "dense",
    ]

    # We retreive the weight matrix of the GAP -> Dense layer connection. Shape (2048, 1)
    gap_dense_weight_matrix = model.get_layer(classification_layers_names[1]).get_weights()[0]

    # The base model used for transfer learning is retreived into a new model outputing the features extracted
    # from the last convolutional layer.
    base_model = tf.keras.Model(model.get_layer(base_model_name).input,
                                model.get_layer(base_model_name).get_layer(last_conv_layer_resnet_name).output)

    # Computes the features map and predictions for each example. Shape (m, 19, 19, 2048) with m = 1
    out_convolutions = base_model.predict(tf.keras.applications.resnet_v2.preprocess_input(image_batch))

    cam_map_list = []

    for i in range(len(image_batch.numpy())):
        # Sets the shape as (19, 19, 2048), as there is only one example in the test program.
        features_cam = out_convolutions[i, :, :, :]

        # Sets the weights to a one dimensional array of shape (2048,)
        cam_weights = np.squeeze(gap_dense_weight_matrix)

        # Upsampling the features from 19x19 to the size of the image 600x600. Result shape: (600, 600, 2048)
        reshaped_feautes_cam = scipy.ndimage.zoom(features_cam,
                                                  (IMAGE_SHAPE[0] / features_cam.shape[0],
                                                   IMAGE_SHAPE[1] / features_cam.shape[1], 1), order=1)

        # As in the paper, we perform the matricial multiplication of the GAP->Dense weights with the reshaped features.
        # An "importance" matrix of 600x600 is generated.
        cam_output = np.dot(reshaped_feautes_cam, cam_weights)

        cam_map_list.append(cam_output)

    return cam_map_list


def compute_single_cam(model, image):
    """
    Returns the CAM (Class Activation Map) Matrix.

    :param image: Matrix containing a single image data.
    :param model: tf.keras.Model object containing the loaded model.

    :returns: cam_output: Stores the CAM ndarray Matrix.
    """
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

    return cam_output


def has_tomatoes(image, model, with_cam):
    """
    Returns a boolean stating whether or not there are tomatoes present in the image.

    :param with_cam: Enables the computations and overlay of the Class Activation Maps (CAM) of the prediction.
    :param image: Matrix containing the image data.
    :param model: tf.keras.Model object containing the loaded model.

    :returns: Returns a boolean stating whether or not there are tomatoes present in the image.
    """
    prediction = model.predict(np.expand_dims(image, axis=0))

    if with_cam:
        cam_mat = compute_single_cam(model, image)

        # The image with the CAM overlayed image are shown.
        plt.imshow(image.numpy().astype(np.uint8))
        plt.imshow(cam_mat, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title("Tomatoes found!" if bool(np.round(np.squeeze(prediction))) else "No tomatoes found!")
        plt.show()
    else:
        plt.imshow(image.numpy().astype(np.uint8))
        plt.axis('off')
        plt.title("Tomatoes found!" if bool(np.round(np.squeeze(prediction))) else "No tomatoes found!")
        plt.show()

    return bool(np.round(np.squeeze(prediction)))


def evaluate_model(ds, model, with_cam, show_images=False):
    """
    Evaluates the given model using the given dataset and prints the classification error (# errors/total_images).
    :param ds: tf.Dataset object used for testing the model.
    :param model: tf.keras.Model object with the model to be tested.
    :param with_cam: Enables the computations and overlay of the Class Activation Maps (CAM) of the prediction.
    :param show_images: Enables the posibility to print the images being tested. For debugging purposes or just to see
    the results in multiple images at the same time.
    """
    ds_iterator = iter(ds)
    image_batch, label_batch = next(ds_iterator)
    error_counter = 0
    image_counter = 0

    while True:

        if with_cam and show_images:
            cam_outputs = compute_multiple_cam(model, image_batch)

        # Predicts the labels of the dataset batch.
        predictions = model.predict(image_batch)
        image_counter += len(image_batch.numpy())

        # Shows the labels/predicted labels toghether for debugging purposes.
        # print("Labels:")
        # print(label_batch.numpy())
        # print("Predicted:")
        # print(np.round(np.squeeze(predictions)))

        # Plots the images along with their true/predicted labels. For debugging purposes.
        if show_images:

            for i in range(len(image_batch.numpy())):
                plt.subplot(2, 2, i + 1)
                plt.imshow(image_batch.numpy()[i].astype(np.uint8))
                if with_cam:
                    plt.imshow(cam_outputs[i], cmap='jet', alpha=0.5)
                plt.axis('off')
                plt.title("Label: " + str(label_batch.numpy()[i]) + " Predicted: " +
                          str(np.round(np.squeeze(predictions[i]))))
            plt.show()

        # Counts the number of calssification errors.
        for i in range(len(image_batch.numpy())):
            if label_batch.numpy()[i] != np.round(np.squeeze(predictions[i])):
                error_counter += 1

        # Try/except block in order to iterate through the ds_iterator with an end.
        try:
            image_batch, label_batch = next(ds_iterator)
        except StopIteration:
            print("Error rate on " + str(image_counter) + " images tested: "
                  + str(error_counter / image_counter))
            exit(0)


def evaluate_model_box_predictions(ds, model):
    """
    Evaluates the performance of the model in localizing tomatoes in the image by comparing
    the predicted box with the ground truth.

    :param ds: tuple containing the paths, labels and boxes of each test image.
    :param model: tf.keras.Model object with the model to be tested.
    """
    x_test = ds[:, 0]
    y_test = ds[:, 1].astype(np.float32)
    b_test = ds[:, 2]

    for i in range(len(x_test)):
        print("Loading new image...")
        image = decode_image(x_test[i])
        prediction = model.predict(np.expand_dims(image, axis=0))

        if len(b_test[i]) > 0:
            cam_mat = compute_single_cam(model, image)
            _, thresholded_cam = cv.threshold(cam_mat, 22, np.max(cam_mat), cv.THRESH_BINARY)
            norm_thresholded_cam = (thresholded_cam / np.max(cam_mat)) * 255
            blobs, _ = cv.findContours(norm_thresholded_cam.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            # Find the index of the largest blob
            # if len(b_test[i]) > 1:
            image = image.numpy().astype(np.uint8)
            for blob in blobs:
                pos_x, pos_y, rect_w, rect_h = cv.boundingRect(blob)

                # Prints the deteceted rectangle.
                image = cv.rectangle(image, (pos_x, pos_y),
                                     (pos_x + rect_w, pos_y + rect_h), (0, 0, 255), 2)
                image = cv.putText(image, 'Tomato ' + str(np.around(np.squeeze(prediction), 2) * 100) + "%",
                                   (pos_x + 10, pos_y - 10), 0, 0.8, (0, 0, 255))

            for rect in b_test[i]:
                # Prints the ground truth rectangle.
                image = cv.rectangle(image,
                                     (rect[0], rect[1]),
                                     (rect[0] + rect[2],
                                      rect[1] + rect[3]), (0, 255, 0), 2)
                image = cv.putText(image, 'Ground truth',
                                   (rect[0] + 10, rect[1] - 10), 0, 0.8, (0, 255, 0))

            plt.imshow(image)
            # plt.imshow(cam_mat, cmap='jet', alpha=0.5)
            plt.axis('off')
            plt.title("Tomatoes found!" if bool(np.round(np.squeeze(prediction))) else "No tomatoes found!")
            plt.show()

        else:
            plt.imshow(image.numpy().astype(np.uint8))
            plt.axis('off')
            plt.title("Tomatoes found!" if bool(np.round(np.squeeze(prediction))) else "No tomatoes found!")
            plt.show()


def get_dataset_objects(x, y, bound_boxes, batch_size, valid_split=0.2, test_split=0.05):
    """
    Slipts dataset intro train, validation and set, shuffles the data and returns the tf.Dataset objects corresponding
    to each of the dataset splits.

    :param bound_boxes: List of bounding boxes.
    :param x: Numpy array containing the images' paths.
    :param y: Numpy array containing the labels.
    :param batch_size: Mini-batch size used during training, validation and testing.
    :param valid_split: % of images of the entire dataset being reserved for validation.
    :param test_split: % of images of the training dataset being reserved for testing.

    :return train_ds: tf.Dataset object of the training set.
    :return training_steps: Integer containing the training steps for each epoch.
    :return val_ds: tf.Dataset object of the validation set.
    :return validation_steps: Integer containing the validation steps for each epoch.
    :return test_ds: tf.Dataset object of the testing set.
    """
    x_train, x_val, y_train, y_val, b_train, b_eval = train_test_split(x, y, bound_boxes, test_size=valid_split,
                                                                       random_state=1805)
    x_train, x_test, y_train, y_test, b_train, b_test = train_test_split(x_train, y_train, b_train,
                                                                         test_size=test_split,
                                                                         random_state=1805)

    z_test = np.column_stack((x_test, y_test, b_test))

    # Let's save the test set for future evaluation.
    np.save(TEST_SET_FILE_PATH, z_test)

    training_steps = int(len(x_train) // batch_size)
    validation_steps = int(len(x_val) // batch_size)

    # Creating the tf.Dataset for the training dataset.
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.repeat(count=-1)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Creating the tf.Dataset for the testing dataset.
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Creating the tf.Dataset for the validation dataset.
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, training_steps, val_ds, validation_steps, test_ds


def plot_training_results(hist, path):
    """
    Plots the given accuracy/loss in «hist» for train and validation datasets and saves the figure into a .png file.

    :param hist: Dictionary containing the historic of metrics obtained during training.
    :param path: Path location where the plot image should be saved.
    """
    # Plot the training loss and accuracy
    plt.figure(figsize=(8, 8))
    plt.style.use("ggplot")

    plt.subplot(2, 1, 1)
    plt.plot(hist.history["accuracy"], label='Training Accuracy')
    plt.plot(hist.history["val_accuracy"], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(hist.history["loss"], label='Training Loss')
    plt.plot(hist.history["val_loss"], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')

    plt.savefig(path)


def get_image_label_pairs(annotations_path, image_dir_path, balanced=True):
    """
    Taking the img_annotations.json file, it extracts all the images paths along with their labels and builds two new
    arrays containing the full image paths along with their label, which is converted to 0 (no tomatoes present)
    or 1 (tomatoes present). In order to achieve that, a tomato_label_list has been cosntructed by hand-selecting the
    labels corresponding to tomatoes from the label_mapping.csv file.

    :param balanced: Wether or not the resulting dataset should be balanced. That is, enable oversampling.
    :param annotations_path: Path to the img_annotations.json file containing the labels of every image.
    :param image_dir_path: List containing all the images paths.
    """

    # Opening the img_annotations.json file.
    f = open(annotations_path, "r")
    labels = f.read()

    # List of tomato labels extracted from the label_mapping.csv
    tomato_label_list = [
        "939030726152341c154ba28629341da6_lab",
        "9f2c42629209f86b2d5fbe152eb54803_lab",
        "4e884654d97603dedb7e3bd8991335d0_lab",
        "f3c12eecb7abc706b032ad9a387e6f01_lab",
        "e306505b150e000f5e1b4f377ad688a0_lab",
        "9de66567ece99ca7265921bf54cc6b9f_lab"
    ]

    # Load the json string to a dict().
    labels_dic = json.loads(labels)

    x_negatives = []
    y_negatives = []
    x_positives = []
    y_positives = []

    positive_boxes_list = list()
    negative_boxes_list = list()

    class_weights = None

    # Compares all the boxes labels from every image with the tomato_label_list and stores a new label 0 or 1, depending
    # on whether there is a tomato label present or not.
    i = 0
    for image_path, boxes in labels_dic.items():
        found_tomatoes = 0.
        for box in boxes:
            if box["id"] in tomato_label_list:
                if len(y_positives + y_negatives) != i + 1:
                    found_tomatoes = 1.
                    x_positives.append(image_dir_path + image_path)
                    y_positives.append(found_tomatoes)
                    positive_boxes_list.append([])
                positive_boxes_list[i - len(y_negatives)].append(box["box"])
        if found_tomatoes == 0:
            x_negatives.append(image_dir_path + image_path)
            y_negatives.append(found_tomatoes)
            negative_boxes_list.append([])
        i += 1

    # Oversampling is applied over the positive set in order to balance the datasets.
    # 2500(other)/500(tomato) = 5 >> 1.5
    if balanced:

        ratio = len(x_positives + x_negatives) // len(x_positives)

        x_positives = x_positives * (ratio - 1)
        y_positives = y_positives * (ratio - 1)
        positive_boxes_list = positive_boxes_list * (ratio - 1)

        x = x_negatives + x_positives
        y = y_negatives + y_positives
        boxes = negative_boxes_list + positive_boxes_list

    else:
        x = x_negatives + x_positives
        y = y_negatives + y_positives
        boxes = negative_boxes_list + positive_boxes_list
        class_weights = compute_class_weight('balanced', np.unique(y), y)

    return np.array(x), np.array(y), boxes, class_weights
