from tensorflow.python.client import device_lib
import json

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight


@tf.function
def map_fn(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size=[600, 600])

    return image, label


def print_available_devices():
    local_device_protos = [(x.name, x.device_type, x.physical_device_desc) for x in device_lib.list_local_devices()]
    for device_name, device_type, device_desc in local_device_protos:
        print("Device : {}\n\t type : {}\n\t desc :{}\n".format(device_name, device_type, device_desc))


def test_dataset(ds, data_augmentation=None):
    ds_iterator = iter(ds)
    image_batch, label_batch = next(ds_iterator)

    if data_augmentation is not None:
        image_batch = data_augmentation(image_batch)

    for i in range(len(image_batch)):
        plt.title("Tomatoes found!" if label_batch[i].numpy() == 1 else "No tomatoes found!")
        plt.imshow(image_batch.numpy()[i].astype(np.uint8))
        plt.axis('off')


def test_model(ds, model, show_images=False):
    ds_iterator = iter(ds)
    image_batch, label_batch = next(ds_iterator)
    error_counter = 0
    image_counter = 0

    while True:
        predictions = model.predict(image_batch)
        # predictions = tf.nn.sigmoid(predictions)
        image_counter += len(image_batch.numpy())
        print("Labels:")
        print(label_batch.numpy())
        print("Predicted:")
        print(np.round(np.squeeze(predictions)))
        if show_images:

            for i in range(len(image_batch.numpy())):
                plt.subplot(4, 4, i + 1)
                plt.imshow(image_batch.numpy()[i].astype(np.uint8))
                plt.axis('off')
                plt.title("Label: " + str(label_batch.numpy()[i]) + " Predicted: " +
                          str(np.round(np.squeeze(predictions[i]))))
            plt.show()

        for i in range(len(image_batch.numpy())):
            if label_batch.numpy()[i] != np.round(np.squeeze(predictions[i])):
                error_counter += 1

        try:
            image_batch, label_batch = next(ds_iterator)
        except StopIteration:
            print("Foodvisor error rate on " + str(image_counter) + " images tested: "
                  + str(error_counter / image_counter))
            break


def get_dataset_objects(x, y, batch_size, valid_split=0.2, test_split=0.1):

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=valid_split, random_state=1805)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=1805)

    training_steps = int(len(x_train) // batch_size)
    validation_steps = int(len(x_val) // batch_size)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.repeat(count=-1)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, training_steps, val_ds, validation_steps, test_ds


def plot_training_results(hist, path):
    # plot the training loss and accuracy
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


def get_image_label_pairs(annotations_path, image_dir_path):
    f = open(annotations_path, "r")
    labels = f.read()
    tomato_label_list = [
        "939030726152341c154ba28629341da6_lab",
        "9f2c42629209f86b2d5fbe152eb54803_lab",
        "4e884654d97603dedb7e3bd8991335d0_lab",
        "f3c12eecb7abc706b032ad9a387e6f01_lab",
        "e306505b150e000f5e1b4f377ad688a0_lab",
        "9de66567ece99ca7265921bf54cc6b9f_lab"
    ]

    labels_dic = json.loads(labels)
    x = []
    y = []
    for image_path, boxes in labels_dic.items():
        x.append(image_dir_path + image_path)
        found_tomatoes = 0
        for box in boxes:
            if box["id"] in tomato_label_list:
                found_tomatoes = 1
                break
        y.append(found_tomatoes)

    # Undersampling the set negatives in order to balance the datasets with the positives.
    # 2500(other)/500(tomato) = 5 >> 1.5
    # result = compute_class_weight('balanced', np.unique(y_train), y_train)

    positives = sum(y)

    mask = np.hstack([np.random.choice(np.where(y == l)[0], positives, replace=False)
                      for l in np.unique(y)])

    x_np = np.array(x)
    y_np = np.array(y).astype(np.float32)

    x = x_np[mask]
    y = y_np[mask]

    return x, y
