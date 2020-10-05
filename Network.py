import tensorflow as tf


class DetectionModel(object):
    """
    The DetectionModel object creates and stores the model architecture. It also stores the base model of the
    TL (Transfer Learning)task separately in order to be able to unfreeze some of its layers used for the fine
    tuning stage.

    :param input_shape: tuple with the shape of the model input.

    Attributes:
        __input_shape: Stores the shape of the model input layer.
        __base_model: Stores the base model separately from the entire detection model.
        __model: Stores the full detection model.
    """

    def __init__(self, input_shape=None):
        self.__input_shape = input_shape

        # Base model creation. The top classification layers are removed to introduce ours for our specific task.
        self.__base_model = tf.keras.applications.ResNet50V2(weights="imagenet",
                                                             input_shape=self.__input_shape,
                                                             include_top=False,
                                                             )

        # Build data_augmentation layers.
        self.__data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                tf.keras.layers.experimental.preprocessing.RandomFlip("vertical"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            ]
        )

        # Building and storing the complete model.
        self.__model = self.__build()

    def __build(self):
        """
        Builds the detection model.

        :returns model: A keras.Model model architecture.
        """

        # Freeze all the base model weights.
        self.__base_model.trainable = False

        # Sets the graph of the initial layers: Input, data_augmentation layer and base model preprocessing layer.
        inputs = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.get_data_augmentation_layers()(inputs)
        x = tf.keras.applications.resnet_v2.preprocess_input(x)

        # New classification upper layers added. AP2D -> FLATTEN -> DENSE (256) -> RELU -> DROPOUT -> DENSE (1) -> SIGM
        # x = self.__base_model(x, training=False)
        # x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(x)
        # x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dense(256, activation="relu")(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        # outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        # New classification upper layers added. GAP -> DROPOUT -> DENSE (1) -> SIGM
        x = self.__base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        return model

    def set_fine_tune_layers(self, last_layers):
        """ Sets to trainable the last «last_layers» of the base model for fine tuning. """
        self.__base_model.trainable = True

        # Returns how many layers are in the base model
        if len(self.__base_model.layers) > last_layers:
            for layer in self.__base_model.layers[:-last_layers]:
                layer.trainable = False

    def get_base_model(self):
        """ Returns the base model used for the TL task. """
        return self.__base_model

    def get_model(self):
        """ Returns the entire detection model. """
        return self.__model

    def get_data_augmentation_layers(self):
        """ Returns the augmentation layers. """
        return self.__data_augmentation
