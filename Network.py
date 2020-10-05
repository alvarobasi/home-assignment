import tensorflow as tf


class DetectionModel(object):

    def __init__(self, input_shape=None):
        self.__input_shape = input_shape
        self.__base_model = tf.keras.applications.ResNet50V2(weights="imagenet",
                                                             input_shape=self.__input_shape,
                                                             include_top=False,
                                                             )
        self.__model = self.__build()

    def __build(self):
        # Freeze all the base model weights.
        self.__base_model.trainable = False

        inputs = tf.keras.layers.Input(shape=self.__input_shape)
        x = self.get_data_augmentation_layer()(inputs)
        x = tf.keras.applications.resnet_v2.preprocess_input(inputs)

        # # New classification upper layers added.
        # x = self.__base_model(x, training=False)
        # x = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(x)
        # x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dense(256, activation="relu")(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        # outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        # New classification upper layers added.
        x = self.__base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        return model

    def set_fine_tune_layers(self, last_layers):
        self.__base_model.trainable = True

        for layer in self.__base_model.layers[:-last_layers]:
            layer.trainable = False

    def get_base_model(self):
        return self.__base_model

    def get_model(self):
        return self.__model

    @staticmethod
    def get_data_augmentation_layer():
        return tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                                    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                                    ])
