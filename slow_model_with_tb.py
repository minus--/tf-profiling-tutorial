# Python packages to manipulate files
import os
import pathlib
from pathlib import Path
import datetime

# Tensorflow and Numpy packages
import tensorflow as tf
import numpy as np

# Display related packages
import matplotlib.pyplot as plt
import tensorflow as tf


from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.python.keras.backend import repeat
import cProfile


def training():
    TRAINING_DIR = "./train/"
    VALID_DIR = "./val/"

    batch_size = 64
    img_height = 70
    img_width = 70

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAINING_DIR, seed=123, image_size=(img_height, img_width), batch_size=batch_size
    )
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VALID_DIR, seed=123, image_size=(img_height, img_width), batch_size=16
    )

    model = tf.keras.models.Sequential(
        [
            # Note the input shape is the desired size of the image 150x150 with 3 bytes color
            # This is the first convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(70, 70, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The second convolution
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The third convolution
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # The fourth convolution
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            # 512 neuron hidden layer
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(1, activation="softmax"),
        ]
    )

    logdir = os.path.join("./logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    model.summary()
    learning_rate = 0.01
    sgd = optimizers.SGD(lr=learning_rate)

    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=0,  # How often to log histogram visualizations
            embeddings_freq=0,  # How often to log embedding visualizations
            update_freq="batch",
            profile_batch="100, 250",
        )
    ]
    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        verbose=1,
        validation_steps=3,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    training()
