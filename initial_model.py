# Python packages to manipulate files
import os
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

TRAINING_DIR = "./train/"
VALID_DIR = "./val/"

training_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

train_generator = (
    training_datagen.flow_from_directory(
        TRAINING_DIR, target_size=(70, 70), class_mode="categorical", batch_size=64, shuffle=True
    )
    .cache()
    .prefetch(tf.data.experimental.AUTOTUNE)
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

val_generator = val_datagen.flow_from_directory(
    VALID_DIR, target_size=(70, 70), class_mode="categorical", batch_size=16
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
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

logdir = os.path.join("./logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=0,  # How often to log histogram visualizations
        embeddings_freq=0,  # How often to log embedding visualizations
        update_freq="batch",
        profile_batch="10, 250",
    )
]

model.summary()
learning_rate = 0.01
sgd = optimizers.SGD(lr=learning_rate)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

history = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=20,
    validation_data=val_generator,
    verbose=1,
    validation_steps=3,
    callbacks=callbacks,
)
