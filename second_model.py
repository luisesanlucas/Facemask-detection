"""
This file exists primarily as a way to test function-oriented model
creation and training.

Last Modified: 05/03/21
"""
import utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib as plt

images, labels = utils.get_images_and_labels(faces=True)
print(images.shape)
print(labels.shape)
train, test = utils.train_and_test(images, labels)

model = models.Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=images.shape[1:]),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(3)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 1
history = model.fit(train[0], train[1], batch_size=1, epochs=epochs, validation_data=(test[0], test[1]))

model.save('./saved_model/test')
