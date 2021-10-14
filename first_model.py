#from utils import *

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras import preprocessing

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,recall_score, classification_report

im_height = 256
im_width = 256

batch_size=32

labelf_path='./images'

train = tf.keras.preprocessing.image_dataset_from_directory(
    labelf_path,
     color_mode='rgb', batch_size=batch_size, image_size=(im_height,
    im_width), shuffle=True, seed=123, validation_split=0.2, subset='training',
    labels='inferred',
    label_mode='int',
    class_names=['mask','no_mask','incorrect_mask']
)

val = tf.keras.preprocessing.image_dataset_from_directory(
    labelf_path,
     color_mode='rgb', batch_size=batch_size, image_size=(im_height,
    im_width), shuffle=True, seed=123, validation_split=0.2, subset='validation',
    labels='inferred',
    label_mode='int',
    class_names=['mask','no_mask','incorrect_mask']
)

np_train=train.as_numpy(train)
np_val = val.as_numpy(val)
for x in np_train:
  print(x)
class_names = train.class_names

print(class_names)

num_classes = 3

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(im_height, im_width,3)),
    layers.experimental.preprocessing.RandomRotation(0.1)
  ])

vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(im_height,im_width,3))

for layer in vgg19.layers:
    layer.trainable = False

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(im_height, im_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.7),
  #vgg19,
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 1
class_weights={
  0:3,
  1:1,
  2:1.5
  }

history = model.fit(
  train,
  validation_data=val,
  epochs=epochs,
  #class_weight=class_weights
)



predictions = np.array([])
labels =  np.array([])
for x, y in val:
  predictions = np.concatenate([predictions, np.argmax(model.predict(x,batch_size=batch_size), axis=-1)])
y = np.concatenate([y for x, y in val], axis=0)
#labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

print(tf.math.confusion_matrix(labels=y, predictions=predictions).numpy())
print('\n')
print(classification_report(y, predictions))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save_weights('weights/vgg19')
model.save('saved_model/vgg19')