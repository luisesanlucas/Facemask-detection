
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator 

###Path to image folders split by label
img_data_dir = './images'
data_with_mask = './images/mask'
data_without_mask = './images/no_mask'
data_incorrect_mask = './images/incorrect_mask'

data_path = {'mask':data_with_mask, 'no_mask':data_without_mask, 'incorrect_mask':data_incorrect_mask}

class_names = ['mask', 'no_mask', 'incorrect_mask']

'''
###show one image of the dataset 
for className in class_names:  
    path = os.path.join(img_data_dir,className)  # create path to the images
    for img in os.listdir(path):  # iterate over each image
        img_array = cv2.imread(os.path.join(path,img))  # in color 
        plt.imshow(img_array)  # graph it in color 
        plt.show()  

        break # to show only one image 
    break 


#print(img_array)
#print(img_array.shape)
'''
###resize images to be the same size 
img_height = 256
img_width = 256
'''
### show image
new_array = cv2.resize(img_array, (img_height, img_width))
plt.imshow(new_array, cmap='gray')
plt.show()
'''
training_data = []

### Create training dataset
def create_training_data():
    for className in class_names:  

        path = os.path.join(img_data_dir,className)  
        class_num = class_names.index(className)  # get the classification (0,1,2). 0=with_mask 1=without_mask, 2=incorrect_mask
       
        for img in os.listdir(path):
            try:                
                img_array = cv2.imread(os.path.join(path,img)) #keep colors
                new_array = cv2.resize(img_array, (img_height, img_width))  # resize 
                training_data.append([new_array, class_num])  # add to training data
            except Exception as e: 
                pass

create_training_data()
#print(f'Training data size {len(training_data)}')

random.shuffle(training_data)  # shuffle dataset

#Create model
X = []  #features
y = []  #labels

for features,label in training_data:
    X.append(features)
    y.append(label)

#print(X[0].reshape(-1, img_height, img_width, 3)) #color
X = np.array(X).reshape(-1, img_height, img_width, 3)  # 3 for color
y = np.array(y) 

##Normalize by dividing pixel values by 255 and convert dtype into float32
def normalize_img(image):
    return tf.cast(image, tf.float32)/255.0

X_norm = normalize_img(X)

X_norm=np.array(X_norm)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=1)
### Neural Network 
batch=64
epochs=3

num_classes = 3

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
    layers.experimental.preprocessing.RandomRotation(0.1)
  ])

vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(img_height,img_width,3))

for layer in vgg19.layers:
    layer.trainable = False

model = Sequential([
  vgg19,
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch, epochs=epochs, validation_data=(X_test,y_test))

### obtain results
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose = 2)
print(f'Training Accuracy  {train_accuracy}')




preds= model.predict(X_test)
y_pred = []
for p in preds:
    y_pred.append(np.argmax(p))
print(y_pred[0])
print('\n')
print(classification_report(y_true=y_test, y_pred=y_pred,target_names=class_names))

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

'''
### Commented out CNN to prevent running. Takes a very long time to run. 
### CNN build model 
batch=32
epochs=3
validation_split=0.3

model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  
model.add(Dense(64))
model.add(Dense(3))
model.add(Activation('softmax'))

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
model.fit(X_norm, y, batch_size=batch, epochs=epochs, validation_split=validation_split) 

'''
