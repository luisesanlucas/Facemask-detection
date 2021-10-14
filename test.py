import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import preprocessing

model= keras.models.load_model('./saved_model/saved CNN_B')

model.summary()
print

im_height = 256
im_width = 256

batch_size=32

im_path='./test_images'


class_names = ['mask', 'no_mask', 'incorrect_mask']

###resize images to be the same size 
img_height = 256
img_width = 256

training_data = []

### Create training dataset
def create_training_data():

    path = im_path 
    
    for img in list(sorted(os.listdir(path))):
        print(img)
        try:                
            img_array = cv2.imread(os.path.join(path,img)) #keep colors
            new_array = cv2.resize(img_array, (img_height, img_width))  # resize 
            training_data.append([new_array])  # add to training data
        except Exception as e: 
            pass

create_training_data()
#print(f'Training data size {len(training_data)}')


#Create model
X = []  #features
y = []  #labels

for features in training_data:
    X.append(features)

#print(X[0].reshape(-1, img_height, img_width, 3)) #color
X = np.array(X).reshape(-1, img_height, img_width, 3)  # 3 for color
y = np.array(y) 

##Normalize by dividing pixel values by 255 and convert dtype into float32
def normalize_img(image):
    return tf.cast(image, tf.float32)/255.0

X_norm = normalize_img(X)

X_norm=np.array(X_norm)

preds= model.predict(X_norm)
y_pred = []
for p in preds:
    y_pred.append(np.argmax(p))

print(y_pred)   
