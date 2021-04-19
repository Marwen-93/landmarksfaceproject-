import tensorflow as tf
from tensorflow.image import ResizeMethod
from sklearn.model_selection import  train_test_split
from tensorflow.keras.applications.vgg16 import VGG16


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import time
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import matplotlib.image as mpimg
from collections import OrderedDict
from skimage import io, transform
from math import *
import xml.etree.ElementTree as ET 


def load_files(xmlpath,folderpath):
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    image_filenames=[]
    landmarks=[]
    crops=[]
    imagename=[]
    
    root_dir = folderpath
       
    for filename in root[2]:
        image_filenames.append(os.path.join(root_dir, filename.attrib['file']))
        crops.append(filename[0].attrib)
        imagename.append(filename.attrib['file'])
        
        landmark = []
        for num in range(68):
            x_coordinate = int(filename[0][num].attrib['x'])
            y_coordinate = int(filename[0][num].attrib['y'])
            landmark.append([x_coordinate, y_coordinate])
        landmarks.append(landmark)
    return image_filenames, landmarks , crops ,imagename
    
def read_image(image_pat):
    image=cv2.imread(image_pat)[:, :, ::-1]
    return image


def crop_face(image, landmarks, crops):
    left = int(crops['left'])
    top = int(crops['top'])
    width = int(crops['width'])
    height = int(crops['height'])

    image = tf.image.crop_to_bounding_box(
        image, left//2, top//2, width+top, height+left)
    img_shape = np.array(image).shape
    landmarks = tf.constant(landmarks, dtype='int32') - \
        tf.constant([[left//2, top//2]], dtype='int32')
    image = tf.image.resize(image, (100, 100), method=ResizeMethod.BILINEAR)
    landmarks = (tf.constant(landmarks, dtype='int32') //
                 tf.constant([[img_shape[1]//100, img_shape[0]//100]], dtype='int32'))
    #landmarks=tf.constant(landmarks,dtype='int32' )*tf.constant( [[100,100]],dtype='int32')

    return image, landmarks


xmlpath = "/home/marwen/Desktop/landmarks_project/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml"
folderpath='/home/marwen/Desktop/landmarks_project/ibug_300W_large_face_landmark_dataset'
image_filenames,landmarks,crops,imagename =load_files(xmlpath,folderpath)

dit ={'images': [],'landmarks':[],'imagename':[]}
for i in range(len(image_filenames)):
    try:
        image=read_image(image_filenames[i])
        
        image,landmark = crop_face(image ,landmarks[i], crops[i])
        dit['images'].append(image)
        dit['landmarks'].append(landmark)
        dit['imagename'].append(imagename[i])
    except :
        -1

assert len(dit['landmarks']) !=0
assert  len(dit['images'])==len(dit['landmarks'])
# for i in range(20):
#     print(dit['imagename'][i])
#     x=[]
#     y=[]
#     t=dit['landmarks'][i]
#     im=dit['images'][i]/255
#     for i in t:
#         x.append(i[0])
#         y.append(i[1])
#     plt.figure(figsize=(10,10))
#     plt.imshow(im)
#     plt.scatter(x,y,s=50, c= 'g')




X= np.array(dit["images"])
y=[]
for i in np.array(dit["landmarks"]):
    y.append(i.flatten())
y=np.array(y)
X_train,X_test,y_train, y_test = train_test_split(X,y, test_size=0.1)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size=0.2)
print(len(X_train,), 'X train examples')
print(len(X_val), 'X validation examples')
print(len(X_test), 'X test examples')

#Create the model

#Create the model

img_size=100
num_classes = 136

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
  layers.ZeroPadding2D(padding=(1, 1)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.ZeroPadding2D(padding=(1, 1)),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.ZeroPadding2D(padding=(1, 1)),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(num_classes)
  ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

#Model summary

model.summary()


epochs=100
history = model.fit(
  X_train,y_train,
  validation_data=(X_val,y_val),
  epochs=epochs
)




acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


y_predict= model.predict(X_test)
for j in range(10):
    x=[]
    y=[]
    t=y_predict[j]
    for i in range(len(t)):
        if i%2==0:
            x.append(t[i])
        else:
            y.append(t[i])
        
    plt.figure(figsize=(10,10))
    plt.imshow(X_test[j]/255)
    plt.scatter(x,y, s = 200, c = 'g')
    plt.show()
    
    
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)

im=read_image('/home/marwen/Desktop/landmarks_project/119408121_1018410488675923_1175409845330392762_n.jpg')
im =np.array(cv2.resize(im,dsize=(100,100))).reshape(100,100,3,-1)
plt.imshow(im)
im_pred=model.predict(im)



