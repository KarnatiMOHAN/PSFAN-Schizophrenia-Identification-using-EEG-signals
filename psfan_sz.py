
import os,cv2
import numpy as np
import pandas as pd
from math import exp, sqrt
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from keras.utils import np_utils
import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.utils import shuffle
from keras import Input
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import  BatchNormalization, Concatenate, Add, DepthwiseConv2D, GlobalAveragePooling2D, SeparableConv2D
from tensorflow.keras.optimizers import SGD,RMSprop,Adam,Adagrad
from keras.layers.convolutional import Conv2D, MaxPooling2D

data_path='/home/drayanserver/Desktop/Generated_Images' # Path for generated images
data_dir_list = ['0','1']
img_data_list=[]
labels = []
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        # img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(128,128))
        labels.append(dataset)
        img_data_list.append(img)
label=np.array(labels)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape

plt.imshow(img_data[30], cmap = 'gray')
plt.show

num_classes = 2 # Number of classes in your dataset
Y = np_utils.to_categorical(label, num_classes)
x,y = shuffle(data,Y, random_state=42)

from keras.layers import LeakyReLU
from keras.layers import *
def PSFAN(input_1,num_classes):

  input = Input(shape=(128, 128, 3), name='input')

  conv1 = Conv2D(32,(3,3), dilation_rate = 1,activation = LeakyReLU(alpha=0.02))(input)
  x = Conv2D(32,(1,1), padding = 'same')(conv1)
  x = Conv2D(32,(3,3), padding = 'same')(x)
  x = Conv2D(32,(1,1), padding = 'same')(x)
  a = Activation('sigmoid')(x)
  attention = Multiply()([a, conv1])
  add = Add()([x, attention])
  M = MaxPooling2D(pool_size = (3,3), padding = 'valid')(add)
  F = Flatten()(M)

  conv2 = Conv2D(32,(3,3), dilation_rate = 2,activation = LeakyReLU(alpha=0.02))(conv1)
  x = Conv2D(32,(1,1), padding = 'same')(conv2)
  x = Conv2D(32,(3,3), padding = 'same')(x)
  x = Conv2D(32,(1,1), padding = 'same')(x)
  a1 = Activation('sigmoid')(x)
  attention1 = Multiply()([a1, conv2])
  add1 = Add()([x, attention1])
  M1 = MaxPooling2D(pool_size = (3,3),padding = 'valid')(add1)
  F1 = Flatten()(M1)

  conv3 = Conv2D(64,(3,3), dilation_rate = 3,activation = LeakyReLU(alpha=0.02))(conv2)
  x = Conv2D(64,(1,1), padding = 'same')(conv3)
  x = Conv2D(64,(3,3), padding = 'same')(x)
  x = Conv2D(64,(1,1), padding = 'same')(x)
  a2 = Activation('sigmoid')(x)
  attention2 = Multiply()([a2, conv3])
  add2 = Add()([x, attention2])
  M2 = MaxPooling2D(pool_size = (3,3),padding = 'valid')(add2)
  F2 = Flatten()(M2)

  conv4 = Conv2D(64,(3,3), dilation_rate = 4,activation = LeakyReLU(alpha=0.02))(conv3)
  x = Conv2D(64,(1,1), padding = 'same')(conv4)
  x = Conv2D(64,(3,3), padding = 'same')(x)
  x = Conv2D(64,(1,1), padding = 'same')(x)
  a3 = Activation('sigmoid')(x)
  attention3 = Multiply()([a3, conv4])
  add3 = Add()([x, attention3])
  M3 = MaxPooling2D(pool_size = (3,3),padding = 'valid')(add3)
  F3 = Flatten()(M3)

  C = Concatenate()([F,F1,F2,F3])

  D2  = Dense(128, activation = LeakyReLU(alpha=0.02))(C)
  D2  = Dense(64, activation = LeakyReLU(alpha=0.02))(D2)

  OUT  = Dense(num_classes, activation='softmax')(D2)
  model = Model(inputs= [input], outputs= OUT, name="BaseModel")
  return model

from sklearn.model_selection import StratifiedKFold

# Initialize your model
model = PSFAN()
model.summary()

# Initialize 10-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store the performance metrics from each fold
accuracy_scores = []
loss_scores = []

# Cross-validation loop
for train_index, test_index in kfold.split(x, y):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Compile and train the CNN model on the training data
    model.compile(optimizer= Adam(lr=0.0001, decay = 0.0006), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    accuracy_scores.append(accuracy)
    loss_scores.append(loss)

# Calculate and display the average performance across all folds
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
mean_loss = sum(loss_scores) / len(loss_scores)

print(f"Mean Accuracy: {mean_accuracy:.2f}")
print(f"Mean Loss: {mean_loss:.2f}")
