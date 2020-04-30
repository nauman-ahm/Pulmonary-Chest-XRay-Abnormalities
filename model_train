# Importing Libraries
import os
import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import keras.initializers
from keras import regularizers
import sklearn.preprocessing
import imageio
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import transform
from skimage import feature
from skimage import exposure
from keras.utils import to_categorical
from scipy.misc import imresize
from keras import optimizers
from glob import glob
from keras import optimizers
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dropout
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score



# Basic input parameters
image_dim = 224
montgomery_path = glob(os.path.join('..','input','Montgomery', 'MontgomerySet','CXR_png', '*.*'))
china_path = glob(os.path.join('..', 'input','ChinaSet_AllFiles', 'ChinaSet_AllFiles','CXR_png', '*.*'))


# Function to check channels and pad each image
def im_pad(temp):
    if(len(temp.shape)==3):
        temp = temp[:,:,0]
    temp = transform.resize(temp,preserve_range = True, output_shape = (image_dim,image_dim)).astype(np.uint8)
    temp = (np.pad(temp, ((math.floor((abs(temp.shape[0] - image_dim))/2),math.ceil((abs(temp.shape[0] - image_dim))/2)),(math.floor((abs(temp.shape[1] - image_dim))/2),math.ceil((abs(temp.shape[1] - image_dim))/2))), 'constant'))
    return(temp)
    
## File read function
def image_read(path, image_dim):
    image_col = np.empty(shape = (0,image_dim,image_dim), dtype = np.uint8)
    target_var = np.empty(shape = (0,1,1), dtype = np.uint8)
    for imagefile in path:
        if (imagefile.endswith('.png')):
            temp = imageio.imread(imagefile)
            temp = im_pad(temp)
            image_col = np.append(image_col, temp.reshape(1,image_dim,image_dim), axis = 0)
            target_var = np.append(target_var, np.array(imagefile[-5]).reshape(1,1,1), axis = 0)
    
    image_col = image_col.reshape(image_col.shape[0],image_dim,image_dim,1)
    target_var = target_var.reshape(target_var.shape[0],1)
    #target_var = to_categorical(target_var)
    return image_col, target_var
        
## Image Gamma Adjustment
def gamma_adjuster(image_set, gamma):
    adjusted = np.empty(shape = (0,image_dim,image_dim), dtype = np.uint8)
    for i in range(0,image_set.shape[0]):
        temp = exposure.adjust_gamma(image_set[i,:,:,0], gamma)
        adjusted = np.append(adjusted, temp.reshape(1,image_dim, image_dim), axis = 0)
    adjusted = adjusted.reshape(adjusted.shape[0],image_dim,image_dim,1)
    return(adjusted)

## Image log Adjustment 
def log_adjuster(image_set, gain):
    adjusted = np.empty(shape = (0,image_dim,image_dim), dtype = np.uint8)
    for i in range(0,image_set.shape[0]):
        temp = exposure.adjust_log(image_set[i,:,:,0], gain = gain)
        adjusted = np.append(adjusted, temp.reshape(1,image_dim, image_dim), axis = 0)
    adjusted = adjusted.reshape(adjusted.shape[0],image_dim,image_dim,1)
    return(adjusted)
    
## Splitter
def image_array_splitter(all_images, all_labels, test_size):
    good_split = False
    while good_split == False:
        test_index = np.random.choice(a = np.arange(0, all_labels.shape[0] -1), size = test_size, replace = False)
        #test_index = np.random.randint(low = 0, high = (all_labels.shape[0])-1, size = test_size, dtype = np.int64)
        ratio = ((all_labels[test_index,1]).sum()/test_size) 
        if ratio >=0.45 and ratio <=0.55:
            good_split = True
    
    train_index = np.setdiff1d(np.arange(0,all_labels.shape[0]), test_index)
    train_X = all_images[train_index,:,:,:]
    train_y = all_labels[train_index,:]
    test_X = all_images[test_index,:,:,:]
    test_y = all_labels[test_index,:]
    print(ratio)
    return train_X, train_y, test_X, test_y

## Image pipeline
image_col_mont, target_var_mont = image_read(montgomery_path, image_dim)
image_col_chn, target_var_chn = image_read(china_path, image_dim)
all_training_images = np.append(image_col_mont, image_col_chn, axis = 0).astype(np.uint8)
all_labels = np.append(target_var_mont, target_var_chn, axis = 0).astype(np.uint8)
#del image_col_mont
#del image_col_chn
#del target_var_mont
#del target_var_chn

gam1 = gamma_adjuster(all_training_images, gamma = 1.5)
gam2 = gamma_adjuster(all_training_images, gamma = 1.75)
gam3 = gamma_adjuster(all_training_images, gamma = 2)
gam4 = gamma_adjuster(all_training_images, gamma = 2.25)
gam5 = gamma_adjuster(all_training_images, gamma = 2.5)
gam6 = gamma_adjuster(all_training_images, gamma = 3)
log1 = log_adjuster(all_training_images, gain = 0.5)
log2 = log_adjuster(all_training_images, gain = 0.75)
log3 = log_adjuster(all_training_images, gain = 1.02)

train_data_X = np.concatenate((gam1, gam2, gam3, gam4, gam5, gam6, all_training_images, log1, log2, log3), axis = 0)
train_data_y = np.concatenate((all_labels,all_labels,all_labels,all_labels,all_labels,all_labels,all_labels,all_labels,all_labels,all_labels), axis = 0)
train_data_y = to_categorical(train_data_y)

train_X, train_y, test_X, test_y = image_array_splitter(all_images = train_data_X, all_labels = train_data_y, test_size = 2000)

## final_model
final_model = Sequential()
final_model.add(Conv2D(input_shape = (224,224,1), strides = 1, filters = 100, kernel_size = (3,3), padding = 'same', kernel_initializer = 'random_uniform'))
final_model.add(MaxPooling2D(pool_size = (3,3)))
final_model.add(Dropout(rate = 0.25))
final_model.add(Conv2D(input_shape = (224,224,1), strides = 1, filters = 80, kernel_size = (3,3), padding = 'same', kernel_initializer = 'random_uniform'))
final_model.add(MaxPooling2D(pool_size = (3,3)))
final_model.add(Dropout(rate = 0.25))
final_model.add(Conv2D(input_shape = (224,224,1), strides = 1, filters = 60, kernel_size = (3,3), padding = 'same', kernel_initializer = 'random_uniform'))
final_model.add(MaxPooling2D(pool_size = (3,3)))
final_model.add(Flatten())
final_model.add(Dropout(rate = 0.25))
final_model.add(Dense(240, activation = 'relu'))
final_model.add(Dropout(rate = 0.25))
final_model.add(Dense(50, activation = 'relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
final_model.add(Dropout(rate = 0.25))
final_model.add(Dense(2, activation = 'softmax', kernel_initializer='random_uniform', bias_initializer='zeros'))
final_model.summary()
custom_optimizer = optimizers.adam(lr = 0.001)
final_model.compile(optimizer = custom_optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
final_model.fit(x = train_X, y = train_y, epochs = 10)

## Metrics
test_predictions = final_model.predict(x = test_X)
test_prediction_classes = test_predictions[:,1]
test_prediction_classes = np.where(test_prediction_classes>0.5,1,0)
## Confusion matrix
confusion_matrix(y_true = test_y[:,1], y_pred = test_prediction_classes)
tn,fp, fn,tp = confusion_matrix(y_true = test_y[:,1], y_pred = test_prediction_classes).ravel()
## Precision 
precision = tp/(tp+fp)
## Recall
recall = tp/(tp+fn)
## F1 Score
f1 = f1_score(y_true = test_y[:,1], y_pred = test_prediction_classes)

## Results
print(precision, recall, f1)
final_model.save('final_model-fulldata.h5')

