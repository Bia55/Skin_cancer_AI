# -*- coding: utf-8 -*-
"""
author: Beatriz Alves

- Training of the networks already done
- Predicts the data (image+metadata) of the validation dataset using the original hierarchical model. 
- Only used in the image + metadata combination methods. Different combinations can be done changing paths and slightly altering the functions
"""


import numpy as np
from numpy import array
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import seaborn as sns
import cv2
import random
import os

from tensorflow.keras.applications.resnet import  ResNet50   
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB2
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Multiply,Add,Flatten, Dropout
from tensorflow.keras.layers import concatenate, Input

import itertools
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from random import randrange
from sklearn.metrics import recall_score
import pydot
import graphviz
import math
def BAAC_calc(scores):
    baac=0 
    for i in range(8):
        baac=baac+scores[i]
    baac=(baac/8)*100
    
    return baac

###ResNet101 multiplication method
def ResNet_17(no_classes):
    resnet = ResNet101(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    inp = Input((224,224,3))
    x = resnet(inp)
    first_dense = layers.GlobalAveragePooling2D()(x)

    ## Metadata part ##
    second_input = Input(shape=(28)) 
    second_dense = Dense(2048,activation='relu')(second_input)

    ## Fusion part ##
    multiplied = Multiply()([first_dense, second_dense])
    softmax= Dense(no_classes,activation='softmax')(multiplied)

    model = Model(inputs=[inp, second_input], outputs=softmax)

    return model


###ResNet50 multiplication method with sigmoid
def ResNet_17_1(no_classes):
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    inp = Input((224,224,3))
    x = resnet(inp)
    first_dense = layers.GlobalAveragePooling2D()(x)

    ## Metadata part ##
    second_input = Input(shape=(28)) 
    second_dense = Dense(2048,activation='sigmoid')(second_input)

    ## Fusion part ##

    multiplied = Multiply()([first_dense, second_dense])
    softmax= Dense(no_classes,activation='softmax')(multiplied)

    model = Model(inputs=[inp, second_input], outputs=softmax)

    return model

### DenseNet Concatenation method
def DenseNet_14(no_classes):

    ## Image part ##    
    resnet = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    inp = Input((224,224,3))
    x = resnet(inp)
    first_dense = layers.GlobalAveragePooling2D()(x)

    ## Metadata part ##
    second_input = Input(shape=(28)) 
    second_dense = Dense(500,activation='relu')(second_input)
    
    ## Fusion part ##
    merge = concatenate([first_dense, second_dense])
    
    softmax= Dense(no_classes,activation='softmax')(merge)
    model = Model(inputs=[inp, second_input], outputs=softmax)
        
    return model

### EfficientNetB2 Concatenation method
def EffB2_14(no_classes):

    ## Image part ##    
    resnet = EfficientNetB2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    inp = Input((224,224,3))
    x = resnet(inp)
    first_dense = layers.GlobalAveragePooling2D()(x)

    ## Metadata part ##
    second_input = Input(shape=(28)) 
    second_dense = Dense(500,activation='relu')(second_input)
    
    ## Fusion part ##
    merge = concatenate([first_dense, second_dense])
    
    softmax= Dense(no_classes,activation='softmax')(merge)
    model = Model(inputs=[inp, second_input], outputs=softmax)
        
    return model


def ResNet_feature_reducer(no_classes,p,metadata_neuron):
    
    no_neurons=(metadata_neuron/(1-p))-metadata_neuron
    
    resnet = EfficientNetB2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    inp = Input((224,224,3))
    x = resnet(inp)
    # x1=Flatten()(x)
    x1 = layers.GlobalAveragePooling2D()(x)
    first_dense = Dense(no_neurons,activation='relu')(x1)
    drop= Dropout(0.5)(first_dense)
    ## Metadata part ##
    second_input = Input(shape=(28)) 
    second_dense = Dense(metadata_neuron,activation='relu')(second_input)
    
    ## Fusion part ##
    merge = concatenate([drop, second_dense])
    # merge = concatenate([drop, second_input])
    
    softmax= Dense(no_classes,activation='softmax')(merge)
    model = Model(inputs=[inp, second_input], outputs=softmax)
        
    return model




# Best individual classifiers of every model.
model_a=DenseNet_14(2)
model_b=EffB2_14(2)
model_c=ResNet_17_1(2)
model_d=ResNet_feature_reducer(3,0.6,200)
model_e=ResNet_17(3)

model_a.load_weights('check/teste14/DenseNet121_model_a_teste14.h5')
model_b.load_weights('check/teste14/EfficientNetB2_model_b_teste14.h5')
model_c.load_weights('check/teste17_1/ResNet50_model_c_teste17_1.h5')
model_d.load_weights('check/teste23/ResNet50_model_d_teste23.h5')
model_e.load_weights('check/teste17/ResNet101_model_e_teste17.h5')



class DataGenerator_val_a(Sequence):
    
    def __init__(self, list_IDs, labels,metadata,batch_size=32, dim=(224,224), n_channels=3,
     n_classes=8, shuffle=True):
         #Initialization
         self.dim = dim
         self.batch_size = batch_size
         self.labels = labels
         self.metadata = metadata
         self.list_IDs = list_IDs
         self.n_channels = n_channels
         self.n_classes = n_classes
         self.shuffle = shuffle
         self.on_epoch_end()
         
    def on_epoch_end(self):
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
        np.random.shuffle(self.indexes)

    def __len__(self):
     return int(math.ceil(len(self.list_IDs) / self.batch_size))
 
    def __data_generation(self, list_IDs_temp):
     #Generates data containing batch_size samples’ # X : (n_samples, *dim, n_channels)
     # Initialization
     X = np.empty((self.batch_size, *self.dim, self.n_channels))
     y = np.empty((self.batch_size), dtype=int)
     xMetadata = np.empty((self.batch_size, 28), dtype=int)
    # Generate data
     for i, ID in enumerate(list_IDs_temp):
     # Store sample
        auxImg = (cv2.imread("model_a_custom/val/"+ ID,cv2.IMREAD_COLOR))
    
        auxImg = cv2.cvtColor(auxImg,cv2.COLOR_BGR2RGB)      
        auxImg=auxImg/255  #Comment in EfficientNet networks
        
        X[i,] = np.array(auxImg)
        # Store class
        y[i] = np.array(self.labels[ID])
        aux_meta=np.array(self.metadata[ID])
        xMetadata[i,] =np.array(aux_meta)
     return [X,xMetadata], tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def __getitem__(self, index):
     #Generate one batch of data’
     # Generate indexes of the batch
     indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    # Find list of IDs
     list_IDs_temp = [self.list_IDs[k] for k in indexes]
    # Generate data
     [X,xMetadata], y = self.__data_generation(list_IDs_temp)
     return [X,xMetadata], y
 
params = {"dim": (224,224),     # Image dimensions
 "batch_size": 16,  
 "n_classes": 2,      # number of classes of the classifier.
 "n_channels": 3,     # number of image channels
 "shuffle": False}     # Order of images is always the same in the validation data.

df = pd.read_csv("model_a_val_imgs_metadata.csv")

partition = {"val": df["image"].tolist()}
labels_a = dict(zip(df["image"], df["label"]))
#true_labels_a=labels.values()

metadata=aux=dict(zip(df["image"],zip(df["sex_female"],df["sex_male"],
                                            df["age_approx_0.0"],df["age_approx_5.0"],
                                            df["age_approx_10.0"],df["age_approx_15.0"],
                                            df["age_approx_20.0"],df["age_approx_25.0"],
                                            df["age_approx_30.0"],df["age_approx_35.0"],
                                            df["age_approx_40.0"],df["age_approx_45.0"],
                                            df["age_approx_50.0"],df["age_approx_55.0"],
                                            df["age_approx_60.0"],df["age_approx_65.0"],
                                            df["age_approx_70.0"],df["age_approx_75.0"],
                                            df["age_approx_80.0"],df["age_approx_85.0"],
                                            df["anatom_site_general_anterior torso"],
                                            df["anatom_site_general_head/neck"],
                                            df["anatom_site_general_lateral torso"],
                                            df["anatom_site_general_lower extremity"],
                                            df["anatom_site_general_oral/genital"],
                                            df["anatom_site_general_palms/soles"],
                                            df["anatom_site_general_posterior torso"],
                                            df["anatom_site_general_upper extremity"],
                                            )))


val_generator_a = DataGenerator_val_a(partition["val"], labels_a, metadata, **params)

################### ler excels b #######################
df1 = pd.read_csv("just_for_custom_hier.csv")

labels=df1["label"]
filenames=df1["image"]

del df1
#### MODEL A ####
pred_a=model_a.predict(val_generator_a)
y_pred_a = np.argmax(pred_a, axis=1)


x_imgs=[]
x_meta=[]
for i in range(317):
    x,y=val_generator_a.__getitem__(i)
    # x_val.extend(x)
    for j in range(16):
        x_imgs.append(x[0][j]) 
        x_meta.append(x[1][j])


del val_generator_a
# del pred_a

list_pred_melano=[]
list_pred_non_melano=[]

list_pred_melano_imgs=[]
list_pred_melano_meta=[]

list_pred_non_melano_imgs=[]
list_pred_non_melano_meta=[]

list_pred_melano_labels=[]
list_pred_non_melano_labels=[]

list_prob_a_melano=[]
list_prob_a_non_melano=[]

#sorts images into lists according to label of model_a, mel(0) vs non_melanocytic(1)
for i in range(5069):
    if(y_pred_a[i]==0):  #melano
        list_pred_melano_imgs.append((x_imgs[i]))
        list_pred_melano_meta.append((x_meta[i]))
        list_pred_melano_labels.append(labels[i])
        list_prob_a_melano.append(pred_a[i])
    if(y_pred_a[i]==1): #non_melano
        list_pred_non_melano_imgs.append((x_imgs[i]))
        list_pred_non_melano_meta.append((x_meta[i]))
        list_pred_non_melano_labels.append(labels[i])
        list_prob_a_non_melano.append(pred_a[i])
        
        
#### MODEL B ####
##### Predicts model_b #######
# list_pred_melano_imgs[:] = [x / 255 for x in list_pred_melano_imgs]
arrayListImage = np.stack(list_pred_melano_imgs, axis=0)
arrayListmeta = np.stack(list_pred_melano_meta, axis=0)

list_pred_melano.append(arrayListImage)
list_pred_melano.append(arrayListmeta)


pred_b=model_b.predict(list_pred_melano)
y_pred_b = np.argmax(pred_b, axis=1)

del list_pred_melano_imgs
del list_pred_melano_meta
del list_pred_melano 
del pred_b
del arrayListImage
del arrayListmeta
#### MODEL C ####
##### Predicts model_c #######
arrayListImage = np.stack(list_pred_non_melano_imgs, axis=0)
arrayListmeta = np.stack(list_pred_non_melano_meta, axis=0)

list_pred_non_melano.append(arrayListImage)
list_pred_non_melano.append(arrayListmeta)

pred_c=model_c.predict(list_pred_non_melano)
y_pred_c = np.argmax(pred_c, axis=1)

# del pred_c

list_pred_benign=[]
list_pred_benign_imgs=[]
list_pred_benign_meta=[]
list_pred_benign_labels=[]

list_pred_malign=[]
list_pred_malign_imgs=[]
list_pred_malign_meta=[]
list_pred_malign_labels=[]

list_prob_a_ben=[]
list_prob_a_mal=[]

list_prob_c_ben=[]
list_prob_c_mal=[]

for i in range(len(y_pred_c)):
    if(y_pred_c[i]==0): #benign
        list_pred_benign_imgs.append(list_pred_non_melano_imgs[i])
        list_pred_benign_meta.append(list_pred_non_melano_meta[i])
        list_pred_benign_labels.append(list_pred_non_melano_labels[i])
        list_prob_a_ben.append(list_prob_non_melano[i])
        list_prob_c_ben.append(pred_c[i])
    if(y_pred_c[i]==1): #malign
        list_pred_malign_imgs.append(list_pred_non_melano_imgs[i])
        list_pred_malign_meta.append(list_pred_non_melano_meta[i])
        list_pred_malign_labels.append(list_pred_non_melano_labels[i])
        list_prob_a_mal.append(list_prob_non_melano[i])
        list_prob_c_ben.append(pred_c[i])

del arrayListImage
del arrayListmeta
del list_pred_non_melano
del list_pred_non_melano_imgs
del list_pred_non_melano_meta
# del list_pred_melano
#### MODEL D ####
##### Predicts model_d #######

arrayListImage = np.stack(list_pred_benign_imgs, axis=0)
arrayListmeta = np.stack(list_pred_benign_meta, axis=0)

list_pred_benign.append(arrayListImage)
list_pred_benign.append(arrayListmeta)

pred_d=model_d.predict(list_pred_benign)
y_pred_d = np.argmax(pred_d, axis=1)

del list_pred_benign_imgs
del list_pred_benign_meta
del arrayListImage
del arrayListmeta
#del pred_d

#### MODEL E ####
##### Predicts model_e #######
arrayListImage = np.stack(list_pred_malign_imgs, axis=0)
arrayListmeta = np.stack(list_pred_malign_meta, axis=0)

list_pred_malign.append(arrayListImage)
list_pred_malign.append(arrayListmeta)

pred_e=model_e.predict(list_pred_malign)
y_pred_e = np.argmax(pred_e, axis=1)

del arrayListImage
del arrayListmeta
### Trasnform y_pred to real classes. NOT TRUE LABELS

#model_b 
replacements = {
    0: 4,
    1: 5,
}
y_pred_b = [replacements.get(x, x) for x in y_pred_b]


#model_d 
replacements = {
    0: 2,
    1: 3,
    2: 7,
}
y_pred_d = [replacements.get(x, x) for x in y_pred_d]

#model_e 
replacements = {
    0: 0,
    1: 1,
    2: 6,
}
y_pred_e = [replacements.get(x, x) for x in y_pred_e]

## Predicted labels of classifiers B, D and E
y_pred_all=[]
y_pred_all.extend(y_pred_b)
y_pred_all.extend(y_pred_d)
y_pred_all.extend(y_pred_e)

## True labels of classifiers B, D and E
true_labels=[]
true_labels.extend(list_pred_melano_labels)
true_labels.extend(list_pred_benign_labels)
true_labels.extend(list_pred_malign_labels)


recall=recall_score(true_labels, y_pred_all,average=None)

print("BACC=" +"%.2f" %BAAC_calc(recall) )

###Flat prediction###

## Function : plots confusion matrixs
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    # plt.figure(figsize=(4,3.5))   
    # plt.figure(figsize=(5,4.5))
    plt.figure(figsize=(15,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.rcParams.update({'font.size': 26})
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh =  cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(i, j, ("{:0.2f}"+"%").format(cm[i, j]*100), fontsize=24,
                     horizontalalignment="center", verticalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('\nPredicted label')
    plt.show()


print(confusion_matrix(true_labels, y_pred_all))
cm=confusion_matrix(true_labels, y_pred_all,normalize='true')


x_axis_labels = ["AK","BCC","BKL","DF","MEL","NV","SCC","VASC"]
plot_confusion_matrix(cm, x_axis_labels) #pretty confusion matrix








