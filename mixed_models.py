# -*- coding: utf-8 -*-
"""
@author: Beatriz Alves
- Training of the networks already done
- Predicts the data (image+metadata) of the validation dataset using mixed models 1, 2 and 3.
- Code using the original hierarachy. To use the modified hierarchy, some chamges have to be done
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

def second_max(pred):
  line_second_max=[]
  aux= pred.argsort()[:,-2] ## retorna indice onde está o segundo máximo
  for i in range(len(aux)):
    line_second_max.append(pred[i][aux[i]])
  return line_second_max

###para o melhor classifier  E
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


###para o melhor classifier  C
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

###para o melhor classifier  A
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

def ResNet_feature_reducer(no_classes,p,metadata_neuron):
    
    no_neurons=(metadata_neuron/(1-p))-metadata_neuron
    
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
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
def ResNet101_feature_reducer(no_classes,p,metadata_neuron):
    
    no_neurons=(metadata_neuron/(1-p))-metadata_neuron
    
    resnet = ResNet101(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
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
        auxImg=auxImg/255
        
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
 
params = {"dim": (224,224),
 "batch_size": 16,
 "n_classes": 2,
 "n_channels": 3,
 "shuffle": False}

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


model_a=DenseNet_14(2)
model_b=ResNet_feature_reducer(2,0.5,200)
model_c=ResNet_17_1(2)
model_d=ResNet_feature_reducer(3,0.6,200)
model_e=ResNet_17(3)

model_a.load_weights('check/teste14/DenseNet121_model_a_teste14.h5')
model_b.load_weights('check/teste24/ResNet50_model_b_teste24.h5')
model_c.load_weights('check/teste17_1/ResNet50_model_c_teste17_1.h5')
model_d.load_weights('check/teste23/ResNet50_model_d_teste23.h5')
model_e.load_weights('check/teste17/ResNet101_model_e_teste17.h5')



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

line_max_pred_a = np.max(pred_a, axis=1)
# del pred_a

list_pred_melano=[]
list_pred_non_melano=[]

list_pred_melano_imgs=[]
list_pred_melano_meta=[]

list_pred_non_melano_imgs=[]
list_pred_non_melano_meta=[]

list_pred_melano_labels=[]
list_pred_non_melano_labels=[]

mixed_model_a_mel=[]
mixed_model_a_non_mel=[]

a_mel_indices=[]
a_non_mel_indices=[]

#sorts images into lists according to label of model_a, mel(0) vs non_melanocytic(1)
for i in range(5069):
    if(y_pred_a[i]==0):  #melano
        list_pred_melano_imgs.append((x_imgs[i]))
        list_pred_melano_meta.append((x_meta[i]))
        list_pred_melano_labels.append(labels[i])
        mixed_model_a_mel.append(filenames[i])
        a_mel_indices.append(i)
    if(y_pred_a[i]==1): #non_melano
        list_pred_non_melano_imgs.append((x_imgs[i]))
        list_pred_non_melano_meta.append((x_meta[i]))
        list_pred_non_melano_labels.append(labels[i])
        mixed_model_a_non_mel.append(filenames[i])
        a_non_mel_indices.append(i)
        
        
        

            
#### MODEL B ####
##### Predicts model_b #######
arrayListImage = np.stack(list_pred_melano_imgs, axis=0)
arrayListmeta = np.stack(list_pred_melano_meta, axis=0)

list_pred_melano.append(arrayListImage)
list_pred_melano.append(arrayListmeta)

pred_b=model_b.predict(list_pred_melano)
y_pred_b = np.argmax(pred_b, axis=1)

line_max_pred_b = np.max(pred_b, axis=1)
len_b=len(pred_b)

del list_pred_melano_imgs
del list_pred_melano_meta
del list_pred_melano 
# del pred_b
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

line_max_pred_c= np.max(pred_c, axis=1)
# del pred_c

list_pred_benign=[]
list_pred_benign_imgs=[]
list_pred_benign_meta=[]
list_pred_benign_labels=[]

list_pred_malign=[]
list_pred_malign_imgs=[]
list_pred_malign_meta=[]
list_pred_malign_labels=[]

mixed_model_c_ben=[]
mixed_model_c_mal=[]


c_ben_indices=[]
c_mal_indices=[]
d_in_a_index=[]
e_in_a_index=[]

for i in range(len(y_pred_c)):
    if(y_pred_c[i]==0): #benign
        list_pred_benign_imgs.append(list_pred_non_melano_imgs[i])
        list_pred_benign_meta.append(list_pred_non_melano_meta[i])
        list_pred_benign_labels.append(list_pred_non_melano_labels[i])
        mixed_model_c_ben.append(mixed_model_a_non_mel[i])
        c_ben_indices.append(i)
        d_in_a_index.append(a_non_mel_indices[i])
    if(y_pred_c[i]==1): #malign
        list_pred_malign_imgs.append(list_pred_non_melano_imgs[i])
        list_pred_malign_meta.append(list_pred_non_melano_meta[i])
        list_pred_malign_labels.append(list_pred_non_melano_labels[i])
        mixed_model_c_mal.append(mixed_model_a_non_mel[i])
        c_mal_indices.append(i)
        e_in_a_index.append(a_non_mel_indices[i])


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


# y_pred_all=[]
# y_pred_all.extend(y_pred_b)
# y_pred_all.extend(y_pred_d)
# y_pred_all.extend(y_pred_e)


# true_labels=[]
# true_labels.extend(list_pred_melano_labels)
# true_labels.extend(list_pred_benign_labels)
# true_labels.extend(list_pred_malign_labels)

# image_names=[]
# image_names.extend(mixed_model_a_non_mel)
# image_names.extend(mixed_model_c_ben)
# image_names.extend(mixed_model_c_mal)


# aux=recall_score(true_labels, y_pred_all,average=None)
# hier_baac=BAAC_calc(aux)



# print("BACC=" +"%.3f" %BAAC_calc(recall) )
# print("BACC=" +"%.2f" %BAAC_calc(recall) )


# print(confusion_matrix(true_labels, y_pred_all))
# sns.heatmap(confusion_matrix(true_labels, y_pred_all,normalize='true'), fmt='.1%', cmap='Blues',annot=True)

############################## Flat prediction   ####################
# flat_model=ResNet101_feature_reducer(8,0.8,200)
# flat_model.load_weights('check/teste22/ResNet101_flat_teste22.h5')

# df = pd.read_csv("val_imgs_metadata.csv")

# flat_imagenames=df["image"]

# partition = {"val": df["image"].tolist()}
# labels = dict(zip(df["image"], df["label"]))
# flat_true_labels=list(labels.values())

# metadata=aux=dict(zip(df["image"],zip(df["sex_female"],df["sex_male"],
#                                             df["age_approx_0.0"],df["age_approx_5.0"],
#                                             df["age_approx_10.0"],df["age_approx_15.0"],
#                                             df["age_approx_20.0"],df["age_approx_25.0"],
#                                             df["age_approx_30.0"],df["age_approx_35.0"],
#                                             df["age_approx_40.0"],df["age_approx_45.0"],
#                                             df["age_approx_50.0"],df["age_approx_55.0"],
#                                             df["age_approx_60.0"],df["age_approx_65.0"],
#                                             df["age_approx_70.0"],df["age_approx_75.0"],
#                                             df["age_approx_80.0"],df["age_approx_85.0"],
#                                             df["anatom_site_general_anterior torso"],
#                                             df["anatom_site_general_head/neck"],
#                                             df["anatom_site_general_lateral torso"],
#                                             df["anatom_site_general_lower extremity"],
#                                             df["anatom_site_general_oral/genital"],
#                                             df["anatom_site_general_palms/soles"],
#                                             df["anatom_site_general_posterior torso"],
#                                             df["anatom_site_general_upper extremity"],
#                                             )))
# params = {"dim": (224,224),
#  "batch_size": 16,
#  "n_classes": 8,
#  "n_channels": 3,
#  "shuffle": False}

# val_generator = DataGenerator_val_a(partition["val"], labels, metadata, **params)
    
# pred_flat=flat_model.predict(val_generator)
# y_pred_flat = np.argmax(pred_flat, axis=1)


# line_max_flat = np.max(pred_flat, axis=1)

# line_max_pred_d = np.max(pred_d, axis=1)
# line_max_pred_e = np.max(pred_e, axis=1)

# all_probs=[]
# all_probs.extend(line_max_pred_b)
# all_probs.extend(line_max_pred_d)
# all_probs.extend(line_max_pred_e)     
      
# aux=recall_score(flat_true_labels,y_pred_flat[:5069], average=None)
# flat_baac=BAAC_calc(aux)

############################## Mixed prediction Model 1  ####################

# y_mixed_2=[]
# a_counter=0
# b_counter=0
# c_counter=0
# d_counter=0
# e_counter=0
# baac=[]
# # no_flat_pred=0

# a_index=[]
# a_index.extend(a_mel_indices)
# a_index.extend(d_in_a_index)
# a_index.extend(e_in_a_index)

# c_index=[]
# c_index.extend(c_ben_indices)
# c_index.extend(c_mal_indices)

# for i in range(5069):
#   for j in range(5069):
#     if(image_names[i]==flat_imagenames[j]):
#       if(line_max_pred_a[a_index[i]]>=line_max_flat[j]):
#         if(i<len_b):
#           if(line_max_pred_b[i]>=line_max_flat[j]):
#             y_mixed_2.append(y_pred_all[i])
#           else :  
#             b_counter=b_counter+1
#             y_mixed_2.append(y_pred_flat[j])
#         elif(line_max_pred_c[c_index[i-len_b]]>=line_max_flat[j]):
#             if(i<(len_b+len(line_max_pred_d))):
#                if(all_probs[i]>line_max_flat[j]):
#                   y_mixed_2.append(y_pred_all[i])
#                else:
#                   d_counter=d_counter+1
#                   y_mixed_2.append(y_pred_flat[j])
#             elif(i<5069):
#                   if(all_probs[i]>=line_max_flat[j]):
#                     y_mixed_2.append(y_pred_all[i])
#                   else:
#                     e_counter=e_counter+1
#                     y_mixed_2.append(y_pred_flat[j])
#         else:  
#           c_counter=c_counter+1
#           y_mixed_2.append(y_pred_flat[j])
#       else: 
#         a_counter=a_counter+1
#         y_mixed_2.append(y_pred_flat[j])
 
# recall=recall_score(true_labels, y_mixed_2,average=None)
# baac.append(BAAC_calc(scores=recall))
# aux=a_counter+b_counter+c_counter+d_counter+e_counter
# no_flat_pred.append(aux)

# print(baac)
# print(no_flat_pred)
############################## Mixed prediction Model 2  ####################

# a=list(range(90,100,5))   #(0.4,0.45,0.5,...,0.95)
# thresholds = [x / 100 for x in a]
# # thresholds=0.65

# a_index=[]
# a_index.extend(a_mel_indices)
# a_index.extend(d_in_a_index)
# a_index.extend(e_in_a_index)

# c_index=[]
# c_index.extend(c_ben_indices)
# c_index.extend(c_mal_indices)

# baac=[]
# no_flat_pred=[]
# for k in thresholds:
#   a_counter=0
#   b_counter=0
#   c_counter=0
#   d_counter=0
#   e_counter=0
#   y_mixed=[]

#   for i in range(5069):
#     for j in range(5069):
#       if(image_names[i]==flat_imagenames[j]):
#         if(line_max_pred_a[a_index[i]]>k):
#           if(i<len_b):
#             if(all_probs[i]>k):
#               y_mixed.append(y_pred_all[i])
#             else:
#               b_counter=b_counter+1
#               y_mixed.append(y_pred_flat[j])
#         ##caso d e e, a passar no c 
#           elif(i>=len_b): 
#               if(line_max_pred_c[c_index[i-len_b]]>k):
#                   if(i<(len_b+len(line_max_pred_d))):
#                       if(all_probs[i]>k):
#                         y_mixed.append(y_pred_all[i])
#                       else:
#                         d_counter=d_counter+1
#                         y_mixed.append(y_pred_flat[j])
#                   elif(i<5069):
#                       if(all_probs[i]>k):
#                         y_mixed.append(y_pred_all[i])
#                       else:
#                         e_counter=e_counter+1
#                         y_mixed.append(y_pred_flat[j])

#               else:  
#                   c_counter=c_counter+1
#                   y_mixed.append(y_pred_flat[j])

#         else: 
#           a_counter=a_counter+1
#           y_mixed.append(y_pred_flat[j])


#   recall=recall_score(true_labels, y_mixed,average=None)
#   baac.append(BAAC_calc(scores=recall))
#   print(baac)
#   aux=a_counter+b_counter+c_counter+d_counter+e_counter
#   no_flat_pred.append(aux)
#   print(confusion_matrix(true_labels, y_mixed))
#   sns.heatmap(confusion_matrix(true_labels, y_mixed,normalize='true'), fmt='.1%', cmap='Blues',annot=True)
  
############################## Mixed prediction Model 3 ####################
# pred_flat_second=second_max(pred_flat)
# pred_a_second=np.min(pred_a, axis=1)
# pred_b_second=np.min(pred_b, axis=1)
# pred_c_second=np.min(pred_c, axis=1)
# pred_d_second=second_max(pred_d)
# pred_e_second=second_max(pred_e)

# a_index=[]
# a_index.extend(a_mel_indices)
# a_index.extend(d_in_a_index)
# a_index.extend(e_in_a_index)

# c_index=[]
# c_index.extend(c_ben_indices)
# c_index.extend(c_mal_indices)

# all_second_max=[]
# all_second_max.extend(pred_b_second)
# all_second_max.extend(pred_d_second)
# all_second_max.extend(pred_e_second)

# a=list(range(90, 100,5))   #(0.4,0.45,0.5,...,0.95)
# thresholds = [x / 100 for x in a]

# baac=[]
# counter=0
# no_flat_pred=[]
# for k in thresholds:
#   a_counter=0
#   b_counter=0
#   c_counter=0
#   d_counter=0
#   e_counter=0
#   y_mixed=[]

#   for i in range(5069):
#     for j in range(5069):
#       if(image_names[i]==flat_imagenames[j]):
#         if(abs(line_max_pred_a[a_index[i]]-pred_a_second[a_index[i]])>k):
#           if(i<len_b and abs(all_probs[i]-all_second_max[i])>k):
#               y_mixed.append(y_pred_all[i])
#         ##caso d e e, a passar no c 
#           elif(i>=len_b): 
#               if(abs(line_max_pred_c[c_index[i-len_b]]-pred_c_second[c_index[i-len_b]])>k):
#                   if(i<(len_b+len(line_max_pred_d))):
#                       if(abs(all_probs[i]-all_second_max[i])>k):
#                         y_mixed.append(y_pred_all[i])
#                       else:
#                         d_counter=d_counter+1
#                         y_mixed.append(y_pred_flat[j])

#                   elif(i<5069):
#                       if(abs(all_probs[i]-all_second_max[i])>k):
#                         y_mixed.append(y_pred_all[i])
#                       else:
#                         e_counter=e_counter+1
#                         y_mixed.append(y_pred_flat[j])
#               else:  
#                   c_counter=c_counter+1
#                   y_mixed.append(y_pred_flat[j])
#           else:  
#                 b_counter=b_counter+1
#                 y_mixed.append(y_pred_flat[j])
#         else: 
#           a_counter=a_counter+1
#           y_mixed.append(y_pred_flat[j])

#   recall=recall_score(true_labels, y_mixed,average=None)
#   baac.append(BAAC_calc(scores=recall))
#   print(baac)
#   # aux=a_counter+b_counter+c_counter+d_counter+e_counter
#   # no_flat_pred.append(aux)
#   print(confusion_matrix(true_labels, y_mixed))
#   sns.heatmap(confusion_matrix(true_labels, y_mixed,normalize='true'), fmt='.1%', cmap='Blues',annot=True)
  

############################################## PLots for mixed models 2 and 3 ##############################
# plt.plot(thresholds,baac,'-o',color='blue')
# plt.axhline(flat_baac,color ="red")
# plt.axhline(hier_baac,color ="green")
# plt.ylabel("BACC")
# plt.xlabel("Threshold")
# plt.legend(['mixed_3', 'flat','hier'], loc='lower right')
# plt.grid(True)
# plt.xlim([0,1])
# plt.xlim([0.4,0.9])

# plt.figure()
# plt.plot(thresholds,no_flat_pred,'-o',color='blue')
# plt.ylabel("Number of cases in flat")
# plt.xlabel("Threshold")




