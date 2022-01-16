# # -*- coding: utf-8 -*-
 """
 @author: Beatriz Alves

- Used to train all the different Models.
- Different combinations can be done changing paths and slightly altering the functions.
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

from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import EfficientNetB2,EfficientNetB0

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Multiply,Add, Flatten,Dropout
from tensorflow.keras.layers import concatenate, Input

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from random import randrange

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import pydot
import graphviz
import math

def BAAC_calc(scores):
    baac=0 
    for i in range(8):
        baac=baac+scores[i]
    baac=baac/8
    
    return baac
############   FUNCTIONS   ############

# Function: Learning Rate schedule.
#
#          Reduces learning rate automaticamente. 
#          Chamada a cada epoch atraveés do callback
def lr_schedule(epoch):
    no_epochs = 50
    lr=10**-5
    if epoch < int(0.5*no_epochs):
        return lr
    elif epoch < int(0.75*no_epochs):
        return lr/10
    else:
        return lr/100
# Function: rotate_image
#
#       Roda a imagem em -90, 0, 90 ou 180 graus
def rotate_image(image):
    return np.rot90(image, np.random.choice([-1, 0, 1, 2]))


def concat_model(no_classes):

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


def mult_model(no_classes,layer):

    ## Image part ##    
    resnet = ResNet101(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    inp = Input((224,224,3))
    x = resnet(inp)
    first_dense = layers.GlobalAveragePooling2D()(x)

    ## Metadata part ##
    second_input = Input(shape=(28)) 
    second_dense = Dense(layer,activation='relu')(second_input)
    
    ## Fusion part ##
    merge = Multiply()([first_dense, second_dense])
    
    softmax= Dense(no_classes,activation='softmax')(merge)
    model = Model(inputs=[inp, second_input], outputs=softmax)
        
    return model

def model_4(no_classes,p,metadata_neuron):
    
    no_neurons=(metadata_neuron/(1-p))-metadata_neuron
    
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    inp = Input((224,224,3))
    x = resnet(inp)
    # x1=Flatten()(x)   #teste19 e 19.1
    x1 = layers.GlobalAveragePooling2D()(x)    #teste19.2
    first_dense = Dense(no_neurons,activation='relu')(x1)
    drop= Dropout(0.5)(first_dense)
    ## Metadata part ##
    second_input = Input(shape=(28)) 
    second_dense = Dense(metadata_neuron,activation='relu')(second_input)  #teste 19.1 e 19.2
    
    ## Fusion part ##
    merge = concatenate([drop, second_dense])
    # merge = concatenate([drop, second_input])
    
    softmax= Dense(no_classes,activation='softmax')(merge)
    model = Model(inputs=[inp, second_input], outputs=softmax)
        
    return model
#########################


class DataGenerator_train(Sequence):
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
        auxImg = (cv2.imread("model_c_custom/train/"+ ID,cv2.IMREAD_COLOR))
        
        u = random.uniform(0, 1)
        if u >= 0.5:
           #vertical flip
           auxImg= cv2.flip(auxImg, 0)
                
        v = random.uniform(0, 1) 
        if v>= 0.5:
           #horizontal flip
           auxImg= cv2.flip(auxImg, 1)
          
        rot = randrange(4)   ##0,1 or 2
        #0 -> no rotation
        #1 -> 90 rotation
        #2 -> 180 rotation
        #3 -> 270 rotation

        if(rot==1):
             auxImg= cv2.rotate(auxImg, cv2.ROTATE_90_CLOCKWISE)     #90 graus
        elif(rot==2):
             auxImg= cv2.rotate(auxImg, cv2.ROTATE_180)  #180 graus
        elif(rot==3):
             auxImg= cv2.rotate(auxImg, cv2.ROTATE_90_COUNTERCLOCKWISE)  #270 graus

        # auxImg = np.array(auxImg, dtype=np.uint8)
        auxImg = cv2.cvtColor(auxImg,cv2.COLOR_BGR2RGB)      
        auxImg=auxImg/255
        
        X[i,] = np.asarray(auxImg)
        # Store class
        y[i] = self.labels[ID]
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


class DataGenerator_val(Sequence):
    
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
        auxImg = (cv2.imread("model_c_custom/val/"+ ID,cv2.IMREAD_COLOR))

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
# image = cv2.imread('./train/ISIC_0000000.jpg')
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# plt.imshow(image)
# plt.show()


############  Training pre-processing ##########

no_classes=6

df = pd.read_csv("model_c_6_lesions_train.csv")


partition = {"train": df["image"].tolist()}
labels = dict(zip(df["image"], df["label"]))

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
params = {"dim": (224,224),
 "batch_size": 16,
 "n_classes": no_classes,
 "n_channels": 3,
 "shuffle": True}

training_generator = DataGenerator_train(partition["train"], labels, metadata, **params)

# x, y = training_generator[0]

# x,y = training_generator.__getitem__(5)
# for i in range(32):
#     img=x[0][i]
#     plt.imshow(img)
#     plt.show()

############  Validation pre-processing ##########

df = pd.read_csv("model_c_6_lesions_val.csv")


partition = {"val": df["image"].tolist()}
labels = dict(zip(df["image"], df["label"]))
true_labels=labels.values()

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

params = {"dim": (224,224),
 "batch_size": 16,
 "n_classes": no_classes,
 "n_channels": 3,
 "shuffle": False}

val_generator = DataGenerator_val(partition["val"], labels, metadata, **params)

# x,y = val_generator.__getitem__(5)
# for i in range(32):
#     img=x[0][i]
#     plt.imshow(img)
#     plt.show()


############   IMAGES + METADATA  MODEL ########## 

# layer=2048


#### Model 1 #### (teste 14)
# model=concat_model(no_classes)

#### Model 2 #### gotta change between relu(teste17) and sigmoid(teste17_1)
# model=mult_model(no_classes,layer)

#### Model 3 #### 
# model=model_3(no_classes,layer,prob)

#### Model 4 #### (teste19)
prob=0.6
metadata_neurons=200
model=model_4(no_classes,prob,metadata_neurons)

# tf.keras.utils.plot_model(model, to_file="Mixed_model_teste19.png",show_shapes=True)
# 
### training variables #### 
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

 
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./check/ResNet50_model_c_6lesions_23.h5', save_weights_only=True,
                                                save_best_only=True,monitor='val_loss',
                                                mode='min',verbose=1) 
early_stop= tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', patience=15)

class_weights = {0: 20262./693,
                1: 20262./2658,
                2: 20262./2099,
                3: 20262./191,
                4: 20262./3617,
                5: 20262./10300,
                6: 20262./502,
                7: 20262./202}  
        
class_weight_a = {0: 20262./13917,
                  1: 20262./6345}    

class_weight_b = {0: 13917./3617,
                  1: 13917./10300}    

# class_weight_c = {0: 6345./2492,
#                   1: 6345./3853}     

class_weights_c = {0: 6345./693,
                1: 6345./2658,
                2: 6345./2099,
                3: 6345./191,
                4: 6345./502,
                5: 6345./202}  

class_weight_d = {0: 2492./2099,
                  1: 2492./191,
                  2:  2492./202}  

class_weight_e = {0: 3853./693,
                  1: 3853./2658,
                  2: 3853./502}              
History= model.fit(training_generator, epochs = 50, callbacks=[lr_schedule, checkpoint, early_stop],
                    validation_data=val_generator, class_weight=class_weights_c).history
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.ylim([0,5])
plt.plot(History['loss'])
plt.plot(History['val_loss'])
plt.legend(['train', 'val'], loc='upper left')

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.ylim([0,1])
plt.plot(History["accuracy"])
plt.plot(History["val_accuracy"])
plt.legend(['train', 'val'], loc='lower right')
