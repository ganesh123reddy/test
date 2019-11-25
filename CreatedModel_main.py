#!/usr/bin/env python
# coding: utf-8

# In[14]:


# import os
# import tensorflow
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# if tensorflow.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


# # test_eval = model.evaluate(X_test, Y_test, verbose=0)
# # print('Test loss:', test_eval[0])
# # print('Test accuracy:', test_eval[1])


# In[2]:


import numpy as np
import scipy.io as scio
import scipy.ndimage as im
import imageio
import matplotlib.pyplot as plt
# import tensorflow


# In[3]:


from tensorflow.keras import models,layers,optimizers,applications
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, RepeatVector, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, ZeroPadding2D, UpSampling2D, LSTM, Bidirectional
from tensorflow.keras.layers import Activation, Reshape, Add, Multiply, Lambda, AveragePooling2D, TimeDistributed
from tensorflow.keras.layers import MaxPooling2D, Dropout, Input, MaxPool2D
from tensorflow.keras.optimizers import SGD, Adam, Nadam, RMSprop, Adagrad
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import PReLU, LeakyReLU
from tensorflow.keras.activations import linear as linear_activation
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical


# In[4]:


from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[5]:


def VGGNET(shape):
    vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=shape)
    for layers in vgg_model.layers:
        layers.trainable = False
    model = Sequential()
    for layer in tuple(vgg_model.layers[:-5]):
        layer_type = type(layer).__name__
        model.add(layer)
    return model


# In[6]:


def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
    if output_channels is None:
        output_channels = input.get_shape()[-1]
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = Add()([x, input])
    return x
  
def attention_block(input, input_channels=None, output_channels=None, encoder_depth=1):
    
    p = 1
    t = 2
    r = 1

    if input_channels is None:
        input_channels = input.get_shape()[-1]
    if output_channels is None:
        output_channels = input_channels

    # First Residual Block
    for i in range(p):
        input = residual_block(input)

    # Trunc Branch
    output_trunk = input
    for i in range(t):
        output_trunk = residual_block(output_trunk)

    # Soft Mask Branch
   
    ## encoder
    ### first down sampling
    output_soft_mask = MaxPool2D(padding='same')(input)  
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)

    skip_connections = []
    for i in range(encoder_depth - 1):
        ## skip connections
        output_skip_connection = residual_block(output_soft_mask)
        skip_connections.append(output_skip_connection)
        # print ('skip shape:', output_skip_connection.get_shape())

        ## down sampling
        output_soft_mask = MaxPool2D(padding='same')(output_soft_mask)
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)

    skip_connections = list(reversed(skip_connections))
    for i in range(encoder_depth - 1):
        for _ in range(r):
            output_soft_mask = residual_block(output_soft_mask)
        output_soft_mask = UpSampling2D()(output_soft_mask)
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])
    
    for i in range(r):
        output_soft_mask = residual_block(output_soft_mask)
    output_soft_mask = UpSampling2D()(output_soft_mask)

    ## Output
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)
    
    # Attention: (1 + output_soft_mask) * output_trunk
    output = Lambda(lambda x: x + 1)(output_soft_mask)
    
    output = Multiply()([output, output_trunk])  #

    # Last Residual Block
    for i in range(p):
        output = residual_block(output)
    
    return output 
  
def ResAttentionNet56(shape=(512,512,5), n_channels=64, n_classes=21,dropout=0.3):
    input_ = Input(shape=shape)
    x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(input_) 
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  
    
    x = residual_block(x, output_channels=n_channels * 4)  
    x = attention_block(x, encoder_depth=3)  

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  
    x = attention_block(x, encoder_depth=2)  

    # x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    # x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    x = residual_block(x, output_channels=n_channels * 32)

    pool_size = (x.get_shape()[1], x.get_shape()[2])
    x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    # x = Flatten()(x)
    if dropout>0:
        x = Dropout(dropout)(x)
#     output = Dense(n_classes, activation='sigmoid')(x)
    
    model = Model(input_, x)
    # print(x.get_shape()[1].value, x.get_shape()[2].value,x.shape()[0].value)
    return model


# In[7]:


# model1 = myModel()
x=ResAttentionNet56()
n_classes = 21
# print(x[1],x[2],x[3])
# vgg_model = VGG19(include_top=False, weights='imagenet',input_shape = (32,32,3))
# for layers in vgg_model.layers:
    # layers.trainable = False
model = Sequential()
model.add(x)
# print(x.get_shape()[1].value, x.get_shape()[2].value,x.shape()[0].value)
model.add(Reshape((1,2048)))
model.add(LSTM(50, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
)
# model.add(Reshape((1,50)))
# model.add(LSTM(50, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
# )
# model.add(Reshape((1,50)))
# model.add(LSTM(21, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
# )

model.add(Dense(3,activation='sigmoid'))
# for layer in tuple(vgg_model.layers[:-5]):
    # layer_type = type(layer).__name__
    # model.add(layer)
# model.add(ResAttentionNet56())
op = Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
# model1.build()
model.summary()


# In[19]:


adm = Adam()
model2.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=adm,metrics=['accuracy'])


# In[20]:


model2.summary()


# In[8]:


OBSERVATIONS_FILE  = 'UcmImages1.npy' # The file containing the data samples.
LABELS_FILE        = 'UcmLabels1.npy' # The file containing the labels.
TESTING_DATA_NUM = 500

images = np.load(OBSERVATIONS_FILE)
labels = np.load(LABELS_FILE)
print(images.shape)


# In[9]:


images = np.reshape(images,(576,512,512,5))
print(labels)
print(images.shape,labels.shape)
labels = labels[:,1:]


# In[10]:


print(labels.shape)
c1=0
c2=0
c3=0
idx1 = []
idx2 = []
idx3 = []
img1 = []
img2 = []
img3 = []
l1 = []
l2 = []
l3 = []
for i in range(len(labels)-3):
    if int(labels[i,0]) == 1:
        c1 = c1 + 1
        idx1.append(i)
        l1.append(labels[i,])
        img1.append(images[i,])
    elif int(labels[i,1]) == 1:
        c2 = c2 + 1
        idx2.append(i)
        l2.append(labels[i,])
        img2.append(images[i,])
    else:
        c3 = c3 + 1
        idx3.append(i)
        l3.append(labels[i,])
        img3.append(images[i,])
print(len(idx1),len(idx2),len(idx3))        
min_data = min(c1,c2,c3)
img1 = np.array(img1)
img2 = np.array(img2)
img3 = np.array(img3)
l1 = np.array(l1)
l2 = np.array(l2)
l3 = np.array(l3)


# In[11]:


from random import shuffle
images = np.concatenate((img1[:140,:],img2[:140,:],img3[:140,:]),axis=0)
labels = np.concatenate((l1[:140,:],l2[:140,:],l3[:140,:]),axis = 0)
print(images.shape)
print(labels.shape)


# In[ ]:


def CA_VGG_LSTM(shape):
    model = Sequential()
    model.add(Input(shape=shape))

    model.add(Conv2D(512, (3, 3), activation='relu', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))

    model.add(Conv2D(21, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='glorot_uniform'))
    model.add(Reshape((21, 30*30), input_shape=(30, 30, 21)))

    model.add(LSTM(21, input_shape=(21, 30*30), activation='tanh', kernel_initializer=initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)))

    model.add(Dense(21, activation='softmax'))

    return model


# In[ ]:


model = CA_VGG_LSTM((32,32,512))
adm = Adam(lr = 0.1)
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=adm,metrics=['accuracy'])
print(model.summary())


# In[ ]:





# In[12]:


X_train,X_test,Y_train,Y_test = train_test_split(images,
                                                 labels,
                                                 test_size=0.3,
                                                 random_state=42)


# In[13]:


print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[ ]:


train = model.fit(X_train, Y_train, batch_size = 32, epochs = 20, verbose=1)


# In[ ]:


test_eval = model2.evaluate(X_test, Y_test, batch_size = 1,verbose = 1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# In[ ]:


img1 = X_test
label = Y_test
predicted_classes = model.predict(img1)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
print(predicted_classes)
print(label)
correct = np.where(predicted_classes==Y_test)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], Y_test[correct]))
    plt.tight_layout()


# In[7]:


fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(256,256,3)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))           
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(21, activation='softmax'))


# In[8]:


fashion_model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])


# In[9]:


fashion_train_dropout = fashion_model.fit(X_train, Y_train, batch_size=10,epochs=10,verbose=1)


# In[1]:


import tensorflow_datasets


# In[2]:


import tensorflow_hub


# In[ ]:




