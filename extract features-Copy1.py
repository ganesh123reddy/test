#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[1]:


import numpy as np
from keras.utils.np_utils import to_categorical
import imageio


# In[2]:


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
from tensorflow.keras.layers import PReLU
from tensorflow.keras.activations import linear as linear_activation
from tensorflow.keras import initializers
from tensorflow.keras.utils import to_categorical


# In[3]:


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


# In[4]:


import os
from shutil import copyfile

# copyfile(src, dst)
directory='../DATASETS/DatasetRef/'
dict_clss = {}
dict_clss['Agricluture'] = 1
dict_clss['Forest'] = 2
dict_clss['River'] = 3
l = []
print('Hi')
images = []
labels = []
for dire in os.listdir(directory):
    isDirectory = (os.path.isdir(directory+dire)) and (dire[0] != '.')
    if isDirectory:
        dire = directory + dire
        for subdir in sorted(os.listdir(dire)):
            isDirectory = (os.path.isdir(dire+'/'+subdir))
            if isDirectory:
#                 print(dire,subdir)
                for ssdir in os.listdir(dire+'/'+subdir):
                    isDirectory = (os.path.isdir(dire+'/'+subdir+'/'+ssdir))
                    for files in os.listdir(dire+'/'+subdir+'/'+ssdir):
#                         print(files,(dire.split('/'))[-1])
                        current_image = imageio.imread(dire + '/' + subdir + '/' + ssdir + '/' + files)
                        remove_from_x = current_image.shape[0] - 512
                        remove_from_y = current_image.shape[1] - 512
                        current_image = current_image[remove_from_x:, remove_from_y:]
                        if current_image.shape[0] != 512:
                            pad_x = 512 - current_image.shape[0]
                            z = np.zeros((pad_x,current_image.shape[1],3), dtype=np.int64)
                            current_image = np.concatenate((current_image,z),axis=0)

                        if current_image.shape[1] != 512:
                            pad_y = 512 - current_image.shape[1]
                            p = np.zeros((512,pad_y,3), dtype=np.int64)
                            current_image = np.concatenate((current_image,p),axis=1)
                        images.append(current_image)
                        labels.append(dict_clss[str((dire.split('/'))[-1])])

images = np.array(images, dtype=np.float32)
images = images / 255

print( "The shape of the image samples matrix is {}.\n".format(images.shape) )
            
#                     for files in os.listdir(directory+'/'+dire+'/'+subdir):
#                         count = count + 1
#                         sub = count % 24
#                         if sub == 0:
#                             sub = 24
#                         if not os.path.exists(outputpath+'/'+ dire +'/' + subdir + '/' + str(sub)):
#                             os.mkdir(outputpath+'/'+ dire +'/' + subdir + '/' + str(sub))
#                         copyfile(directory+'/'+dire+'/'+subdir+'/'+files , outputpath+'/'+ dire +'/' + subdir + '/' + str(sub) + '/' + files )

                


# In[5]:


np.array(labels).shape


# In[6]:


# classes = ['Agricultural','Forest','River']
for i in range(len(labels)):
    labels[i] = labels[i]
labels1 = []
labels1 = to_categorical(labels, num_classes = None)
images1 = []
labels2 = []
print(labels1.shape)
images = np.reshape(images,(2880,3,512,512))
for i in range(0,2880,5):
    t=[]
    for j in range(5):
        t.append(np.maximum(images[i+j][0],images[i+j][1],images[i+j][2]))
    images1.append(t)
    labels2.append(labels1[i])
images1 = np.array(images1,dtype = np.float32)
print(np.array(labels2).shape)
# images1 = np.reshape(images1,(2880,512,512,5))
print( "The shape of the image samples matrix is {}.\n".format(images1.shape) )
# np.save("UcmImages.npy", images)
# np.save("UcmLabels.npy", labels1)


# In[7]:


np.save("UcmImages1.npy", images1)
np.save("UcmLabels1.npy", labels2)


# In[ ]:




