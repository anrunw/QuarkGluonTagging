#!/usr/bin/env python
# coding: utf-8

# In[148]:


import json
import pandas as pd
import h5py
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Dropout, LSTM
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
import h5py
import matplotlib.pyplot as plt


# In[1]:


df = pd.read_json('dataset.json', lines=True
                 )


# In[22]:


df


# In[30]:


df.columns = ['j_pt', 'j_eta', 'flavor', 'high_level_track','high_level_vertex','track_variables']


# In[40]:


df.to_hdf('data.h5', key = 'data', mode='w')


# In[41]:


df


# In[110]:


length = len(df)
featuresnum = 16


# In[83]:


features = np.zeros([length,features])


# In[51]:


z = df.values


# In[56]:


z[:,0].shape


# In[70]:


labmatrix = np.zeros([length,1])


# Makes a label matrix

# In[72]:


num = 0
for e in z:
    labmatrix[num] = e[2]
    num = num+1


# In[73]:


labmatrix


# one-hot encodes labels

# In[76]:


lab = np.zeros([length,3])
num = 0
for x in labmatrix:
    if x == 0:
        lab[num][0]= 1
    elif x == 4:
        lab[num][1] = 1
    else:
        lab[num][2] = 1
    num = num + 1


# In[77]:


lab


# Appends all Pt values into matrix

# In[84]:


num = 0
for e in z:
    features[num][0] = e[0]
    num = num+1


# In[85]:


features


# Appends eta values into matrix

# In[87]:


num = 0 
for e in z:
    features[num][1] = e[1]
    num = num+1


# In[88]:


features


# Appends all high level tracking variables

# In[100]:


num = 0

for e in z:
    num2 = 2
    for var in e[3]:
        features[num][num2] = var
        num2 = num2+1
    num = num+1


# In[94]:


features[2]


# Appends all high level vertex variables

# In[95]:


num = 0

for e in z:
    num2 = 10
    for var in e[4]:
        features[num][num2] = var
        num2 = num2+1
    num = num+1


# In[115]:


features[1,0]


# normalizes matrix with SD of 1
# 

# In[111]:


normfeatures = np.zeros([length,featuresnum])


# In[113]:


transpose = features.transpose()


# In[117]:


num = 0
for column in transpose:
    g = (column - np.mean(column))/np.std(column)
    num2 = 0
    for e in g:
        normfeatures[num2,num] = e
        num2 = num2+1
    num = num+1
        


# In[121]:


normfeatures = np.nan_to_num(normfeatures,copy=True)


# In[122]:


normfeatures[0]


# In[124]:


X_train, X_test, y_train, y_test = train_test_split(normfeatures,lab,test_size = 0.2, random_state = 42)


# In[193]:


Inputs = Input(shape=(16,))
x = Dense(128, activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu')(Inputs)
x= Dropout(rate = 0.5)(x)
x = Dense(128, activation='relu', kernel_initializer='lecun_uniform', name = 'fc2')(x)
x= Dropout(rate = 0.5)(x)
x = Dense(128, activation='relu', kernel_initializer='lecun_uniform', name = 'fc3')(x)
x= Dropout(rate = 0.5)(x)
x = Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='fc4')(x)
x= Dropout(rate = 0.5)(x)
x = Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='fc5')(x)
x= Dropout(rate = 0.5)(x)
x = Dense(64, activation='relu', kernel_initializer='lecun_uniform', name='fc6')(x)
x= Dropout(rate = 0.5)(x)
x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc7')(x)
x= Dropout(rate = 0.5)(x)
x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc8')(x)
x= Dropout(rate = 0.5)(x)
x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name='fc9')(x)
predictions = Dense(3, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'output_sigmoid')(x)
model = Model(inputs=Inputs, outputs=predictions)
model.summary()


# In[195]:


adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[196]:


history = model.fit(X_train, y_train, batch_size = 1024, epochs = 10, 
                    validation_split = 0.25, shuffle = True, callbacks = None,
                    use_multiprocessing=True, workers=4)


# In[197]:


def learningCurve(history):
    plt.figure(figsize=(10,8))
    plt.plot(history.history['loss'], linewidth=1)
    plt.plot(history.history['val_loss'], linewidth=1)
    plt.title('Model Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['training sample loss','validation sample loss'])
    #plt.savefig('Learning_curve.pdf')
    plt.show()
    plt.close()


# In[198]:


learningCurve(history)


# In[199]:


labels = ['light', 'charm', 'bottom']


# In[200]:


def makeRoc(features_val, labels_val, labels, model, outputDir='', outputSuffix=''):
    from sklearn.metrics import roc_curve, auc
    labels_pred = model.predict(features_val)
    df = pd.DataFrame()
    fpr = {}
    tpr = {}
    auc1 = {}
    plt.figure(figsize=(10,8))       
    for i, label in enumerate(labels):
        df[label] = labels_val[:,i]
        df[label + '_pred'] = labels_pred[:,i]
        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])
        auc1[label] = auc(fpr[label], tpr[label])
        plt.plot(fpr[label],tpr[label],label='%s tagger, AUC = %.1f%%'%(label.replace('j_',''),auc1[label]*100.))
    plt.plot([0, 1], [0, 1], lw=1, color='black', linestyle='--')
    #plt.semilogy()
    plt.xlabel("Background Efficiency")
    plt.ylabel("Signal Efficiency")
    plt.xlim([-0.05, 1.05])
    plt.ylim(0.001,1.05)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.figtext(0.25, 0.90,'Feed Forward Roc Curve',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    #plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    #plt.savefig('%sROC_%s.pdf'%(outputDir, outputSuffix))
    return labels_pred


# In[201]:


y_pred = makeRoc(X_test, y_test, labels, model, outputSuffix='two-layer')


# In[162]:


images = np.stack([X_train, X_train],axis= -1)


# In[166]:


images.shape


# In[202]:


Inputs = Input(shape=(16,2))
x = LSTM(16, dropout = 0.5, kernel_initializer = 'lecun_uniform', name = 'lstm1')(Inputs)
x = Dense(128, activation='relu', kernel_initializer='lecun_uniform', name='fc1_relu')(x)
x= Dropout(rate = 0.5)(x)
x = Dense(64, activation='relu', kernel_initializer='lecun_uniform', name = 'fc2')(x)
x= Dropout(rate = 0.5)(x)

x = Dense(32, activation='relu', kernel_initializer='lecun_uniform', name = 'fc3')(x)

predictions = Dense(3, activation='sigmoid', kernel_initializer='lecun_uniform', name = 'output_sigmoid')(x)
model = Model(inputs=Inputs, outputs=predictions)
model.summary()


# In[203]:


adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:





# In[204]:


history = model.fit(images, y_train, batch_size = 1024, epochs = 10, 
                    validation_split = 0.25, shuffle = True, callbacks = None,
                    use_multiprocessing=True, workers=4)


# In[170]:


learningCurve(history)


# In[173]:


imagetest = np.stack([X_test, X_test],axis= -1)


# In[174]:


y_pred = makeRoc(imagetest, y_test, labels, model, outputSuffix='two-layer')


# In[ ]:




