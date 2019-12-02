#!/usr/bin/env python
# coding: utf-8

# ## 데이터 로드

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from keras.datasets import mnist
from keras.utils import to_categorical
import pandas as pd
import os


# In[3]:


#print (os.path.realpath("./"))
data_dir = 'nsynth-train'
data = pd.read_json(os.path.join(data_dir, 'data.json'))
data = data.T
data = data.reset_index()
data.rename(columns={'index':'filename'}, inplace='True')
#data.velocity = data.velocity.map({25:0, 50:1, 75:2, 100:3, 127:4})
#print (data.note.nunique())


# In[4]:


# wav to numpy
from scipy.io.wavfile import read
wav_list = []
velocity_list = []
pitch_list = []
note_list = []
label_list = []
cnt = 0
for index, row in data.iterrows():
    cnt += 1
    wav = read(data_dir+"/audio/{}.wav".format(row['filename'])) 
    wav_list.append(wav[1]) # wav to numpy
    #velocity_list.append(row['velocity'])
    #pitch_list.append(row['pitch'])
    #note_list.append(row['note'])
    #print(row['filename'])
    if cnt > 45001:
        break
label_list = note_list
#label_list


# ## 전처리

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#print(data.head())

genre_list = data['instrument_family_str'] #data.iloc[:, 5]

print(genre_list)
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
y


# In[6]:


import numpy as np

nd = np.array(wav_list[:40000])
#train_waves = nd[:231364]
#test_waves = nd[231364:]

train_waves = nd
test_waves = np.array(wav_list[40000:45000])

#train_test_split(X, y, test_size=0.2)


# In[7]:


#labels = np.array(label_list)
labels = y #np.array(genre_list)
train_labels = labels[:40000]
test_labels = labels[40000:45000]
labels


# In[8]:


print (labels.shape)
print (test_labels.shape)


# In[9]:


train_waves = train_waves.reshape((40000, 64000))
test_waves = test_waves.reshape((5000, 64000))


# In[10]:


train_waves = train_waves.astype('float32') / (2**15)
test_waves = test_waves.astype('float32') / (2**15)


# In[11]:


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[ ]:


print (train_labels.shape)


# ## 네트워크 모델 설계

# In[ ]:


from keras import models
from keras import layers
from keras.utils import multi_gpu_model


# In[ ]:


model = models.Sequential()
model.add(layers.Dense(4096, activation='relu', input_shape=(64000,)))
model.add(layers.Dense(2048, activation='relu', input_shape=(64000,)))
model.add(layers.Dense(1024, activation='relu', input_shape=(64000,)))
model.add(layers.Dense(512, activation='relu', input_shape=(64000,)))
model.add(layers.Dense(128, activation='relu', input_shape=(64000,)))
model.add(layers.Dense(11, activation='softmax'))


# In[ ]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.summary()


# ## 모델 훈련(training)

# In[ ]:


model.fit(train_waves, train_labels, epochs=20, batch_size=1000)


# ## 모델 평가

# In[16]:


from keras.models import load_model
model = load_model('./model_instrument.h5')


# In[ ]:


test_loss, test_acc = model.evaluate(test_waves, test_labels)


# In[ ]:


print('test_acc:', test_acc)


# ## 모델 save

# In[ ]:


model.save('./model_instrument.h5')


# In[ ]:




