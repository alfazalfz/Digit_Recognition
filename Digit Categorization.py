#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf


# In[2]:


get_ipython().system('pip install tensorflow')


# In[7]:


import matplotlib.pyplot as plt


# In[5]:


import cv2


# In[32]:


import numpy as np


# In[8]:


import os


# In[9]:


mnist = tf.keras.datasets.mnist


# In[10]:


(x_train, y_train), (x_test, y_test) =  mnist.load_data()


# In[11]:


x_train = tf.keras.utils.normalize(x_train, axis=1)


# In[12]:


x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[13]:


model = tf.keras.models.Sequential()


# In[14]:


model.add(tf.keras.layers.Flatten(input_shape=(28,28)))


# In[15]:


model.add(tf.keras.layers.Dense(128, activation='relu'))


# In[16]:


model.add(tf.keras.layers.Dense(128, activation='relu'))


# In[17]:


model.add(tf.keras.layers.Dense(10, activation='softmax'))


# In[18]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[20]:


model.fit(x_train,y_train,epochs=3)


# In[21]:


model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')
# In[22]:


model = tf.keras.models.load_model('handwritten.model')


# In[23]:


loss,accuracy = model.evaluate(x_test,y_test)


# In[24]:


print(loss)


# In[25]:


print(accuracy)


# In[27]:


os.getcwd()


# In[100]:


img = cv2.imread("img.png")[:,:,0]


# In[101]:


img = np.invert(np.array([img]))


# In[102]:


prediction = model.predict(img)


# In[103]:


print(f"this digit is {np.argmax(prediction)}")


# In[104]:


plt.imshow(img[0],cmap=plt.cm.binary)


# In[ ]:




