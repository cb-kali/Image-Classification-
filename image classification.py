#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, cv2, itertools # cv2 -- OpenCV
import numpy as np 
import pandas as pd 
 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


TRAIN_DIR = './train/'
TEST_DIR = './test1/'
ROWS = 64
COLS = 64
CHANNELS = 3


# In[3]:


train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]


# In[4]:


def read_image(file_path):
  img = cv2.imread(file_path, cv2.IMREAD_COLOR)
  return cv2.resize(img, (ROWS, COLS),interpolation=cv2.INTER_CUBIC)


# In[5]:


def prep_data(images):
  m = len(images)
  n_x = ROWS*COLS*CHANNELS
  
  X = np.ndarray((n_x,m), dtype=np.uint8)
  y = np.zeros((1,m))
  print("X.shape is {}".format(X.shape))
  
  for i,image_file in enumerate(images) :
    image = read_image(image_file)
    X[:,i] = np.squeeze(image.reshape((n_x,1)))
    if 'dog' in image_file.lower() :
      y[0,i] = 1
    elif 'cat' in image_file.lower() :
      y[0,i] = 0
    else : # for test data
      y[0,i] = image_file.split('/')[-1].split('.')[0]
      
    if i%8000 == 0 :
      print("Proceed {} of {}".format(i, m))
    
  return X,y


# In[6]:


X_train, y_train = prep_data(train_images)
X_test, test_idx = prep_data(test_images)


# In[7]:


print("Train shape: {}".format(X_train.shape))
print("Test shape: {}".format(X_test.shape))


# In[8]:


classes = {0: 'cats',
           1: 'dogs'}


# In[9]:


def show_images(X, y, idx) :
  image = X[idx]
  image = image.reshape((ROWS, COLS, CHANNELS))
  plt.figure(figsize=(4,2))
  plt.imshow(image)
  plt.title("This is a {}".format(classes[y[idx,0]]))
  plt.show()


# In[10]:


show_images(X_train.T, y_train.T, 2)


# In[11]:


from sklearn.linear_model import LogisticRegressionCV


# In[12]:


clf = LogisticRegressionCV(solver='liblinear', max_iter=1000)


# In[13]:


X_train_lr, y_train_lr = X_train.T, y_train.T.ravel()


# In[ ]:


clf.fit(X_train_lr, y_train_lr)


# In[ ]:


print("Model accuracy: {:.2f}%".format(clf.score(X_train_lr, y_train_lr)*100))


# In[ ]:




