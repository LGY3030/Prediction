
# coding: utf-8

# In[1]:


# Multilayer perceptron (Keras)


# In[1]:


import pandas as pd
import numpy as np
from statsmodels.robust.scale import mad
import os
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


def mergeData():
    SaveFile_Name = 'data.csv'
    file_list = os.listdir('data')
    df = pd.read_csv('data'+'\\'+file_list[0])
    df.to_csv(SaveFile_Name,encoding="utf_8_sig",index=False)
    for i in range(1,len(file_list)):
        df = pd.read_csv('data'+'\\'+file_list[i])
        df.to_csv(SaveFile_Name,encoding="utf_8_sig",index=False, header=False, mode='a+')


# In[3]:


def readData():
  train = pd.read_csv("data.csv")
  return train


# In[4]:


def changeYear(data):
    for i in range(0,data.shape[0]):
        Date=data["日期"][i].split('/')
        year,month,date=Date[0],Date[1],Date[2]
        year=str(int(year)+1911)
        data.loc[i,"日期"]=year+'/'+month+'/'+date
    return data


# In[5]:


def augFeatures(data):
  data["日期"] = pd.to_datetime(data["日期"])
  data["年"] = data["日期"].dt.year
  data["月"] = data["日期"].dt.month
  data["日"] = data["日期"].dt.day
  data["第幾日"] = data["日期"].dt.dayofweek
  return data


# In[6]:


def normalizeFunction(x):
    return ((x - np.mean(x)) / (np.max(x) - np.min(x)))


# In[7]:


def normalize(data):
    for i in range(0,data.shape[0]):
        if data["漲跌價差"][i]=='X0.00':
            data.loc[i,"漲跌價差"]=str(int(data["收盤價"][i])-int(data["收盤價"][i-1]))
    data=data.drop(["日期"], axis=1)
    data=data.drop(["成交股數"], axis=1)
    data=data.drop(["成交金額"], axis=1)
    data=data.drop(["漲跌價差"], axis=1)
    data=data.drop(["成交筆數"], axis=1)
    data=data.convert_objects(convert_numeric=True)
    #datanormalize=data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return data


# In[8]:


def buildTrain(train):
    train_x,train_y=[],[]
    for i in range(train.shape[0]-1):
        train_x.append(np.array(train.iloc[i]))
        train_y.append(np.array(train.iloc[i+1]["開盤價"]))
    return np.array(train_x), np.array(train_y)


# In[9]:


def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


# In[10]:


mergeData()
getdata=readData()
changedata=changeYear(getdata)
adddata=augFeatures(changedata)
train=normalize(adddata)
train_x,train_y=buildTrain(train)
train_x,train_y=shuffle(train_x,train_y)
train_y=train_y.reshape(6190,1)
print(train_x.shape)
print(train_y.shape)
#print(train_x)
#print(train_y)


# In[13]:


model=Sequential()
model.add(Dense(units=40,input_dim=train_x.shape[1],kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=30,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units=1,kernel_initializer='uniform',activation='softmax'))
print(model.summary())


# In[16]:


model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])


# In[19]:


train_history=model.fit(x=train_x,y=train_y,validation_split=0.1,epochs=1000,batch_size=30,verbose=2)

