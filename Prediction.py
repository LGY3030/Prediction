
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
    datanormalize=data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return datanormalize


# In[8]:


def predictAns(data):
    for i in range(0,data.shape[0]-1):
        data.loc[i,"結果"]=data["開盤價"][i+1]
    data=data.drop([data.shape[0]-1])
    return data


# In[12]:


mergeData()
getdata=readData()
changedata=changeYear(getdata)
adddata=augFeatures(changedata)
addans=predictAns(adddata)
train_x=addans
train_x=train_x.drop(["結果"], axis=1)
train_x=normalize(train_x)
train_x=train_x.as_matrix()
train_y=addans["結果"]
train_y=train_y.as_matrix()


# In[15]:


model=Sequential()
model.add(Dense(units=256,input_dim=train_x.shape[1],kernel_initializer='normal',activation='relu'))
model.add(Dense(units=1,kernel_initializer='normal',activation='softmax'))
print(model.summary())


# In[20]:


model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[21]:


train_history=model.fit(x=train_x,y=train_y,validation_split=0.2,epochs=10,batch_size=200,verbose=2)

