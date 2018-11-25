#!/usr/bin/env python
# coding: utf-8

# In[62]:


# Multilayer perceptron (Keras)


# In[1]:


import pandas as pd
import numpy as np
from statsmodels.robust.scale import mad
import os


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


# In[19]:


def normalize(data):
    for i in range(0,data.shape[0]):
        if data["漲跌價差"][i]=='X0.00':
            data.loc[i,"漲跌價差"]=str(int(data["收盤價"][i])-int(data["收盤價"][i-1]))
    data=data.drop(["日期"], axis=1)
    data=data.drop(["成交股數"], axis=1)
    data=data.drop(["成交金額"], axis=1)
    #data=data.drop(["漲跌價差"], axis=1)
    data=data.drop(["成交筆數"], axis=1)
    datanormalize=data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return datanormalize


# In[20]:


mergeData()
getdata=readData()
changedata=changeYear(getdata)
adddata=augFeatures(changedata)
train=normalize(adddata)
print(train)


# In[34]:


train = pd.read_csv("data.csv")
train=train.drop(["日期"], axis=1)
for i in range(0,train.shape[0]):
    if train["漲跌價差"][i]=='X0.00':
        train.loc[i,"漲跌價差"]='-10.00'
train=train.convert_objects(convert_numeric=True)
print(train.dtypes)


# In[ ]:




