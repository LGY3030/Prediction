
# coding: utf-8

# In[1]:


# RNN(Long Short-Term Memory, LSTM)


# In[15]:


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def mergeData():
    SaveFile_Name = 'data.csv'
    file_list = os.listdir('data')
    df = pd.read_csv('data'+'\\'+file_list[0])
    df.to_csv(SaveFile_Name,encoding="utf_8_sig",index=False)
    for i in range(1,len(file_list)):
        df = pd.read_csv('data'+'\\'+file_list[i])
        df.to_csv(SaveFile_Name,encoding="utf_8_sig",index=False, header=False, mode='a+')


# In[4]:


def readData():
    train = pd.read_csv("data.csv")
    return train


# In[5]:


def changeYear(data):
    for i in range(0,data.shape[0]):
        Date=data["日期"][i].split('/')
        year,month,date=Date[0],Date[1],Date[2]
        year=str(int(year)+1911)
        data.loc[i,"日期"]=year+'/'+month+'/'+date
    return data


# In[6]:


# Augment Features
def augFeatures(data):
  data["日期"] = pd.to_datetime(data["日期"])
  data["年"] = data["日期"].dt.year
  data["月"] = data["日期"].dt.month
  data["日"] = data["日期"].dt.day
  data["第幾日"] = data["日期"].dt.dayofweek
  return data


# In[7]:


def manage(data):
    for i in range(0,data.shape[0]):
        if data["漲跌價差"][i]=='X0.00':
            data.loc[i,"漲跌價差"]=str(int(data["收盤價"][i])-int(data["收盤價"][i-1]))
    data=data.drop(["日期"], axis=1)
    data=data.drop(["成交股數"], axis=1)
    data=data.drop(["成交金額"], axis=1)
    data=data.drop(["漲跌價差"], axis=1)
    data=data.drop(["成交筆數"], axis=1)
    data=data.convert_objects(convert_numeric=True)
    return data


# In[8]:


def normalize(data):
    datanormalize=data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return datanormalize


# In[52]:


def buildTrain(train, pastDay=1, futureDay=1):
    X_train, Y_train, Z_train= [], [], []
    X,Y,Z=[],[],[]
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["開盤價"]))
        Z_train.append(np.array(train.iloc[i:i+pastDay]["開盤價"]))
    X=np.array(X_train)
    Y=np.array(Y_train)
    Z=np.array(Z_train)
    Y=(Y-Z)/2
    Y_train=[]
    for i in range(len(Y)):
        if Y[i]<-5:
            Y_train.append(np.array([0]))
        elif -5<=Y[i]<-4:
            Y_train.append(np.array([1]))
        elif -4<=Y[i]<-3:
            Y_train.append(np.array([2]))
        elif -3<=Y[i]<-2:
            Y_train.append(np.array([3]))
        elif -2<=Y[i]<-1:
            Y_train.append(np.array([4]))
        elif -1<=Y[i]<0:
            Y_train.append(np.array([5]))
        elif 0<=Y[i]<1:
            Y_train.append(np.array([6]))
        elif 1<=Y[i]<2:
            Y_train.append(np.array([7]))
        elif 2<=Y[i]<3:
            Y_train.append(np.array([8]))
        elif 3<=Y[i]<4:
            Y_train.append(np.array([9]))
        elif 4<=Y[i]<5:
            Y_train.append(np.array([10]))
        elif 5<=Y[i]:
            Y_train.append(np.array([11]))
    Y=np.array(Y_train)
    return X, Y


# In[53]:


def shuffle1(X,Y):
  np.random.seed()
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]


# In[54]:


# 將Training Data取一部份當作Validation Data
def splitData(X,Y,rate):
    X_train = X[:-int(X.shape[0]*rate)]

    Y_train = Y[:-int(Y.shape[0]*rate)]
    
    X_val = X[-int(X.shape[0]*rate):]

    Y_val = Y[-int(Y.shape[0]*rate):]
    return X_train, Y_train, X_val, Y_val


# In[79]:


def buildOneToOneModel(shape):
    model = Sequential()
    model.add(LSTM(12,input_shape=(None, 8),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(24,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(36))
    model.add(Dense(12))
    model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
    model.summary()
    return model


# In[80]:


from sklearn.utils import shuffle

mergeData()
train=readData()
train=changeYear(train)
train=augFeatures(train)
train=manage(train)
#train = shuffle(train)

temp=train
train=normalize(train)

train_x1, train_y1 = buildTrain(train, 1, 1)
train_x2, train_y2 = buildTrain(temp, 1, 1)
train_x, train_y = train_x1,train_y2
#train_x, train_y = shuffle(train_x, train_y )

train_x, train_y , val_x, val_y = splitData(train_x, train_y , 0.1)

train_y=np_utils.to_categorical(train_y)
val_y=np_utils.to_categorical(val_y)

#train_x= np.reshape(train_x, (train_x.shape[0],train_x.shape[2]))
#val_x= np.reshape(val_x, (val_x.shape[0],val_x.shape[2]))


model = buildOneToOneModel(train_x.shape)



#callback = EarlyStopping(monitor="acc", patience=10, verbose=1, mode="auto")

model.fit(train_x, train_y, epochs=500, batch_size=200,verbose=2, validation_split=0.2)


# In[ ]:


scores= model.evaluate(val_x, val_y,verbose=1,batch_size=150)
print(scores)


# In[15]:


print(train_y )

