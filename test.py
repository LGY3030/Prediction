
# coding: utf-8

# In[14]:


from keras.datasets import mnist
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print(X_train.shape)
print(y_train.shape)


# In[15]:


model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.summary()


# In[4]:


rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)


# In[5]:


model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[8]:


model.fit(X_train, y_train, epochs=20, batch_size=32)


# In[9]:


loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)

