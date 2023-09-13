#!/usr/bin/env python
# coding: utf-8

# # Stock Price Prediction Project

# In[1]:


from platform import python_version
print(python_version())


# In[2]:


pip install keras


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM
 
import warnings
warnings.filterwarnings('ignore')


# ## Importing Data

# In[4]:


stock_df=pd.read_csv('Stock_Price_data_set.csv')


# In[5]:


stock_df


# ## Exploring Data

# In[6]:


stock_df.shape


# In[7]:


stock_df.info()


# In[8]:


stock_df.columns


# In[9]:


stock_df.describe


# In[10]:


plt.figure(figsize=(15,8))
plt.plot(stock_df['Low'],color="red",label='Low')
plt.plot(stock_df['High'],color="green",label='High')
plt.title('Stock Price', fontsize=15)

plt.ylabel('Price')
plt.show()


# ## Missing Values

# In[11]:


stock_df.isna().any()


# ## Duplicates

# In[12]:


stock_df.duplicated().sum()


# ## Column Data Type

# In[13]:


stock_df.dtypes


# ## Outliers

# In[14]:


plt.subplot(2,3,1)
stock_df['Open'].plot(kind='box') 

plt.subplot(2,3,2)
stock_df['Close'].plot(kind='box')

plt.subplot(2,3,3)
stock_df['Adj Close'].plot(kind='box')

plt.subplot(2,3,4)
stock_df['High'].plot(kind='box')

plt.subplot(2,3,5)
stock_df['Low'].plot(kind='box')

plt.subplot(2,3,6)
stock_df['Volume'].plot(kind='box')

plt.tight_layout()


# ## ML MODELING

# In[15]:


stock_df


# In[16]:


X = stock_df.iloc[:, 1:8]
X = pd.get_dummies(X)
X


# In[17]:


features = ['Open', 'High', 'Low', 'Close', 'Volume']
 
plt.subplots(figsize=(20,10))
 
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(stock_df[col])
plt.show()


# In[18]:


splitted = stock_df['Date'].str.split('-', expand=True)
 
stock_df['month'] = splitted[1].astype('int')
stock_df['year'] = splitted[0].astype('int')
stock_df['date'] = splitted[2].astype('int')
 
stock_df.head()


# In[19]:


#columns_to_drop = [8,9,,17,18] 

stock_df.drop(stock_df.columns[0], axis=1, inplace=True)


# In[20]:


stock_df['is_quarter_end'] = np.where(stock_df['month']%3==0,1,0)
stock_df.head()


# In[21]:


df=stock_df


# In[22]:


data_grouped = df.groupby('year').nunique()
plt.subplots(figsize=(20,10))
 
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
plt.show()


# In[23]:


df.groupby('is_quarter_end').nunique()


# In[24]:


df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# In[25]:


plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()


# ### Correlation

# In[26]:


plt.figure(figsize=(15, 10))
sb.heatmap(df.corr(), annot=True)
plt.show()


# ## DATA SPLITTING AND NORMALISING

# In[27]:


data=df.filter(['Close'])
dataset=data.values
training_data_len=math.ceil(len(dataset)*0.8)
training_data_len
training_data_len

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data


# In[28]:


train_data=scaled_data[0:training_data_len,:]
x_train=[]
y_train=[]

for i in range (60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()


# In[29]:


x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[30]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[31]:


model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(x_train,y_train,batch_size=1,epochs=1)


# In[32]:


test_data=scaled_data[training_data_len-60:,:]

x_test=[]
y_test=dataset[training_data_len:, :]
for i in range (60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[33]:


x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


# In[34]:


predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
print(predictions)


# In[35]:


rmse=np.sqrt(np.mean(predictions-y_test)**2)
rmse


# In[36]:



train=data[:training_data_len]
valid=data[training_data_len:]
valid['predictions']=predictions

plt.figure(figsize=(15,5))
plt.title('Model')
plt.xlabel('date',fontsize=18)
plt.ylabel('close price',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','predictions']])
plt.legend(['train','val','prediction'],loc='lower right')
plt.show()


# Here the **Green line** depict Predicted values where as **Yellow line** depicts the actual value. 

# In[37]:


valid


# We can observe that the accuracy achieved by the ML model is no better than simply guessing with a probability of 50%. Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market Price prediction.
