#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
import pandas as pd


# In[5]:


df = pd.read_csv(r"C:\Users\MANISH SHARMA\Desktop\datasets\kaggledatasetstitanic\train.csv")


# In[6]:


df


# In[7]:


df.isna().sum()


# In[8]:


df = df.fillna(df["Age"].mean())


# In[9]:


df


# In[10]:


df.isna().sum()


# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
sex = le.fit_transform(df["Sex"])


# In[12]:


df["sex"] = sex


# In[13]:


df


# In[14]:


x = df[["Age","sex","Pclass","Parch"]]


# In[15]:


y = df["Survived"]


# In[16]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x , y )


# In[17]:


gnb.score(x,y)


# In[18]:


df1 = pd.read_csv(r"C:\Users\MANISH SHARMA\Desktop\datasets\kaggledatasetstitanic\test.csv")
df1


# In[19]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
sex = le.fit_transform(df1["Sex"])


# In[20]:


df1["sex"] = sex


# In[21]:


df1


# In[22]:


df1 = df1.fillna(df1["Age"].mean())


# In[23]:


df1


# In[24]:


x_test = df1[["Age","sex","Pclass","Parch"]]


# In[25]:


x_test


# In[26]:


y_pred = gnb.predict(x_test)


# In[27]:


y_pred


# In[28]:


df1["Survived"] = y_pred


# In[29]:


df1


# In[32]:


kaggleans = df1[["PassengerId" , "Survived"]]


# In[33]:


kaggleans


# In[34]:


kaggleans.to_csv(r"C:\Users\MANISH SHARMA\Desktop\datasets\kaggledatasetstitanic\kaggle_titanic_ans.csv" , index = False)


# In[ ]:





# In[ ]:




