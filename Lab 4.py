#!/usr/bin/env python
# coding: utf-8

# # AI in Enterprise Systems (AIDI 2004-02)

# ## Lab Assignment 4 - Heroku

# #### Done by:- Abraham Mathew (100829875)

# ## SVM Model to predict the fish species

# In[1]:


#Load Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


#Load Dataset
dataset = pd.read_csv('Fish.csv')
dataset.head()


# In[3]:


dataset.info()


# In[5]:


#Show Key Statistics
dataset.describe()


# In[6]:


#Class Balance
print('Class Split')
print(dataset['Species'].value_counts())
dataset['Species'].value_counts().plot.bar(figsize=(10,4),title='Species Split for fish')
plt.xlabel('Species')
plt.ylabel('Count')


# In[9]:


#Correlation Matrix
fig = plt.figure(figsize=(5,5))
plt.title('Correlation Matrix')
sns.heatmap(dataset.corr(),annot=True, cmap = 'Blues')


# In[10]:


#All the length columns are highly correlated. Keep only Length 1 for the analysis


# In[16]:


#Splitting Features and Target variables
X = dataset.drop(['Length2','Length3','Species'] , axis = 1)
y = dataset['Species']


# In[17]:


#Standardizing the variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_std = sc_X.fit_transform(X)


# In[22]:


# Train and Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_std, y, test_size=0.25,stratify=y,random_state=100)


# In[27]:


#SVM Model
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix  


# In[28]:


model = SVC()
model = model.fit(x_train,y_train)
predict = model.predict(x_test)
print('\nEstimator: SVM') 
print(confusion_matrix(y_test,predict))  
print(classification_report(y_test,predict))  


# In[29]:


#Saving the m odel
import pickle
pickle.dump(model, open('lab4_model.pkl', 'wb'))


# In[ ]:




