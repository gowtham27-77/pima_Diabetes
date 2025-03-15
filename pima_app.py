#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install joblib


# In[3]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib


# In[5]:


df = pd.read_csv("diabetes.csv")
df


# In[7]:


# preprocess
X = df.drop('class', axis=1)
y = df['class']


# In[9]:


# standaridize X data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[11]:


# K-Fold Cross-Validation
model = DecisionTreeClassifier(random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_scaled, y, cv=kf)
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())


# In[15]:


# Train  final model 
model.fit(X_scaled, y)

# save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# In[ ]:




