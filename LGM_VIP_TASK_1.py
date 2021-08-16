#!/usr/bin/env python
# coding: utf-8

# # Lets Grow More - Data Science Internship
# 
# Author: Swapnil S Yadav
# 
# TASK-1 Iris - Flower Classification ML Project
# 
# Dataset link: https://lnkd.in/eQUVG9H5
# 
# 

# Importing Required Packages¶

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix


# Importing Iris Dataset

# In[2]:


from sklearn.datasets import load_iris
iris=load_iris()


# Let's see first 5 rows of dataset

# In[27]:


data=pd.DataFrame(iris.data)
data = data.rename(columns={0:'sepal length (cm)',1:'sepal width (cm)',2:'petal length (cm)',3:'petal width (cm)'})
data.head(5)


# Let's see basic Information about the Dataset

# In[8]:


print(iris.target_names)


# In[9]:


data.info()


# In[10]:


data.describe()


# Plotting the Dataset to Visualization

# In[19]:


data1= pd.DataFrame(iris.target)


# In[20]:


data1


# In[29]:


data_t=data.copy()
data_t['Target']=data1


# In[32]:


sns.pairplot(data_t, hue= 'Target')
plt.show


# In[43]:


sns.scatterplot(data=data_t,x=data_t['sepal length (cm)'] ,y=data_t['petal length (cm)'] , hue= 'Target')
plt.show()


# Defineing X and y independent and dependent

# In[47]:


X=iris.data 
y=iris.target

X[:5]


# In[48]:


y[:5]


# Preprocessing Data using Standard Scaler

# In[49]:


SC = StandardScaler()

X = SC.fit_transform(X)


# 
# Splitting Dataset for Training and Testing

# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Machine learning algorithm
# 
# KNN Classifier

# In[53]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train,y_train)


# Now we predict the output on testing data

# In[56]:


y_pred=knn.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
print("Classification Report is:\n",classification_report(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Accuracy of Model:\n",knn.score(X_train,y_train)*100)


# By using KNN Classifier we have got  95.83 accuracy.

# In[ ]:




