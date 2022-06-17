#!/usr/bin/env python
# coding: utf-8

# # Iris class prediction

# In[2]:


import pandas
print('pandas version is: {}'.format(pandas.__version__))
import numpy
print('numpy version is:{}'.format(numpy.__version__))
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing data set

# In[3]:


import pandas as pd
iris = pd.read_csv('iris.csv')


# In[4]:


iris.head(15)


# # Analyse and Visualize data set

# In[5]:


print(len(iris['class']))


# In[6]:


for col in iris.columns:
    print(col)


# In[7]:


print(iris.groupby('class').size())


# In[9]:


plt.figure(figsize = (15,10))
plt.subplot(2,2,1)
sns.boxplot(x= 'class', y='sepallength', data=iris)
plt.subplot(2,2,2)
sns.boxplot(x= 'class', y='sepalwidth', data=iris)
plt.subplot(2,2,3)
sns.boxplot(x='class', y='petallength', data=iris)
plt.subplot(2,2,4)
sns.boxplot(x= 'class', y= 'petalwidth', data=iris)


# In[10]:


#data cleaning


# In[11]:


iris.isnull().values.any()


# In[12]:


iris.info()


# # Splitting up of Data

# In[13]:


from sklearn.model_selection import train_test_split
array = iris.values
X = array[:,0:4]
Y = array[:,4]
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0)


# # Apply Algorithms and Evaluate

# # SUPPORT VECTOR CLASSIFIER

# In[15]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc = SVC(max_iter = 1000, gamma = 'auto')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(accuracy_score(y_pred, y_test), 2) * 100
print("Accuracy : ", acc_svc)


# # DECISION TREE CLASSIFIER

# In[16]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier(random_state = 0)
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_test)
acc_decisiontree = round(accuracy_score(y_pred, y_test), 2 ) *100
print("Accuracy : ", acc_decisiontree)


# # LOGISTIC REGRESSION

# In[17]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter = 1000)
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred, y_test), 2) * 100
print("Accuracy : ", acc_logreg)

