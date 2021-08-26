#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# ## Importing dataset, Preprocessing and Variable seperation

# In[2]:


pop_data = pd.read_csv('/media/shreyashkharat/Storage Drive/Machine Learning/Python/Projects/Wine Quality Prediction/winequality-red.csv', header = 0)
pop_data.head()


# In[3]:


pop_data.describe()


# * From the above description, free SO2 and total SO2 may have outliers. To make sure lets draw boxplots.

# In[4]:


sns.boxplot(x = 'free sulfur dioxide', data = pop_data)


# In[5]:


sns.boxplot(x = 'total sulfur dioxide', data = pop_data)


# * The plots show that there are considerable amount of outliers.

# ### Outlier Treatment

# In[6]:


pop_data['free_so2'] = pop_data['free sulfur dioxide']
pop_data['total_so2'] = pop_data['total sulfur dioxide']
del pop_data['free sulfur dioxide']
del pop_data['total sulfur dioxide']
pop_data.head()


# In[7]:


max_free_so2 = np.percentile(pop_data.free_so2, [99])[0]
pop_data.free_so2[pop_data.free_so2 > max_free_so2] = max_free_so2
max_total_so2 = np.percentile(pop_data.total_so2, [99])[0]
pop_data.total_so2[pop_data.total_so2 > max_total_so2] = max_total_so2


# In[8]:


pop_data.describe()


# #### As per requirement, a wine with quality greater than 6.5 is considered good. Our goal is to classify whether the wine is good or not.

# In[9]:


pop_data['quality_bool'] = 0


# In[10]:


pop_data.quality_bool[pop_data.quality > 6.5] = 1


# In[11]:


del pop_data['quality']
pop_data.describe()


# ### Variable Seperation and Train Test Split

# In[12]:


x = pop_data.loc[:, pop_data.columns != 'quality_bool']
y = pop_data['quality_bool']


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# ## Logistic Model

# In[14]:


from sklearn.linear_model import LogisticRegression
model_logi = LogisticRegression()
model_logi.fit(x_train, y_train)


# In[15]:


# Predictions
logi_predict_train = model_logi.predict(x_train)
logi_predict_test = model_logi.predict(x_test)


# In[16]:


from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_train, logi_predict_train)


# In[17]:


confusion_matrix(y_test, logi_predict_test)


# ### Performance of Logistic Model

# In[18]:


accuracy_logi_test = accuracy_score(y_test, logi_predict_test)
accuracy_logi_train = accuracy_score(y_train, logi_predict_train)
accuracy_logi_train, accuracy_logi_test


# In[19]:


from sklearn.metrics import precision_score, recall_score, roc_auc_score
precision_score(y_test, logi_predict_test)
# Precision score is a scale of ability of model not to classify false obs. as true.


# In[20]:


recall_score(y_test, logi_predict_test)
# Recall score is a scale of ability of model to predict the true obs.


# In[21]:


roc_auc_score(y_test, logi_predict_test)
# Area under Reciever Operator Chaaracteristic curve.


# ## Linear Discriminant Analysis

# In[22]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
model_lda = lda()
model_lda.fit(x_train, y_train)
lda_predict_train = model_lda.predict(x_train)
lda_predict_test = model_lda.predict(x_test)


# In[23]:


confusion_matrix(y_train, lda_predict_train)


# In[24]:


confusion_matrix(y_test, lda_predict_test)


# ### Performance of Linear Discriminant Analysis

# In[25]:


accuracy_lda_train = accuracy_score(y_train, lda_predict_train)
accuracy_lda_test = accuracy_score(y_test, lda_predict_test)
accuracy_lda_train, accuracy_lda_test


# In[26]:


precision_score(y_test, lda_predict_test)
# Precision score is a scale of ability of model not to classify false obs. as true.


# In[27]:


recall_score(y_test, lda_predict_test)
# Recall score is a scale of ability of model to predict the true obs.


# In[28]:


roc_auc_score(y_test, lda_predict_test)
# Area under Reciever Operator Chaaracteristic curve.


# ## K Nearest Neighbors

# ### Scaling of data

# In[29]:


from sklearn import preprocessing
scaler_train = preprocessing.StandardScaler().fit(x_train)
x_train_scaled  = scaler_train.transform(x_train)
scaler_test = preprocessing.StandardScaler(). fit(x_test)
x_test_scaled = scaler_test.transform(x_test)


# In[30]:


from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import GridSearchCV
n_neighbors_grid = {'n_neighbors' : range(1,16,1)}
grid_knn_search = GridSearchCV(knn(), n_neighbors_grid)


# In[31]:


grid_knn_search.fit(x_train, y_train)


# In[32]:


grid_knn_search.best_params_


# In[33]:


optimised_model_knn = grid_knn_search.best_estimator_
knn_predict_train = optimised_model_knn.predict(x_train)
knn_predict_test = optimised_model_knn.predict(x_test)


# ### Performance of Optimised KNN Model

# In[34]:


accuracy_knn_train = accuracy_score(y_train, knn_predict_train)
accuracy_knn_test = accuracy_score(y_test, knn_predict_test)
accuracy_knn_train, accuracy_knn_test


# In[35]:


precision_score(y_test, knn_predict_test)
# Precision score is a scale of ability of model not to classify false obs. as true.


# In[36]:


recall_score(y_test, knn_predict_test)
# Recall score is a scale of ability of model to predict the true obs.


# In[37]:


roc_auc_score(y_test, knn_predict_test)
# Area under Reciever Operator Chaaracteristic curve.


# Each of the three models have their own merits, the required model can be selected as per application.

# In[ ]:




