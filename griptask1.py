#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("http://bit.ly/w-data")
data.head(10)


# In[3]:


data.plot(x="Hours",y= "Scores", style ="o")
plt.title('Hours vs Scores')
plt.xlabel('Hours studied')
plt.ylabel('Scored')
plt.show()


# In[4]:


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.30 , random_state=0)


# In[6]:


from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()  
regressor.fit(x_train, y_train)


# In[7]:


line=regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[17]:


print(x_test)
y_predict=regressor.predict(x_test)


# In[18]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})
df


# In[19]:


hours=[[9.25]]
our_predict=regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(our_predict[0]))


# In[20]:


from sklearn import metrics
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_predict))


# In[21]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_predict))
print(rmse)


# In[ ]:




