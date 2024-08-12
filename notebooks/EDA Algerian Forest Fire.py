#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#os.chdir('C:\\Users\\Manab Das\\Downloads\\PWSkills\\EDA')


# In[3]:


df=pd.read_excel('Algerian_forest_fires_dataset_UPDATE.xlsx',header=1)
df.head()


# In[4]:


df.info()


# # Data Cleaning:

# In[5]:


df[df.isnull().any(axis=1)]


# The dataset is converted into two sets based on Region from 122th index. We can make a new column based on the region.

# In[6]:


df.loc[:122,'Region']=0
df.loc[122:,'Region']=1


# In[7]:


df.info()


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df['Region']=df['Region'].astype(int)


# In[11]:


df.info()


# In[12]:


df.head()


# In[13]:


df.isna().sum()


# In[14]:


df=df.dropna().reset_index(drop=True)


# In[15]:


df.isna().sum()


# In[16]:


df


# In[17]:


df=df.drop(index=122).reset_index(drop=True)


# In[18]:


df.iloc[[122]]


# In[19]:


#Fix spaces in column names
df.columns=df.columns.str.strip()
df.columns


# In[20]:


df.shape


# In[21]:


df.info()


# # Changes of columns type:

# In[22]:


df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']]=df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)


# In[23]:


df.info()


# # Changing the other column into float datatype:
# 

# In[24]:


objects=[feature for feature in df.columns if df[feature].dtypes=='O']


# In[25]:


objects


# In[26]:


for i in objects:
    if i!='Classes':
        df[i]=df[i].astype(float)


# In[27]:


df.info()


# In[28]:


df.describe()


# In[29]:


df.to_csv('Algerian_forest_fires_dataset_CLEANED.csv',index=False)


# # EDA

# In[30]:


df_copy=df.drop(['day','month','year'],axis=1)


# In[31]:


df_copy


# In[32]:


df['Classes'].value_counts()


# In[33]:


df_copy['Classes']=np.where(df_copy['Classes'].str.contains('not fire'),0,1)


# In[34]:


df_copy.head()


# In[35]:


df_copy['Classes'].value_counts()


# In[36]:


plt.style.use('seaborn')
df_copy.hist(bins=50,figsize=(20,25))
plt.show()


# In[37]:


percentage=df_copy['Classes'].value_counts(normalize=True)*100


# In[38]:


classlabel=['Fire','Not Fire']
plt.figure(figsize=(12,7))
plt.pie(percentage,labels=classlabel,autopct='%1.1f%%')
plt.title('Pie Chart of Class')
plt.show()


# In[39]:


df_copy.corr()


# In[40]:


plt.figure(figsize=(12,10))
sns.heatmap(df_copy.corr(),annot=True)


# In[41]:


#Box Plot
sns.boxplot(df_copy['FWI'],color='green')


# In[44]:


## Independent And dependent features
x=df_copy.drop('FWI',axis=1)
y=df_copy['FWI']


# In[43]:


dfssssssssssssdf


# In[46]:


def corelations(dataset,thereshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>thereshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[47]:


corr_features=corelations(x,0.85)


# In[48]:


corr_features


# In[51]:


x.drop(corr_features,axis=1,inplace=True)


# In[52]:


x.head()


# In[53]:


x.shape


# In[ ]:





# # Train Test Split

# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


x_train,x_test,y_train,y_tset=train_test_split(x,y,test_size=0.25,random_state=42)


# # Feature Scaling or Standardization

# In[56]:


from sklearn.preprocessing import StandardScaler


# In[57]:


scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


# # Box Plot to Understand the Effect of Feature Scaling

# In[58]:


plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=x_train)
plt.title('X_train before scaling')
plt.subplot(1,2,2)
sns.boxplot(data=x_train_scaled)
plt.title('X-train after scaling')


# # Linear Regression Model

# In[59]:


from sklearn.linear_model import LinearRegression


# In[60]:


lm=LinearRegression()


# In[61]:


lm.fit(x_train_scaled,y_train)


# In[62]:


lm.intercept_


# In[63]:


lm.coef_


# In[64]:


y_predict=lm.predict(x_test_scaled)


# In[65]:


y_predict


# In[66]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# # There are lots of outliers present, that's why we used mean absolute error

# In[67]:


mae=mean_absolute_error(y_tset,y_predict)


# In[68]:


score=r2_score(y_tset,y_predict)


# In[69]:


print('Mean Absolute Error:',mae)
print('R Squre value:',score)


# # Lasso Regression

# In[70]:


from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
las=Lasso()
las.fit(x_train_scaled,y_train)
y_pred=las.predict(x_test_scaled)
mae=mean_absolute_error(y_tset,y_pred)
score=r2_score(y_tset,y_predict)
print('Mean Absolute Error:',mae)
print('R Squre value:',score)


# # Ridge Regression

# In[71]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
rg=Ridge()
rg.fit(x_train_scaled,y_train)
y_pred=rg.predict(x_test_scaled)
mae=mean_absolute_error(y_tset,y_pred)
score=r2_score(y_tset,y_predict)
print('Mean Absolute Error:',mae)
print('R Squre value:',score)


# # Elestic_Net

# In[72]:


from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
en=ElasticNet()
en.fit(x_train_scaled,y_train)
y_pred=en.predict(x_test_scaled)
mae=mean_absolute_error(y_tset,y_pred)
score=r2_score(y_tset,y_predict)
print('Mean Absolute Error:',mae)
print('R Squre value:',score)


# In[73]:


import pickle
pickle.dump(scaler,open('scaler.pkl','wb'))
pickle.dump(rg,open('ridge.pkl','wb'))


# In[ ]:




