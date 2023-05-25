#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
plt.style.use('ggplot')
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import re
from io import BytesIO
import requests
import json
from urllib.parse import urlencode
import gspread
import pingouin as pg
from pingouin import multivariate_normality
import math as math
import scipy as scipy
import scipy.stats as stats
from df2gspread import df2gspread as d2g
from oauth2client.service_account import ServiceAccountCredentials 
df=pd.read_csv('/mnt/HC_Volume_18315164/home-jupyter/jupyter-a-/Stat_less10/hw_aa.csv' , sep=';')


# In[2]:


df


# In[3]:


df[df.experimentVariant==1]['purchase']


# In[4]:


df1=df.query('experimentVariant==1')


# In[5]:


df1.purchase.unique()


# In[6]:


df1


# In[7]:


df2=df.query('experimentVariant==0')
df2


# In[8]:


n = 100000
simulations = 1000
n_s = 1000
res = []

# df = pd.DataFrame({
#     "s1": np.random.exponential(scale=1/0.001, size=n),
#     "s2": np.random.exponential(scale=1/0.001, size=n)
# })


# Запуск симуляций A/A теста
for i in tqdm(range(simulations)):
    s1 = df1['purchase'].sample(n_s, replace = False).values
    s2 = df2['purchase'].sample(n_s, replace = False).values
    res.append(stats.ttest_ind(s1, s2, equal_var = False)[1]) # сохраняем pvalue

plt.hist(res, bins = 50)
plt.style.use('ggplot')
plt.xlabel('pvalues')
plt.ylabel('frequency')
plt.title("Histogram of ttest A/A simulations ")
plt.show()

# Проверяем, что количество ложноположительных случаев не превышает альфа
sum(np.array(res) == 0.05) / simulations


# In[13]:


# FPR > альфа, значит сплит-система работает некорректно


# In[14]:


df111=df.query('purchase==1')
df111


# In[15]:


df1111=df111.groupby(['version','experimentVariant']).agg({'purchase': 'count'})
df1111


# In[16]:


df1111['conv']=(df1111.purchase/127018)*100
df1111


# In[17]:


ss=df.groupby(['version','experimentVariant']).agg({'purchase': 'count'})
ss


# In[18]:


ss['conv']=(ss.purchase/127018)*100


# In[19]:


ss


# In[20]:


f=df[df.version=='v2.8.0']['purchase']


# In[21]:


f281=df1.query('version=="v2.8.0"')
f281


# In[22]:


f280=df2.query('version=="v2.8.0"')
f280


# In[23]:


# v/"v2.8.0"
n = 100000
simulations = 1000
n_s = 1000
res = []

# df = pd.DataFrame({
#     "s1": np.random.exponential(scale=1/0.001, size=n),
#     "s2": np.random.exponential(scale=1/0.001, size=n)
# })


# Запуск симуляций A/A теста
for i in tqdm(range(simulations)):
    s1 = f281['purchase'].sample(n_s, replace = False).values
    s2 = f280['purchase'].sample(n_s, replace = False).values
    res.append(stats.ttest_ind(s1, s2, equal_var = False)[1]) # сохраняем pvalue

plt.hist(res, bins = 50)
plt.style.use('ggplot')
plt.xlabel('pvalues')
plt.ylabel('frequency')
plt.title("Histogram of ttest A/A simulations ")
plt.show()

# Проверяем, что количество ложноположительных случаев не превышает альфа
sum(np.array(res) > 0.05) / simulations


# In[24]:


f291=df1.query('version=="v2.9.0"')
f291


# In[25]:


f290=df2.query('version=="v2.9.0"')
f290


# In[26]:


# v/"v2.9.0"
n = 100000
simulations = 1000
n_s = 1000
res = []

# df = pd.DataFrame({
#     "s1": np.random.exponential(scale=1/0.001, size=n),
#     "s2": np.random.exponential(scale=1/0.001, size=n)
# })


# Запуск симуляций A/A теста
for i in tqdm(range(simulations)):
    s1 = f291['purchase'].sample(n_s, replace = False).values
    s2 = f290['purchase'].sample(n_s, replace = False).values
    res.append(stats.ttest_ind(s1, s2, equal_var = False)[1]) # сохраняем pvalue

plt.hist(res, bins = 50)
plt.style.use('ggplot')
plt.xlabel('pvalues')
plt.ylabel('frequency')
plt.title("Histogram of ttest A/A simulations ")
plt.show()

# Проверяем, что количество ложноположительных случаев не превышает альфа
sum(np.array(res) > 0.05) / simulations


# In[27]:


f2371=df1.query('version=="v3.7.4.0"')
f2371


# In[28]:


f2370=df2.query('version=="v3.7.4.0"')
f2370


# In[29]:


# v/"v3.7.4.0"
n = 100000
simulations = 1000
n_s = 1000
res = []

# df = pd.DataFrame({
#     "s1": np.random.exponential(scale=1/0.001, size=n),
#     "s2": np.random.exponential(scale=1/0.001, size=n)
# })


# Запуск симуляций A/A теста
for i in tqdm(range(simulations)):
    s1 = f2371['purchase'].sample(n_s, replace = False).values
    s2 = f2370['purchase'].sample(n_s, replace = False).values
    res.append(stats.ttest_ind(s1, s2, equal_var = False)[1]) # сохраняем pvalue

plt.hist(res, bins = 50)
plt.style.use('ggplot')
plt.xlabel('pvalues')
plt.ylabel('frequency')
plt.title("Histogram of ttest A/A simulations ")
plt.show()

# Проверяем, что количество ложноположительных случаев не превышает альфа
sum(np.array(res) > 0.05) / simulations


# In[30]:


f2381=df1.query('version=="v3.8.0.0"')
f2381


# In[31]:


f2380=df2.query('version=="v3.8.0.0"')
f2380


# In[32]:


# v/"v3.7.4.0"
n = 100000
simulations = 1000
n_s = 1000
res = []

# df = pd.DataFrame({
#     "s1": np.random.exponential(scale=1/0.001, size=n),
#     "s2": np.random.exponential(scale=1/0.001, size=n)
# })


# Запуск симуляций A/A теста
for i in tqdm(range(simulations)):
    s1 = f2381['purchase'].sample(n_s, replace = False).values
    s2 = f2380['purchase'].sample(n_s, replace = False).values
    res.append(stats.ttest_ind(s1, s2, equal_var = False)[1]) # сохраняем pvalue

plt.hist(res, bins = 50)
plt.style.use('ggplot')
plt.xlabel('pvalues')
plt.ylabel('frequency')
plt.title("Histogram of ttest A/A simulations ")
plt.show()

# Проверяем, что количество ложноположительных случаев не превышает альфа
sum(np.array(res) > 0.05) / simulations


# In[33]:


df_v3=df.query('version !="v2.8.0"') 
# Убираем версию v2.8.0 т.к. тест показал, что эта версия проблемная


# In[34]:


df_v31=df_v3.query('experimentVariant==1')
df_v31


# In[35]:


df_v30=df_v3.query('experimentVariant==0')
df_v30


# In[36]:


# v/"v0"
n = 100000
simulations = 1000
n_s = 1000
res = []

# Запуск симуляций A/A теста
for i in tqdm(range(simulations)):
    s1 = df_v31['purchase'].sample(n_s, replace = False).values
    s2 = df_v30['purchase'].sample(n_s, replace = False).values
    res.append(stats.ttest_ind(s1, s2, equal_var = False)[1]) # сохраняем pvalue

plt.hist(res, bins = 50)
plt.style.use('ggplot')
plt.xlabel('pvalues')
plt.ylabel('frequency')
plt.title("Histogram of ttest A/A simulations ")
plt.show()

# Проверяем, что количество ложноположительных случаев не превышает альфа
print('Проверяем, что количество ложноположительных случаев не превышает альфа:',sum(np.array(res) > 0.05) / simulations)


# In[37]:


from scipy import stats

stats.ttest_ind(f281['purchase'], f280['purchase'])


# Вывод: был проведен АА-тест. Тест показал, что FPR > альфа, следовательно сплит-система работает некорректно. 
# Чтобы понять где проблема, провели тесты отдельно для каждой версии приложения. Проблема обнаружилась в версии v2.8.0.
# Итоговый тест без учета этой версии показал корректность работы сплит-системы.

# In[ ]:




