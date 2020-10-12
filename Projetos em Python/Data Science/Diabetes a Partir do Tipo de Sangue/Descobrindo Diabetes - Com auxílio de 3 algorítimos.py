#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('C:/Users/wesle/Desktop/Graduação Eng Química/DataSciencePython/DataScienceDSA/datasets_4511_6897_diabetes.csv')
df.head()


# In[6]:


df.shape
df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.tail()


# In[34]:


df.head(10)


# In[33]:


colunas_x = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
nozero = df.replace(0, df[colunas_x].mean())
nozero
#Trocando os zeros pela média dos valores


# In[35]:


num_True = df.loc[df['Outcome'] == 1]
num_False = df.loc[df['Outcome'] == 0]
#print(f'O número de diabéticos são: {len(num_True)})
#print(f'O número de não diabíticos são: {len(num_False)})
print(len(num_True))
print(len(num_False))
y = df['Outcome']
x = nozero.drop('Outcome', axis=1)


# In[36]:


y.head(10)


# In[37]:


x.head(10)


# In[39]:


#Dividindo os dados de forma randômica
from sklearn.model_selection import train_test_split
x_teste,x_treino,y_teste,y_treino = train_test_split(x,y,test_size=0.3, random_state=42)


# In[40]:


len(x_teste)


# In[41]:


#Importando o modelo
from sklearn.naive_bayes import GaussianNB


# In[43]:


modelo = GaussianNB()
modelo


# In[44]:


modelo.fit(x_treino,y_treino)


# In[45]:


# Testar a acurácia do método com os dados de treino, mas é melhor usar os dados de treino
from sklearn import metrics


# In[47]:


previsoes_treino = modelo.predict(x_treino)
previsoes_treino


# In[54]:


acuracia = metrics.accuracy_score(y_treino,previsoes_treino)
acuracia*100


# In[55]:


previsoes_teste = modelo.predict(x_teste)
acuracia_treino = metrics.accuracy_score(y_teste,previsoes_teste)
acuracia_treino*100


# ## Alterando o algorítimo de machoneLearn para otimizar o método

# In[59]:


from sklearn.ensemble import RandomForestClassifier


# In[74]:


modelo_forest = RandomForestClassifier()
modelo_forest.fit(x_treino,y_treino)


# In[75]:


prev = modelo_forest.predict(x_teste)
acu = metrics.accuracy_score(y_teste,prev)
acu*100


# ## Usando regressão logistíca

# In[78]:


from sklearn.linear_model import LogisticRegression


# In[82]:


modelo_regr = LogisticRegression()
modelo_regr.fit(x_treino,y_treino)


# In[84]:


acuracia_regr = metrics.accuracy_score(y_teste,modelo_regr.predict(x_teste))
acuracia_regr*100


# In[ ]:





# In[ ]:




