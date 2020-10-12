#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
arquivo = pd.read_csv('C:/Users/wesle/Desktop/DataSciencePython/DataScienceYoutube/datasets_16721_22034_wine_dataset.csv')
arquivo.head()


# In[5]:


arquivo.shape


# In[6]:


arquivo.describe()


# In[9]:


arquivo['style']


# In[12]:


#Troca onde tiver, red é 0 e onde tiver white é 1
arquivo['style'].replace('red', 0, inplace=True)
arquivo['style'].replace('white', 1, inplace=True)


# In[18]:


arquivo['style'].head()


# In[20]:


#Vamos separar o nosso dataSet em variáveis preditoras e variáveis alvo
y = arquivo['style']
#axis = 1 indica para apagar as colunas
x = arquivo.drop('style', axis=1)


# In[24]:


y.head()


# In[25]:


x.head()


# In[32]:


import matplotlib.pyplot as plt
plt.plot(y)


# In[33]:


from sklearn.model_selection import train_test_split
# biblioteca para dividir os dados entre treino e teste


# In[34]:


#Esse comando divide os dados em treino e teste com 70% dos dados para treino e 30% dos dados para teste de forma aleatória
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)


# In[36]:


arquivo.shape


# In[38]:


x_treino.shape


# In[40]:


(x_treino.shape[0]/arquivo.shape[0])*100


# In[41]:


# Agora vai rodar o algorítimo de machinelearn 
# Esse algorítmo é do tipo de árvore de decisão
from sklearn.ensemble import ExtraTreesClassifier


# In[49]:


# Criaçã do modelo 
modelo = ExtraTreesClassifier(n_estimators=100)
# A função fit aplica os dados ao modelo .fit(x,y)
modelo.fit(x_treino,y_treino)
# resultado do modelo
resultado = modelo.score(x_teste, y_teste)
print('Acurácia:', resultado)


# In[54]:


# O programa acertou 99,3% dos dados
y_teste[400:408]


# In[56]:


x_teste[400:408]


# In[57]:


previsoes = modelo.predict(x_teste[400:408])


# In[58]:


previsoes


# In[60]:


print(y_teste[400:408])


# In[ ]:





# In[ ]:




