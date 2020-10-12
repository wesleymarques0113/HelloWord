#!/usr/bin/env python
# coding: utf-8

# # Manipulação dos dados
# ## Retirando variáveis inúteis, alterando variáveis string para binário e repartindo os dados em X e Y

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
#Importar o DataSet

#PC do estágio
#data = pd.read_excel("C:/Users/est.wesleymg/OneDrive - Votorantim/Votorantim CA/Sistema PI/Dados_um_mes_poty_12variaveis_treino.xlsx")
#PC aqui de casa
data = pd.read_excel("C:/Users/wesle/OneDrive - Votorantim/Votorantim CA/Sistema PI/Dados_1ano_poty_12variaveis_treino.xlsx")

#Apagando as variáveis que contém 'W' (Forno desativado)

for tags in data.columns:
    if 'W' in tags:
        data = data.drop(tags, axis=1)

# Alterando algumas variáveis do tipo string para valores binários

data = data.replace('Bad', 1)
data = data.replace('CPIIZ32', 1)
data = data.replace('CPIIZ40', 2)
data = data.replace('SemProducao', 0)
data = data.replace('Ligado', 1)
data = data.replace('Desligado', 0)
data = data.replace('No Sample', 0)
data = data.replace('Normal', 1)
data = data.replace('No Data', 0)
#data = data.drop('PP-Z1HE_CPIIZ32_OBS', axis=1)
#Salvando as datas numa variável

datatempo = data['Data'].copy

#Apagando a variável Data do meu DataSet

data = data.drop('Data', axis=1)

#Separando as variáveis em X e Y

X = data.drop('PP-Z1PRODUCAO', axis=1)
y = data['PP-Z1PRODUCAO']


# # Dividindo os dados em teste e treino

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# # Aplicando os modelos (Ridge, Lasso, RandomForestRegressor, MLPRegressor)

# ### Ridge

# In[ ]:


from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train, y_train)
train_score_ridge=ridge.score(X_train, y_train)
test_score_ridge=ridge.score(X_test, y_test)

print(train_score_ridge)
print(test_score_ridge)


# In[ ]:


Variaveis_importantes_Ridge = []
for tupla in sorted(zip(ridge.coef_, X.columns), reverse=True):
    Variaveis_importantes_Ridge.append(tupla)


# ### Lasso

# In[ ]:


from sklearn.linear_model import Lasso
Lassos = Lasso(max_iter=100000)
Lassos.fit(X_train, y_train)
train_score_Lasso=Lassos.score(X_train, y_train)
test_score_Lasso=Lassos.score(X_test, y_test)

print(train_score_Lasso)
print(test_score_Lasso)


# In[ ]:


Variaveis_importantes_Lasso = []
for tupla1 in sorted(zip(Lassos.coef_, X.columns), reverse=True):
    Variaveis_importantes_Lasso.append(tupla1)


# ### RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
parametros = {'n_estimators': [ 63, 65, 66], 'random_state':[0], 'max_features':['sqrt']}
modelo_r = RandomForestRegressor()
grid = GridSearchCV(modelo_r, param_grid=parametros, cv=3)
grid.fit(X_train, y_train)
print(f'Os melhores parâmetros são: {grid.best_estimator_} \n com o score: {grid.best_score_}')


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
modelo = RandomForestRegressor(max_features='sqrt',n_estimators=63, random_state=0)
modelo.fit(X_train,y_train)
R2_treino = modelo.score(X_train,y_train)
R2_teste = modelo.score(X_test,y_test)
print(R2_treino)
print(R2_teste)


# In[ ]:


Variaveis_importantes_RandomForest = []
for tuplax in sorted(zip(modelo.feature_importances_, X.columns), reverse=True):
    Variaveis_importantes_RandomForest.append(tuplax)


# ### MLPRegressor

# In[ ]:


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
parametros_m = {'hidden_layer_sizes': [(14,14)], 'alpha':[1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1,10,20,50,100,200]}
modelo_m = MLPRegressor(random_state=0, max_iter=10000, solver='lbfgs')
grid_m = GridSearchCV(modelo_m, param_grid=parametros_m, cv=3)
grid_m.fit(X_train, y_train)
print(f'Os melhores parâmetros são: {grid_m.best_estimator_} \n com o score: {grid_m.best_score_}')


# In[ ]:


from sklearn.neural_network import MLPRegressor
modelo_RedeNeural = MLPRegressor(hidden_layer_sizes=(14,14), alpha=10,
                                 solver='lbfgs', max_iter=10000, random_state=0).fit(X_train,y_train)
R2_treino_RedeNeural = modelo_RedeNeural.score(X_train,y_train)
R2_teste_RedeNeural = modelo_RedeNeural.score(X_test,y_test)

print(R2_treino_RedeNeural)
print(R2_teste_RedeNeural)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
modelo_knn = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
R2_treino_knn = modelo_knn.score(X_train,y_train)
R2_teste_knn = modelo_knn.score(X_test,y_test)
print(R2_treino_knn)
print(R2_teste_knn)


# # Importando dados de um dia de operação para realizar o teste dos modelos!

# In[ ]:


#caminho no PC do estágio
#datat = pd.read_excel("C:/Users/est.wesleymg/OneDrive - Votorantim/Votorantim CA/Sistema PI/Dados_um_mes_poty_12variaveis_treino.xlsx")
#caminho no meu pc
datat = pd.read_excel("C:/Users/werico/OneDrive - Votorantim/Votorantim CA/Sistema PI/Dados_um_mes_poty_12variaveis_teste.xlsx")

#Limpando dados do forno (que está desativado)
for tags in datat.columns:
    if 'W' in tags:
        datat = datat.drop(tags, axis=1)
        
#Substituindo alguns valores de string por número
datat = datat.replace('Bad', 1)
datat = datat.replace('CPIIZ32', 1)
datat = datat.replace('CPIIZ40', 1)
datat = datat.replace('SemProducao', 0)
datat = datat.replace('Ligado', 1)
datat = datat.replace('Desligado', 0)
datat = datat.replace('Normal', 1)
datat = datat.replace('No Sample', 0)
datat = datat.replace('Bad', 1)
#datat = datat.drop('PP-Z1HE_CPIIZ32_OBS', axis=1)
#Limpando valores do tipo texto
#for tag in datat.columns:
#    if datat[tag].dtype == object:
#        datat = datat.drop(tag, axis=1)

DataTempoT = datat['Data'].copy()
#Apagando dados do tipo data
datat = datat.drop('Data', axis=1)

#Definindo eixo X e y
X1 = datat.drop('PP-Z1PRODUCAO', axis=1)
y1 = datat['PP-Z1PRODUCAO']


# # Comparando os resultados dos modelos

# In[ ]:


Resultados = []
for result in zip(ridge.predict(X1), Lassos.predict(X1), modelo.predict(X1), modelo_RedeNeural.predict(X1), 
                  modelo_knn.predict(X1), y1):
    Resultados.append(result)
scores = pd.DataFrame(Resultados, columns = ['Ridge', 'Lasso', 'RandomForest', 'MLP', 'KNN', 'Dados da Planta'])
scores['Data'] = DataTempoT
scores


# In[ ]:


plt.plot(scores['Data'], scores['Ridge'], color='blue' )
plt.plot(scores['Data'], scores['Lasso'], color = 'green')
plt.plot(scores['Data'], scores['RandomForest'], color = 'red')
plt.plot(scores['Data'], scores['MLP'], color = 'gray')
#plt.plot(scores['Data'], scores['KNN'], color='purple')
plt.plot(scores['Data'], scores['Dados da Planta'], color='black')
plt.xlabel("Data")
plt.ylabel("Produção de cimento (ton/h)")
plt.title('Comparação entre Modelos de Regressão')
plt.legend()
plt.show()


# ## Variáveis importantes para cada modelo

# ### Ridge

# In[ ]:


Variaveis_importantes_Ridge


# ### Lasso

# In[ ]:


Variaveis_importantes_Lasso


# ### RandomForestRegressor

# In[ ]:


Variaveis_importantes_RandomForest


# In[ ]:


for x in range(0, 12):
    print(Variaveis_importantes_RandomForest[x][1])

