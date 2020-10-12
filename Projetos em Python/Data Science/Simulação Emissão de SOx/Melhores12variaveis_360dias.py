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
#data = pd.read_excel("C:/Users/est.wesleymg/OneDrive - Votorantim/Votorantim Laranjeiras/Estudo_SOx/BaseDeDados/Dados_360dias_laranjeiras_12tags.xlsx"
#                    , sheet_name=0)
#PC aqui de casa
data = pd.read_excel("C:/Users/wesle/OneDrive - Votorantim/Votorantim Laranjeiras/Estudo_SOx/BaseDeDados/Dados_360dias_laranjeiras_12tags.xlsx"
                  , sheet_name=0)

#eliminando dados do forno 1 (desativado)
for tags in data.columns:
    if 'W1'in tags:
        data = data.drop(tags, axis=1)

# Alterando algumas variáveis do tipo string para valores binários

data = data.replace('Bad', 1)
data = data.replace('SemProducao', 0)
data = data.replace('Ligado', 1)
data = data.replace('Desligado', 0)
data = data.replace('No Sample', 0)
data = data.replace('Normal', 1)
data = data.replace('ON', 1)
data = data.replace('Off', 0)
data = data.replace('No Data', 0)
data = data.replace('Comm Fail', 0)
data = data.replace('Pt Created', 0)
data = data.replace('Calc Failed', 0)
data = data.replace('Invalid Data', 0)

#pegando somente os últimos valores (2 últimos meses)
data = data.tail(1440)
#Salvando as datas numa variável

datatempo = data['Data'].copy()

#Apagando a variável Data do meu DataSet

data = data.drop('Data', axis=1)
#data = data.drop('CI-MSE_U3U14M1', axis=1)

#Separando as variáveis em X e Y

X = data.drop('CI-W2X22A4_COR', axis=1)
y = data['CI-W2X22A4_COR']


# In[ ]:


data


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression


# In[ ]:


f_regressor = SelectKBest(f_regression, k=5)
selecao = f_regressor.fit(X,y)
melhores_features = selecao.transform(X)
print(melhores_features)


# In[ ]:


cols = selecao.get_support(indices=True)
data.iloc[:,cols]


# In[ ]:


cols = selecao.get_support(indices=True)
data.iloc[:,cols].columns


# In[ ]:


plt.plot(datatempo, data['CI-H2_SO3'], color='red', label='CI-J2T01X9')
plt.plot(datatempo, data['CI-J2T01X8'], color='gray', label ='CI-MSE_J2T01M1_I1_PLC' )
plt.plot(datatempo, data['CI-W2V21F1'], color='green', label= 'CI-MSE_J2T01M1_I2_PLC')
#plt.plot(datatempo, data['CI-W2_OPER_ON'], color='blue', label='CI-W2PRODUZ_ON_2')
plt.plot(datatempo, data['CI-W2X22A4_COR'], color='black', label ='CI-W2X22A4_COR' )
plt.legend()
plt.show()


# In[ ]:


data[['CI-J2T01X8', 'CI-H2_SO3','CI-W2X22A4_COR' ]].corr('spearman')


# In[ ]:


data.shape


# In[ ]:


y[y>0].describe()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
y[y>0].hist(bins=2000)


# In[ ]:


X['CI-H2_SO3'].describe()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
X['CI-H2_SO3'].hist(bins=100)


# # Dividindo os dados em teste e treino

# In[ ]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# # Aplicando os modelos (Ridge, Lasso, RandomForestRegressor, MLPRegressor)

# ### Ridge

# In[ ]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)
train_score_ridge=ridge.score(X_train, y_train)
test_score_ridge=ridge.score(X_test, y_test)

print(train_score_ridge)
print(test_score_ridge)


# In[ ]:


Variaveis_importantes_Ridge = []
for tupla in sorted(zip(ridge.coef_, X.columns), reverse=True):
    Variaveis_importantes_Ridge.append(tupla)


# ### Ridge Regularizado

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled_Ridge = scaler.fit_transform(X_train)
X_test_scaled_Ridge = scaler.fit_transform(X_test)

linRidge = Ridge(alpha=100)
linRidge.fit(X_train_scaled_Ridge, y_train)
train_scaled_score_ridge=ridge.score(X_train_scaled_Ridge, y_train)
test_scaled_score_ridge=ridge.score(X_test_scaled_Ridge, y_test)
print(train_scaled_score_ridge)
print(test_scaled_score_ridge)


# ### Lasso

# In[ ]:


from sklearn.linear_model import Lasso
Lassos = Lasso(max_iter=100000, alpha=10)
Lassos.fit(X_train, y_train)
train_score_Lasso=Lassos.score(X_train, y_train)
test_score_Lasso=Lassos.score(X_test, y_test)

print(train_score_Lasso)
print(test_score_Lasso)


# In[ ]:


Variaveis_importantes_Lasso = []
for tupla1 in sorted(zip(Lassos.coef_, X.columns), reverse=True):
    Variaveis_importantes_Lasso.append(tupla1)


# ### Regressão Polinomial Grau 2

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
features_2 = PolynomialFeatures(degree=4)
#X_train_2 = X_train.reshape(-1,1)
X_train_poly2 = features_2.fit_transform(X_train)
X_test_poly2 = features_2.fit_transform(X_test)
X_train_poly2.shape


# #### Modelo Linear

# In[ ]:


from sklearn.linear_model import LinearRegression
modelo_linear_2 = LinearRegression()
modelo_linear_2.fit(X_train_poly2, y_train)
train_score_linear_2=modelo_linear_2.score(X_train_poly2, y_train)
test_score_linear_2=modelo_linear_2.score(X_test_poly2, y_test)
print(train_score_linear_2)
print(test_score_linear_2)


# #### Ridge

# In[ ]:


modelo_Ridge_2 = Ridge(alpha=10)
modelo_Ridge_2.fit(X_train_poly2, y_train)
train_score_linear_2=modelo_Ridge_2.score(X_train_poly2, y_train)
test_score_linear_2=modelo_Ridge_2.score(X_test_poly2, y_test)
print(train_score_linear_2)
print(test_score_linear_2)


# #### Lasso

# In[ ]:


modelo_Lasso_2 = Lasso(alpha=10, max_iter=1000000)
modelo_Lasso_2.fit(X_train_poly2, y_train)
train_score_Lasso_2=modelo_Lasso_2.score(X_train_poly2, y_train)
test_score_Lasso_2=modelo_Lasso_2.score(X_test_poly2, y_test)
print(train_score_Lasso_2)
print(test_score_Lasso_2)


# ### RandomForestRegressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
parametros = {'max_features':np.arange(6,12),'n_estimators':[500],'min_samples_leaf': [10,50,100,200,500]}
modelo_r = RandomForestRegressor()
grid = GridSearchCV(modelo_r, param_grid=parametros, cv=3)
grid.fit(X_train, y_train)
print(f'Os melhores parâmetros são: {grid.best_estimator_} \n com o score: {grid.best_score_}')


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
#modelo = RandomForestRegressor(max_features=9, min_samples_leaf=50, n_estimators=500, random_state=0)
modelo = RandomForestRegressor(max_features=10, min_samples_leaf=10,
                      n_estimators=500, random_state=0)
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
parametros_m = {'hidden_layer_sizes': [(30,30), (35,35), (40,40)], 'alpha':[10**-3, 1e-01,1,10,100,200]}
modelo_m = MLPRegressor(random_state=0, max_iter=10000, solver='lbfgs')
grid_m = GridSearchCV(modelo_m, param_grid=parametros_m, cv=3)
grid_m.fit(X_train, y_train)
print(f'Os melhores parâmetros são: {grid_m.best_estimator_} \n com o score: {grid_m.best_score_}')


# In[ ]:


from sklearn.neural_network import MLPRegressor
#modelo_RedeNeural = MLPRegressor(hidden_layer_sizes=(10,10),alpha=1,solver='lbfgs',
#                                 max_iter=2000, random_state=0).fit(X_train,y_train)
modelo_RedeNeural = MLPRegressor(alpha=0.1, hidden_layer_sizes=(30, 30), max_iter=10000,
             random_state=0, solver='lbfgs').fit(X_train,y_train)
R2_treino_RedeNeural = modelo_RedeNeural.score(X_train,y_train)
R2_teste_RedeNeural = modelo_RedeNeural.score(X_test,y_test)

print(R2_treino_RedeNeural)
print(R2_teste_RedeNeural)


# ### SVR 

# In[ ]:


from sklearn.svm import SVR
modelo_SVR = SVR(kernel='poly', gamma=10**-5).fit(X_train, y_train)
R2_treino_svr = modelo_SVR.score(X_train, y_train)
R2_teste_svr = modelo_SVR.score(X_test, y_test)

print(R2_treino_svr)
print(R2_teste_svr)


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
modelo_knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
R2_treino_knn = modelo_knn.score(X_train,y_train)
R2_teste_knn = modelo_knn.score(X_test,y_test)
print(R2_treino_knn)
print(R2_teste_knn)


# # Importando dados de um dia de operação para realizar o teste dos modelos!

# In[ ]:


#PC do estágio
datat = pd.read_excel("C:/Users/est.wesleymg/OneDrive - Votorantim/Votorantim Laranjeiras/Estudo_SOx/BaseDeDados/Dados_360dias_laranjeiras_12tags.xlsx"
                    , sheet_name=1)
#PC aqui de casa
#datat = pd.read_excel("C:/Users/werico/OneDrive - Votorantim/Votorantim Laranjeiras/Estudo_SOx/BaseDeDados/Dados_360dias_laranjeiras_64tags.xlsx"
#                  , sheet_name=1)


# Alterando algumas variáveis do tipo string para valores binários

datat = datat.replace('Bad', 1)
datat = datat.replace('SemProducao', 0)
datat = datat.replace('Ligado', 1)
datat = datat.replace('Desligado', 0)
datat = datat.replace('No Sample', 0)
datat = datat.replace('Normal', 1)
datat = datat.replace('ON', 1)
datat = datat.replace('Off', 0)
datat = datat.replace('No Data', 0)
datat = datat.replace('Comm Fail', 0)
datat = datat.replace('Pt Created', 0)
datat = datat.replace('Calc Failed', 0)
datat = datat.replace('Invalid Data', 0)
#datat = datat.drop('CI-W2X22A4', axis=1)

#Limpando dados do tipo string

"""lista_str_t = []

for label_t in datat.columns:
    for value_t in datat[label_t]:
        if type(value_t) is str:
            if label_t not in lista_str_t:
                lista_str_t.append(label_t)

datat = datat.drop(lista_str_t, axis=1)"""
#Salvando as datas numa variável

DataTempoT = datat['Data'].copy()

#Apagando a variável Data do meu DataSet

datat = datat.drop('Data', axis=1)
#datat = datat.drop(lista, axis=1)

#Separando as variáveis em X e Y

X1 = datat.drop('CI-W2X22A4_COR', axis=1)
y1 = datat['CI-W2X22A4_COR']


# In[ ]:


X1.shape


# In[ ]:


X.shape


# # Comparando os resultados dos modelos

# In[ ]:


Resultados = []
for result in zip(ridge.predict(X1), Lassos.predict(X1), modelo.predict(X1), modelo_RedeNeural.predict(X1),
                  modelo_SVR.predict(X1), modelo_knn.predict(X1), y1):
    Resultados.append(result)
scores = pd.DataFrame(Resultados, columns = ['Ridge', 'Lasso', 'RandomForest', 'MLP', 'SVR', 'KNN', 'Dados da Planta'])
scores['Data'] = DataTempoT
scores.head(50)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.plot(scores['Data'], scores['Ridge'], color='blue', label='Ridge' )
plt.plot(scores['Data'], scores['Lasso'], color = 'green', label='Lasso')
plt.plot(scores['Data'], scores['RandomForest'], color = 'red', label='RandomForest')
plt.plot(scores['Data'], scores['MLP'], color = 'gray', label='MLP')
plt.plot(scores['Data'], scores['KNN'], color = 'purple', label='KNN')
plt.plot(scores['Data'], scores['Dados da Planta'], color='yellow', label='Dados da Planta')
plt.xlabel("Data")
plt.ylabel("Emissão SOX")
plt.title('Comparação entre Modelos de Regressão')
plt.legend()
plt.show()


# ## Variáveis importantes para cada modelo

# ### Ridge

# In[ ]:


Variaveis_importantes_Ridge


# In[ ]:


for c in Variaveis_importantes_Ridge:
    if c[0]>0:
        print(c[1])


# ### Lasso

# In[ ]:


Variaveis_importantes_Lasso


# In[ ]:


for c in Variaveis_importantes_Lasso:
    if c[0]>0:
        print(c[1])


# ### RandomForestRegressor

# In[ ]:


Variaveis_importantes_RandomForest


# In[ ]:


datat[['CI-H3_SO3', 'CI-W2X22A4_COR']].tail(50)


# In[ ]:


c = 0
for x in Variaveis_importantes_RandomForest:
    c = c+ 1
    if 'CI-H3_SO3' in x[1]:
        print(x[])


# In[ ]:


c = 0
for x in Variaveis_importantes_Ridge:
    c = c+ 1
    if 'SO3' in x[1]:
        print(x[0])


# In[ ]:


c = 0
for x in Variaveis_importantes_Lasso:
    c = c+ 1
    if 'SO3' in x[1]:
        print(x[0])


# In[ ]:


data[['CI-H3_SO3', 'CI-W2X22A4_COR']].corr()


# In[ ]:


data[['CI-W2V21F1_T', 'CI-W2X22A4_COR']].corr()


# In[ ]:




