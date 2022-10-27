#!/usr/bin/env python
# coding: utf-8

# <h1 style='color: green; font-size: 36px; font-weight: bold;'>Data Science - Regressão Linear</h1>

# # <font color='red' style='font-size: 30px;'>Conhecendo o Dataset</font>
# <hr style='border: 2px solid red;'>

# ## Importando bibliotecas

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle


# ## O Dataset e o Projeto
# <hr>
# 
# ### Fonte: https://www.kaggle.com/greenwing1985/housepricing
# 
# ### Descrição:
# <p style='font-size: 18px; line-height: 2; margin: 10px 50px; text-align: justify;'>Nosso objetivo neste exercício é criar um modelo de machine learning, utilizando a técnica de Regressão Linear, que faça previsões sobre os preços de imóveis a partir de um conjunto de características conhecidas dos imóveis.</p>
# 
# <p style='font-size: 18px; line-height: 2; margin: 10px 50px; text-align: justify;'>Vamos utilizar um dataset disponível no Kaggle que foi gerado por computador para treinamento de machine learning para iniciantes. Este dataset foi modificado para facilitar o nosso objetivo, que é fixar o conhecimento adquirido no treinamento de Regressão Linear.</p>
# 
# <p style='font-size: 18px; line-height: 2; margin: 10px 50px; text-align: justify;'>Siga os passos propostos nos comentários acima de cada célular e bons estudos.</p>
# 
# ### Dados:
# <ul style='font-size: 18px; line-height: 2; text-align: justify;'>
#     <li><b>precos</b> - Preços do imóveis</li>
#     <li><b>area</b> - Área do imóvel</li>
#     <li><b>garagem</b> - Número de vagas de garagem</li>
#     <li><b>banheiros</b> - Número de banheiros</li>
#     <li><b>lareira</b> - Número de lareiras</li>
#     <li><b>marmore</b> - Se o imóvel possui acabamento em mármore branco (1) ou não (0)</li>
#     <li><b>andares</b> - Se o imóvel possui mais de um andar (1) ou não (0)</li>
# </ul>

# ## Leitura dos dados
# 
# Dataset está na pasta "Dados" com o nome "HousePrices_HalfMil.csv" em usa como separador ";".

# In[2]:


dados = pd.read_csv('dados/HousePrices.csv', sep=';')


# ## Visualizar os dados

# In[3]:


dados.head(10)


# ## Verificando o tamanho do dataset

# In[4]:


dados.shape


# # <font color='red' style='font-size: 30px;'>Análises Preliminares</font>
# <hr style='border: 2px solid red;'>

# ## Estatísticas descritivas

# In[5]:


dados.describe().round(2)


# ## Matriz de correlação
# 
# <p style='font-size: 18px; line-height: 2; margin: 10px 50px; text-align: justify;'>O <b>coeficiente de correlação</b> é uma medida de associação linear entre duas variáveis e situa-se entre <b>-1</b> e <b>+1</b> sendo que <b>-1</b> indica associação negativa perfeita e <b>+1</b> indica associação positiva perfeita.</p>
# 
# ### Observe as correlações entre as variáveis:
# <ul style='font-size: 16px; line-height: 2; text-align: justify;'>
#     <li>Quais são mais correlacionadas com a variável dependete (Preço)?</li>
#     <li>Qual o relacionamento entre elas (positivo ou negativo)?</li>
#     <li>Existe correlação forte entre as variáveis explicativas?</li>
# </ul>

# In[6]:


dados.corr().round(4)


# # <font color='red' style='font-size: 30px;'>Comportamento da Variável Dependente (Y)</font>
# <hr style='border: 2px solid red;'>

# # Análises gráficas

# <img width='700px' src='dados/img/Box-Plot.png'>

# ## Importando biblioteca seaborn

# In[7]:


import seaborn as sns


# ## Configure o estilo e cor dos gráficos (opcional)

# In[ ]:





# ## Box plot da variável *dependente* (y)
# 
# 
# ### Avalie o comportamento da distribuição da variável dependente:
# <ul style='font-size: 16px; line-height: 2; text-align: justify;'>
#     <li>Parecem existir valores discrepantes (outliers)?</li>
#     <li>O box plot apresenta alguma tendência?</li>
# </ul>

# https://seaborn.pydata.org/generated/seaborn.boxplot.html?highlight=boxplot#seaborn.boxplot

# In[8]:


ax = sns.boxplot(data = dados['precos'], orient = 'h', width=0.5)
ax.figure.set_size_inches(12,8)
ax.set_title('Preços dos Imóveis', fontsize=20)
ax.set_xlabel('Valores', fontsize = 15)
ax


# ## Investigando a variável *dependente* (y) juntamente com outras característica
# 
# Faça um box plot da variável dependente em conjunto com cada variável explicativa (somente as categóricas).
# 
# ### Avalie o comportamento da distribuição da variável dependente com cada variável explicativa categórica:
# <ul style='font-size: 16px; line-height: 2; text-align: justify;'>
#     <li>As estatísticas apresentam mudança significativa entre as categorias?</li>
#     <li>O box plot apresenta alguma tendência bem definida?</li>
# </ul>

# ### Box-plot (Preço X Garagem)

# In[9]:


ax = sns.boxplot(y='precos', x = 'garagem', data = dados, orient='v', width=0.5)
ax.figure.set_size_inches(14,8)
ax.set_title('Preço do Imóvel em relação á Garagem', fontsize=20)
ax.set_ylabel('Valor', fontsize=16)
ax.set_xlabel('Quantidade de Garagem', fontsize=16)
ax


# ### Box-plot (Preço X Banheiros)

# In[10]:


ax = sns.boxplot(y='precos', x = 'banheiros', data = dados, orient='v', width=0.5)
ax.figure.set_size_inches(14,8)
ax.set_title('Preço do Imóvel em relação á Banheiros', fontsize=20)
ax.set_ylabel('Valor', fontsize=16)
ax.set_xlabel('Quantidade de Banheiros', fontsize=16)
ax


# ### Box-plot (Preço X Lareira)

# In[11]:


ax = sns.boxplot(y='precos', x = 'lareira', data = dados, orient='v', width=0.5)
ax.figure.set_size_inches(14,8)
ax.set_title('Preço do Imóvel em relação á Lareira', fontsize=20)
ax.set_ylabel('Valor', fontsize=16)
ax.set_xlabel('Quantidade de Lareiras', fontsize=16)
ax


# ### Box-plot (Preço X Acabamento em Mármore)

# In[12]:


ax = sns.boxplot(y='precos', x = 'marmore', data = dados, orient='v', width=0.5)
ax.figure.set_size_inches(14,8)
ax.set_title('Preço do Imóvel em relação á Mármore', fontsize=20)
ax.set_ylabel('Valor', fontsize=16)
ax.set_xlabel('Mármore', fontsize=16)
ax


# ### Box-plot (Preço X Andares)

# In[13]:


ax = sns.boxplot(y='precos', x = 'andares', data = dados, orient='v', width=0.5)
ax.figure.set_size_inches(14,8)
ax.set_title('Preço do Imóvel em relação aos Andares', fontsize=20)
ax.set_ylabel('Valor', fontsize=16)
ax.set_xlabel('Andares', fontsize=16)
ax


# ## Distribuição de frequências da variável *dependente* (y)
# 
# Construa um histograma da variável dependente (Preço).
# 
# ### Avalie:
# <ul style='font-size: 16px; line-height: 2; text-align: justify;'>
#     <li>A distribuição de frequências da variável dependente parece ser assimétrica?</li>
#     <li>É possível supor que a variável dependente segue uma distribuição normal?</li>
# </ul>

# https://seaborn.pydata.org/generated/seaborn.distplot.html?highlight=distplot#seaborn.distplot

# In[14]:


ax = sns.histplot(dados['precos'], kde = True)
ax.figure.set_size_inches(14,8)
ax.set_title('Distribuição de Frequência', fontsize=20)
ax.set_ylabel('Valores dos imóveis', fontsize=16)
ax


# ## Gráficos de dispersão entre as variáveis do dataset

# ## Plotando o pairplot fixando somente uma variável no eixo y
# 
# https://seaborn.pydata.org/generated/seaborn.pairplot.html?highlight=pairplot#seaborn.pairplot
# 
# Plote gráficos de dispersão da variável dependente contra cada variável explicativa. Utilize o pairplot da biblioteca seaborn para isso.
# 
# Plote o mesmo gráfico utilizando o parâmetro kind='reg'.
# 
# ### Avalie:
# <ul style='font-size: 16px; line-height: 2; text-align: justify;'>
#     <li>É possível identificar alguma relação linear entre as variáveis?</li>
#     <li>A relação é positiva ou negativa?</li>
#     <li>Compare com os resultados obtidos na matriz de correlação.</li>
# </ul>

# In[15]:


ax = sns.pairplot(dados, y_vars='precos', x_vars=['area','garagem', 'banheiros', 'lareira', 'marmore', 'andares'])
ax.fig.suptitle('Dispersão entre as Variábeis', fontsize=20, y=1.2)
ax


# In[16]:


ax = sns.pairplot(dados, y_vars='precos', x_vars=['area','garagem', 'banheiros', 'lareira', 'marmore', 'andares'], kind='reg')
ax.fig.suptitle('Dispersão entre as Variábeis', fontsize=20, y=1.2)
ax


# # <font color='red' style='font-size: 30px;'>Estimando um Modelo de Regressão Linear</font>
# <hr style='border: 2px solid red;'>

# ## Importando o *train_test_split* da biblioteca *scikit-learn*
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# In[17]:


from sklearn.model_selection import train_test_split


# ## Criando uma Series (pandas) para armazenar a variável dependente (y)

# In[18]:


y = dados['precos']


# ## Criando um DataFrame (pandas) para armazenar as variáveis explicativas (X)

# In[19]:


X = dados[['area','garagem', 'banheiros', 'lareira', 'marmore', 'andares']]


# ## Criando os datasets de treino e de teste

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811) 


# ## Importando *LinearRegression* e *metrics* da biblioteca *scikit-learn*
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# 
# https://scikit-learn.org/stable/modules/classes.html#regression-metrics

# In[ ]:





# ## Instanciando a classe *LinearRegression()*

# In[21]:


modelo = LinearRegression()


# ## Utilizando o método *fit()* para estimar o modelo linear utilizando os dados de TREINO (y_train e X_train)
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.fit

# In[22]:


modelo.fit(X_train, y_train)


# ## Obtendo o coeficiente de determinação (R²) do modelo estimado com os dados de TREINO
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.score
# 
# 
# ### Avalie:
# <ul style='font-size: 16px; line-height: 2; text-align: justify;'>
#     <li>O modelo apresenta um bom ajuste?</li>
#     <li>Você lembra o que representa o R²?</li>
#     <li>Qual medida podemos tomar para melhorar essa estatística?</li>
# </ul>

# In[23]:


print('R² = {}'.format(modelo.score(X_train, y_train).round(2)))


# ## Gerando previsões para os dados de TESTE (X_test) utilizando o método *predict()*
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.predict

# In[24]:


y_previsto = modelo.predict(X_test)


# ## Obtendo o coeficiente de determinação (R²) para as previsões do nosso modelo
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score

# In[25]:


print('R² = %s' % metrics.r2_score(y_test, y_previsto).round(2))


# # <font color='red' style='font-size: 30px;'>Obtendo Previsões Pontuais</font>
# <hr style='border: 2px solid red;'>

# ## Criando um simulador simples
# 
# Crie um simulador que gere estimativas de preço a partir de um conjunto de informações de um imóvel.

# In[26]:


entrada = X_test[0:1]
entrada


# In[27]:


modelo.predict(entrada)[0]


# In[28]:


area = 100
garagem = 2
banheiros = 2
lareira = 0
marmore = 0
andares = 1
entrada = [[area, garagem, banheiros, lareira, marmore, andares]]

print('{0:.2f} USD'.format(modelo.predict(entrada)[0]))


# # <font color='red' style='font-size: 30px;'>Métricas de Regressão</font>
# <hr style='border: 2px solid red;'>

# ## Métricas da regressão
# <hr>
# 
# fonte: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
# 
# Algumas estatísticas obtidas do modelo de regressão são muito úteis como critério de comparação entre modelos estimados e de seleção do melhor modelo, as principais métricas de regressão que o scikit-learn disponibiliza para modelos lineares são as seguintes:
# 
# ### Erro Quadrático Médio
# 
# Média dos quadrados dos erros. Ajustes melhores apresentam $EQM$ mais baixo.
# 
# $$EQM(y, \hat{y}) = \frac 1n\sum_{i=0}^{n-1}(y_i-\hat{y}_i)^2$$
# 
# ### Raíz do Erro Quadrático Médio
# 
# Raíz quadrada da média dos quadrados dos erros. Ajustes melhores apresentam $\sqrt{EQM}$ mais baixo.
# 
# $$\sqrt{EQM(y, \hat{y})} = \sqrt{\frac 1n\sum_{i=0}^{n-1}(y_i-\hat{y}_i)^2}$$
# 
# ### Coeficiente de Determinação - R²
# 
# O coeficiente de determinação (R²) é uma medida resumida que diz quanto a linha de regressão ajusta-se aos dados. É um valor entra 0 e 1.
# 
# $$R^2(y, \hat{y}) = 1 - \frac {\sum_{i=0}^{n-1}(y_i-\hat{y}_i)^2}{\sum_{i=0}^{n-1}(y_i-\bar{y}_i)^2}$$

# ## Obtendo métricas para o modelo com Temperatura Máxima

# In[29]:


EQM = metrics.mean_squared_error(y_test, y_previsto).round(2)
REQM = np.sqrt(metrics.mean_squared_error(y_test, y_previsto)).round(2)
R2 = metrics.r2_score(y_test, y_previsto).round(2)

metricas_regressao = pd.DataFrame([EQM, REQM, R2], ['EQM', 'REQM', 'R2'], columns=['Métricas'])
metricas_regressao


# # <font color='red' style='font-size: 30px;'>Salvando e Carregando o Modelo Estimado</font>
# <hr style='border: 2px solid red;'>

# ## Importando a biblioteca pickle

# In[30]:


import pickle


# ## Salvando o modelo estimado

# In[31]:


output = open('modelo_precos_imoveis', 'wb')
pickle.dump(modelo, output)
output.close()


# ### Em um novo notebook/projeto Python
# 
# <h4 style='color: blue; font-weight: normal'>In [1]:</h4>
# 
# ```sh
# import pickle
# 
# modelo = open('modelo_preço','rb')
# lm_new = pickle.load(modelo)
# modelo.close()
# 
# area = 38
# garagem = 2
# banheiros = 4
# lareira = 4
# marmore = 0
# andares = 1
# 
# entrada = [[area, garagem, banheiros, lareira, marmore, andares]]
# 
# print('$ {0:.2f}'.format(lm_new.predict(entrada)[0]))
# ```
# 
# <h4 style='color: red; font-weight: normal'>Out [1]:</h4>
# 
# ```
# $ 46389.80
# ```

# In[ ]:




