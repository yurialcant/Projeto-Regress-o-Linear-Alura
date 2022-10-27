# Projeto-Regressao-Linear-Alura
| 🪧 Vitrine.Dev |     |
| -------------  | --- |
| ✨ Nome        | Projeto de Regressão Linear
| 🏷️ Tecnologias | Python, Pandas, numpy, seaborn, sklearn, pickle
| 🚀 URL         |https://github.com/yurialcant/Projeto-Regress-o-Linear-Alura
| 🤿 Desafio |https://cursos.alura.com.br/
#vitrine-dev #vitrinedev#alura

<h1 align ="center"> Projeto de Regressão Linear</h1>
<p align="center">
<img src="http://img.shields.io/static/v1?label=STATUS&message=PROJETO%20COMPLETO&color=GREEN&style=for-the-badge"/>
</p>

Este projeto foi elaborado, durante o curso de Regressão Linear parte I, nosso objetivo é criar um 
modelo de previsão de preço de casas nos utilizando como base as informações do nosso DataSet(" https://www.kaggle.com/greenwing1985/housepricing"), 
para isso nos utilizaremos de algumas ferramentas como as funções de análise do DataSet fornecida pela biblioteca Pandas, a construção de gráficos
com o auxílio da biblioteca Seaborn e por fim a construção do modelo de Regressão Linear com auxílio da biblioteca Scikit-Learn.

<h1>Conhecendo o DataSet</h1>
<h2>Análises Preliminares</h2>

Nosso primeiro passo é importar as bibliotecas que faremos uso durante o projeto, portanto realizamos o import do pandas, numpy, matplotlib, seaborn, sklearn e o pickle.
![Captura de Tela (229)](https://user-images.githubusercontent.com/102321564/198372902-4b7e2a95-d7ad-436b-84d0-412104fd585c.png)
Em seguida realizamos a leitura de nossos dados e verificamos o tamanho do nosso dataset nos utilizando da função shape.
![Captura de Tela (230)](https://user-images.githubusercontent.com/102321564/198373213-83989384-8056-47a2-b555-747b4f804231.png)
Agora partiremos para nossa análise preliminar, o primeiro passo é obter as estatísticas descritivas como a média, desvio padrão, valor mínimo e valor máximo.

![Captura de Tela (231)](https://user-images.githubusercontent.com/102321564/198373581-497e3cf6-23f3-46ab-b25f-58ff4df1ba3a.png)
Agora analisaremos nossa Matriz de correlação que nos ajuda a identificar o coeficiente de correlação que é uma medida de associação linear entre duas variáveis
e situa-se entre -1 (associação negativa perfeita) e +1(associação positiva perfeita), para isso utilizaremos a função no pandas corr junto do round(4), para arrendondar
nossos valores em 4 casas decimais.
![Captura de Tela (232)](https://user-images.githubusercontent.com/102321564/198374224-3a76aada-f53e-4a2e-b20c-2129886c2aa0.png)

<h1>Análises Gráficas</h1>
<h2>Analisando o Comportamento da Variável Dependente(Y)</h2>
