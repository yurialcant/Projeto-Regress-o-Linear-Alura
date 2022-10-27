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
Primeiro iremos utilizar o gráfico do tipo Box-plot em nossa variável dependente(y), para verificarmos o comportamento da distribuição da variável dependente, verificando a existência de outliers e se o gráfico demonstra alguma tendência.
![Captura de Tela (233)](https://user-images.githubusercontent.com/102321564/198375312-4032b875-8171-4e36-8ccf-2e8e3c7f4822.png)
Em nosso boxp-lot, não possuímos a presença de outliers, agora iremos realizar outros box-plots para avaliarmos nossa variável dependente juntamente com outras variáveis explicativas categóricas.
![Captura de Tela (234)](https://user-images.githubusercontent.com/102321564/198375871-5d564c16-a659-445e-80cf-1824645c549d.png)

![Captura de Tela (235)](https://user-images.githubusercontent.com/102321564/198376037-45bda128-c8d9-4a3d-9622-df10373b4a85.png)
![Captura de Tela (236)](https://user-images.githubusercontent.com/102321564/198376111-d41b6c33-2a81-4be8-ad4b-4e533cee4302.png)
![Captura de Tela (237)](https://user-images.githubusercontent.com/102321564/198376212-ce0513cb-3b53-47ce-8f9b-63fccc52b654.png)

![Captura de Tela (238)](https://user-images.githubusercontent.com/102321564/198376307-414f404c-6b6c-4464-811d-ab5b0a3de3f0.png)

Agora nos utilizaremos de um histograma para analisar a distribuição de frequência da variável dependente.
![Captura de Tela (239)](https://user-images.githubusercontent.com/102321564/198376719-8b87ae45-a45b-435f-a466-cc1a1a43d8ad.png)
Como podemos ver nosso gráfico fica bem próximo de uma assemetria, por fim para completar nossas análises gráficas iremos realizar a construção de um gráfico de dispersão entre as variáveis presentes em nosso DataSet, utilizaremos para isso o pairplot, fixando somente uma variável no eixo y.

![Captura de Tela (240)](https://user-images.githubusercontent.com/102321564/198377201-ade54f8b-b86e-4efc-b45b-2dd28241b610.png)

<h1>Estimando um Modelo de Regressão</h1>
Nesta etapa iremos estimar um modelo de regressão, para isso utilizaremos o train_test_split da biblioteca scikit-learn, com ela importada iremos criar uma series do pandas para armazenar nossa variável dependente, depois iremos criar um DataFrame para armazenar as variáveis explicativas e por fim criaremos o nosso dataset de treino e de teste.
![Captura de Tela (241)](https://user-images.githubusercontent.com/102321564/198378046-5acbb1a0-eac6-4fb3-b18e-459f3b76bff7.png)
Agora utilizaremos outra ferramenta da biblioteca scikit-learn, desta a LinearRegression, para isso iremos instanciar esta classe após o import, e em seguida utilizaremos o método fit, para estimar o modelo linear utilizando os dados de treino, também iremos obter o coeficiente de determinação do modelo estimado com os dados de treino(lembrando que o resultado do coeficiente sempre deve estar próximo a 1, para o modelo ser viável).

![Captura de Tela (242)](https://user-images.githubusercontent.com/102321564/198378582-2df83eea-e65a-4bb8-92cf-3b869b923cb7.png)

Após isso iremos gerar as previsões para os dados de teste utilizando o método predict e obter o coeficiente de determinação para as previsões do nosso modelo com os dados de teste.

![Captura de Tela (243)](https://user-images.githubusercontent.com/102321564/198379189-49df4ceb-d104-444b-9438-b646131e8015.png)
