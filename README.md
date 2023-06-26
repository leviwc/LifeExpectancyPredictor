# LifeExpectancyPredictor

Utilizando World Development Indicators da base the world database https://databank.worldbank.org/source/world-development-indicators/preview/on# esse trabalho utiliza o algoritmo KNN, Naive Bayes e MLP parar prever a expectativa de vida de um pais a partir de indicadores de desenvolvimento



# Pre processing

Para o pré processamento dos dados, temos o csv de cada etapa na pasta prints/

Inicialmente são removidas as colunas desnecessárias 'Country Code', 'Series Name' ficando apenas com o nome do país, código do indicador e anos

Queremos que os indicadores sejam as features, e o indicador de "life expectancy at birth seja a classe a ser predita, então vamos pivotear a base de dados de forma que os indicadores sejam as colunas.

Agora temos uma escolha com prós e contras, se iremos isolar o ano ou utilizar o dataset para todos os anos, formando então indices de pais - ano

O lado positivo de utilizar um ano só é mais dados podendo criar modelos mais robustos, o negativo é aumentar o tempo de processamento e criar dados duplicados.
Sendo assim fiz o teste com ambas as possibilidades

Agora vamos lidar com os dados nulos, foram testados algumas formas, preenchendo com 0, eliminando todas as linhas sem dados, eliminando todas as colunas com algum dado nulo. E a que teve um sucesso maior foi inicialmente retirar linhas com muitos dados faltando (no caso 14 teve mais sucesso) e após isso eliminar todas as caracteristicas com dados nulos

Também foram retirados todas as linhas com o "life expectancy at birth nulo", porque esses casos geralmente também tinham muitos outros nulos e n teria um beneficio mante-los adicionando uma predição como dado

Agora vamos classificar a coluna de expectativa de vida para facilitar as predições,
inicialmente foi testado com classes de tamanho de idade 2: (60-62), (63 - 64), ...
Isso teve resultados de predição muito ruins, com 13% - 18% testando com o metodo leave one out
Então seguí com classes de tamanho 5 (60 - 64), (65 - 60) obtendo um resultado inicial de em 50% no KNN

Após isso o vetor foi normalizado para facilitar a aplicação nos modelos, utilizando o algoritmo MinMaxScaler ficamos com um dataset tratado, com valores de 0 a 1 e com a classes definidas para o life expectancy at birth

# Aplicando Modelos

Com o dataset tratado foram testados com a técnica leave_one_out para determinar acuracia. Após receber o resultado, eram feitas mudanças no dataset de forma que a acuracia aumente

# KNN

Inicialmente os dados foram tratados com testes seguindo o KNN,

Com o ano 2019 apenas:

que teve acuracia de 13% para classes de tamanho 2, e 50% para classes de tamanho 5

Após as alterações e testes com K de 1 a 100 e melhor resultado foi atingido 23% para classe 2 e 63% para classes de tamanho 5

Com todos os anos:

Com os dados tratados, a accuracia inicial foi de:
83%, porém isso se deu principalmente pois os dados de um pais mudam muito pouco de um ano para o seguinte, e utilizando o leave one out acaba enviesando o teste
por isso retirei no treino todas as instancias de um pais quando ele é o escolhido

53% KNN

Após melhorias no processamento:

66%

# Naive Bayes
Utilizado após dado tratado anteriormente, teve uma acuracia de cerca de 43%
Foi tentado separar as features em 3 classes para tentar melhor a acuracia mas apenas diminuiu


# MLP
Para 2019 apenas
Utilizando 100, 100 hidden layers e 1000 max_iteration inicialmente, o resultado foi cerca de 33% inicialmente

Para todos os anos
Aqui a base de dados cresceu, foi de 266 para 1400, então rodar o leave one out começou a demorar também
Para tentar otimizar esse processo foi adicionado execução paralela no leave one out utilizando o maximo de recursos

E no fim utilizados apenas uma camada com 14, max_iter 3000, activation= 'Relu'

Com resultado supreendente de 68%!

Tentando com 64 nós, o resultado foi também de 68%


# Resultado

classes de idade tamanho 5:

2019:<br/>
Final K: 21 <br/>
KNN: 64.13043478260869%<br/>
Naive Bayes: 42.391304347826086%<br/>
MLP: 64.67391304347827%<br/>

Todos os anos:<br/>
Após testar k de 1 a 100 temos:<br/>
Final K: 9<br/>
KNN: 66.21717530163235%<br/>
Naive Bayes: 40.17033356990773%<br/>
MLP: 68.27537260468416%<br/>

classes de idade tamanho 2:<br/>

2019:<br/>
Final K: 1<br/>
KNN: 41.30434782608695%<br/>
Naive Bayes: 23.369565217391305%<br/>
MLP: 32.608695652173914%<br/>

Todos os anos:<br/>
Final K: 13<br/>
KNN: 38.963804116394606%<br/>
Naive Bayes: 22.995031937544358%<br/>
MLP: 35.060326472675655%<br/>




