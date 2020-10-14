# Computer-Vision-Lego-Classifier--Python
Deteção e Classificação de Legos

Autores: Miguel Rosa; Ricardo Pinto; Agostinho Pires

Introdução

No âmbito da cadeira de Visão por computador na indústria foi desenvolvido um algoritmo de
deteção e classificação de legos. Este algoritmo recebe imagens com multiplos e diversos legos e procede à identificação e classificação de cada lego em função da sua forma e cor.

Tratamento inicial

O algoritmo começa com o redimensionamento da imagem RGB original para uma imagem
400x520. São posteriormente criadas mais quatro imagens correspondendo à conversão da
imagem original para os formatos LUV, YUV, HSV e GRAY uma vez que todos estes formatos
possuem caraterísticas individuais úteis no processamento a seguir realizado.

Procedimento

![Capture](https://user-images.githubusercontent.com/40301612/96031792-bbc21580-0e55-11eb-889c-5eb9b2fc8e78.PNG)

1. Remoção de background

Para a remoção de background é utilizada a imagem em formato RGB, à qual é
inicialmente aplicado um filtro gaussiano.
É de seguida utilizado “Watershed” em cada componente do formato RGB, o que torna
possível a deteção de peças de cores mais claras. Isto permite a distinção entre as peças
e o fundo e cria uma máscara com as peças realçadas a branco.
É ainda utilizado “FloodFill” num ponto de referência normalmente corresponde ao
fundo da imagem, criando uma máscara com o fundo da imagem realçado a preto.
O produto final desta fase corresponde à multiplicação bit a bit destas 4 mascaras, uma
vez que todas as máscaras vão ter a região do fundo e alguns legos a branco, se esses
legos não estiverem a branco em todas as máscaras eles são eliminados da seleção.
Nesta fase, é perdido o lego branco, porque devido às grandes diferenças de
luminosidade e proximidade da cor de fundo não foi possível encontrar uma forma de
o separar.

2. Equilíbrio de luminosidade

Esta etapa visa minimizar a influência de variações de luminosidade no algoritmo.
Foram recolhidas amostras da média da cor de fundo de várias imagens do data set
fornecido, este valor tem pigmentação azulada em alguns dos casos, então torna-se
impossível equilibrar a luminosidade com o formato RGB.
Usou se então o formato YUV em que o Y é uma camada muito fiel á luminosidade.
Calculamos a média da cor do fundo no formato YUV e somamos à camada Y o desvio
que a imagem a ser tratada tem para com a média de todas as imagens juntas.
Isto permite ter uma maior estabilidade na cor dos legos.
Após isto convertemos a imagem YUV para o formato RGB e continuamos o processo
com ela.

3. Separação de peças e recolhimento de posições

De seguida é feita uma deteção de diferenças de contraste utilizando o algoritmo
“Canny” para tentar separar eventuais legos que estejam em contacto uns com os
outros.
A máscara criada pelo algoritmo é ampliada utilizando a função dilate do openCV e
aplicada à máscara resultante da remoção de background. Ao resultado é aplicada a
função erode de modo a eliminar ruido indesejado.
Torna-se agora possível descobrir o centroide das peças utilizando a função
“findContours” e calculando os momentos da imagem.
Esta separação é por vezes demasiado ofensiva podendo dividir o mesmo lego em dois
devido a reflexos na face do lego.

4. Segmentação individual das peças

Neste momento tem se um conjunto de pontos que representam o centro dos legos,
sendo que alguns podem estar a tentar representar o mesmo lego, então usando a
função “FloodFill”, desta vez mais sensível, aplicada em cada centroide podemos
eliminar redundância uma vez que a mascara devolvida vai ser a mesma para pontos no
mesmo lego.
Calculamos novamente os centroides e cantos nestas novas máscaras usando
“approxPolyDP” e temos verdadeira informação sobre os legos.
Cada mascara é também guardada para usar na fase seguinte.

5. Analise da Cor

Através de um programa extra foram recolhidas amostras das cores reais dos legos e
calculada uma média de cada cor individualmente.
Aqui apenas precisamos de calcular a média da cor da imagem original usando a
máscara do lego obtida no passo anterior e comparar com todos os valores e selecionar
a mais provável.
Devido à calibração da luz conseguimos uma taxa de sucesso de 75% com este método.

6. Análise da Forma

O algoritmo usado para a deteção dos pontos dos cantos apenas nos permite avaliar as
formas de 1 até 12.
Para estes, é calculada a razão entre o comprimento e a largura(K), sendo que só existem
6 valores possíveis (1, 2, 3, 4, 6, 8), agora compara-se com as áreas que os legos com um
certo valor K podem ter, que são bastante distantes entre si.
Sabendo a razão entre o comprimento e a altura e a área é possível classificar com os
valores de 1 a 12 a forma do lego.


