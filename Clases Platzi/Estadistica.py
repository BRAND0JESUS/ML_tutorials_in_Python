#%%
# Carga de datos de fuentes locales Operative System
import os
os.name  # Nombre del sistema operativo usado
os.getcwd()  # directorio activo get workin directory
os.listdir('.')   # Listar elementos de la carpeta
# Funcion para configurara la ruta de trabajo
# path = 'c:\\Users\\Brando\\Desktope'
# os.chdir(path)        # Validar la ruta

import pandas as pd
df = pd.read_csv('bicicletas-compartidas.csv')
df.head()       # funcion implicita de pandas para ver algunos mienbros
df.columns      # lista de las columnas
df.shape        # nymero de fil y col


import tensorflow as tf     # principal para procesamiento de imagenes

fashion_mnist = tf.keras.datasets(fashion_mnist)  # base de datos de la libreria
(imagenes, categorias) = fashion_mnist.load_data()[0]  # lo que vamos a poder cargas
imagenes.shape      # La imagen debe ser matrices de pixel (60000,28,28) cada columna es valor de un pixel
imagenes[0]     # ver las variables o color de cada pixel
imagenes[0].shape # dimensiones
imagenes[0].shape, categorias[0]   # categoria (labels) asociada al primer registro 9 


# %%

import numpy as np      
import datetime 
from datetime import datetime

type(0.2)

# operadores logicos

(5<7) & (7>5)   # y
(5<3) | (7>9)   # o

# listas -> guarda diferentes tipos de datos, stg, int, list
lista_1 = ['1', 0, [1,2]]
lista_1[2][1]       # accede al 2 de la lista dentro de la lista







# %%        EXPERIMENTO ALEATORIO
# esta definido por el universo
universo = ['cara','sello']

# experimento bernoulli -> binario, probalbilidad de p asociada

p_cara = 1/2

from scipy.stats import bernoulli

universo[bernoulli.rvs(p_cara)]#random variable sample() tenemos varios resultados = 0-1
bernoulli.rvs (p=p_cara, size=10) # accedemos al inibverso de datos y enviamos el resultado del experimento
sum(bernoulli.rvs (p=p_cara, size=10))

from scipy.stats import binom  # uncion binomial

binom.rvs(p=p_cara, n = 10, size=100) # p, numero de experimentos, repeticiones del experimento

# distribucion de probabilidad -> agunos valores tienen mas prob de efectuarse dentro de a distribucion


import pandas as pd

pd.Series(binom.rvs(p=p_cara, n = 10, size=100)).value_counts()  # lista de como se repite cada uno de los valores (frecuencia de repeticion)


# %%

import numpy as np
import pandas as pd
import scipy
import scipy.stats

df = pd.read_csv('bicicletas-compartidas.csv')
df.columns

y = df['bicis-compartidas'].values  # variable sobre la cual vamos a generar el analisis
y

y = np.where( y == 0, 1, y)  # Remplazar los 0 por 1, y sino concervas y

np.min(y)
np.max(y)

# medida de tendencia central
# promedio = sum(yi)/n
np.mean(y)
np.sum(y)/len(y)

# media geometrica armonica
scipy.stats.mstats.hmean(y)  # se trae mediciones de estats, diferentes media armonica, 

# mediana valor que divide en 2 (percentil 50 en cada lado 50/50)
np.median(y)

# moda = valor de y con la maxima frecuencia

moda = np.nan # valor nulo
valores, conteo_valores = np.unique(y, return_counts = True)  # contero de valores
pos = np.argmax(conteo_valores)  # dentro de conteo ver el maximo valor
moda = valores[pos]
moda

# medidas de dispersion  
# 
# Desviacion estandar = que tanto se aleja xi de su valor promedio
np.std(y)


# Revisiones
y_alterado = y.copy()   # copia de y original
y_alterado[y_alterado == max(y_alterado)] = 10000000

print(np.mean(y))
print(np.mean(y_alterado))

print(np.median(y))         # no se ven altyerado los resulatados 
print(np.median(y_alterado))

#%%

import pandas as pd
import numpy as np
import scipy.stats

df = pd.read_csv('bicicletas-compartidas.csv')
df.columns

# Frecuencias para variables categoricas
y_cat = df['cuartil-ano']  # pose numeros pero no es var bumerica  # df['cuartil-ano'].value_counts()
y_cat = y_cat.apply(lambda x: 'Cat-' + str(int(x)))  # x es cada uno de los valores dentro de y categ y se le agrega texto
y_cat.head()

# encontrar las freq asociadas
valores, conteo_freq = np.unique(y_cat, return_counts= True)
valores, conteo_freq
tabla_frec = dict(zip(valores, conteo_freq))  # dicionario
tabla_frec

# Frecuencias para variables numericas

y_num = df['viento'].copy()
np.min(y_num), np.max(y_num)

np.percentile(y_num, q=50)   # percentiles q=50 es mediana

# quartiles medida similar a percentiles pero son valores puentuales que acumulan cierta probabilidad
valores = [0,25,50,75,100]
np.percentile(y_num, q = valores)  # nuemors dentro del y que acumulan los porcentajes de VALORES son los siguinets

# quintiles
valores = [0,20,40,60,80,100]
np.percentile(y_num, q = valores)

# deciles
valores = list(range(0,110,10))
np.percentile(y_num, q = valores)

# Valores atipocos estan asociados a los percentiles

y = df['bicis-compartidas']  
y.describe()  # permita ver fucniones etadisticas

# outlier todo valor que caiga fuera de rango

Q1 = np.percentile(y_num, q=25)
Q3 = np.percentile(y_num, q=75)

RI = Q3 - Q1       # rango intercuartilico
lim_inf = Q1-1.5*RI
lim_sup = Q3+1.5*RI

[lim_inf, lim_sup]


import matplotlib.pyplot as plt

plt.hist(y)


#%%

import pandas as pd
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('bicicletas-compartidas.csv')
df.columns
########################################################################################
#                                   Var Categoricas

y = df['cuartil-ano'].apply(lambda x:'cuartil-' + str(int(x)))

fig, ax = plt.subplots()
ax.bar(y.value_counts().index, y.value_counts())  #diagrama e barras sobre eje y; 
ax.set_xlabel('Cuartiles del anio')
ax.set_ylabel('Frecuencia')

# colorear barras
fig, ax = plt.subplots()
ax.bar(y.value_counts().index, y.value_counts())  #diagrama e barras sobre eje y; 
ax.set_xlabel('Cuartiles del anio')
ax.set_ylabel('Frecuencia')
ax.patches[3].set_facecolor('red')    # cada una de las barra se guarda en patches
# barras de mayor frecuencia al lado izquierdo

# diagrama de pay
fig, ax = plt.subplots()  # definir figua y eje
ax.pie(y.value_counts(), labels = y.value_counts().index)  #diagrama e barras sobre eje y; 
ax.set_title('Diagrama de pie')


#########################################################################################
#                       variables numericas
 # Histograma
y = df['viento']

fig, ax = plt.subplots()
ax.hist(y, bins = 30)  # bin = cuantas divisiones tiene que mostrar la variable continua
ax.set_xlabel('Viento')
ax.set_ylabel('Frecuencia')
plt.axvline(np.mean(y), c = 'red', linestyle='-.', label = 'Promedio')  # axvertical
plt.axvline(np.mean(y) + np.std(y), c = 'k', linestyle='--', label = '+ 1 desviacion')  # axvertical
plt.axvline(np.mean(y) - np.std(y), c = 'g', linestyle='--', label = '- 1 desviacion')  # axvertical
ax.grid(color = 'gray', linestyle = '-', linewidth = 0.5)
ax.legend(loc = 'best')

# Boxplor o caja
y = df['bicis-compartidas']
fig, ax = plt.subplots()
# ax.boxplot(x=y)         # la varia x es la var de interes (y)
sns.boxplot(x='cuartil-ano', y = 'bicis-compartidas', data = df)   # boxplot permite mapera comportamiento de bicis compartidas en funcion de cuartil anio

# graficos de dos variables simpultaneos
fig, ax = plt.subplots()
cs = ax.scatter(df['viento'], df['bicis-compartidas'], c = df['bicis-compartidas'], alpha= 0.03,cmap = 'viridis')  # se necesita dos variables continuas, alpha = difumina cada punto
fig.colorbar(cs, ax=ax) 
ax.set_title('Distribucion conjunta de viento y bicis compartidas')
ax.set_xlabel('Viento')
ax.set_ylabel('Bicis compartidas')

# %%                                    TEOREMA DE BAYES


import pandas as pd
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('juego-azar.csv',sep=';')  
df.columns

# probabilidades univariadas
df.numero.value_counts()/len(df)        # numero es columna de df, se cuenta las frecuencias univariadas

df.color.value_counts()/len(df)        # probabilidad que salga la esfera de cada color

# Probabilidad conjunta bivariada, de algun color y numero especifico
df.groupby(['color','numero']).size()#/len(df)  # agupar color y numero, hacer conteo

# PROBABILIDAD CONDICIONAL

# P(A|B) = P(B|2) = 1/3    evneto A dado B; ver numero de casos (quitar len) para ver que N Blancas con numero 2 es 1 y N total de esfereas con 2 son 3
1/3

##############################################################################################
#                                   TEOREMA DE BAYES
############################################################################################

P_blanc = 4/10 # 4 bolas blanc de 10 origunales

# p(P_blanc|1) + (P_blanc|2) + (P_blanc|3)
(1/4)*(4 /10) + (1/3)*(3/10) + (2/3)*(3/10)

#p(blanco) = p(P_blanc|1)*p(1) + (P_blanc|2)*p(2) + (P_blanc|3)*p(3) 

# Tablas de contingencia
a = pd.crosstab(index = df.color, columns = df.numero, margins = True)
b = pd.crosstab(index = df.color, columns = df.numero, margins = True)/len(df)
a, print (str('\n')), b


# %%


import pandas as pd
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binom

p = 0.3         # probab de exito
n = 8       # numero de intentosnuero de exitos

x = list(range(0,9,1))       # variable aleatorio
y = list([])        # distribucion o valor probabilidad

for i in x:
    y.append(binom. pmf(i, p = p, n = n)) # probability mas funtion ( para x particular calcula su prob asociada)

fig, ax = plt.subplots()
ax.bar(x,y)
ax.set_ylabel('Probabilidad discreta')

np.sum(y)  # la suma de probabilidades es 1

media = np.average(x, weights=y)  # promedio ponderado por que se calcula el valor que va a 
                                #tomar cada uno de los valores de x de la distribucion ponderada por su probabilidad
varianza = np.average(((x-media)**2),weights = y)   # ponderado por la prob de ocurrencia de cada x, con wegths se mukltiplica para cada error

fig, ax = plt.subplots()
ax.bar(x,y)
ax.set_ylabel('Probabilidad discreta')
ax.axvline(x = media, c = 'g', linestyle= '--', label = 'Valor Esperado')
ax.axvline(x = media + 3*np.sqrt(varianza), c = 'r', linestyle= '--', label = 'desviacion Estandar')
ax.legend()


################################################################################
N = 10000         # entre mas grande N, mucho mas continua es la distribucion y se puede aproximar mejor la var cont
x = list(range(0,N+1,1))   # muestra aleatoria de 100 variables
y  = ([])     # PROBABILIDAD

for i in x:
    #y.append(binom.pmf(i, p=0.3, n = N))  # p = parametro poblacional
    y.append(binom.cdf(i, p=0.3, n = N))   # cumulative density funtion
fig, ax = plt.subplots()
ax.plot(x,y)
ax.set_title('Probabilidad continua')
from scipy.stats import norm
media, var, skew, kurt = norm.stats(moments = 'mvsk' )               # media, varian , asimetria, kurtosis; norm para funciones normales
                                        # momentos de la fincuion son las derivadas de la funcion
                                        # la funcion norm = de distribucion nos permite ver los valores de caraterizacion

media, var


#%%
# import pandas as pd
# import numpy as np
# import scipy.stats
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.stats import bernoulli
p = 0.3     # parametro pob;lacional de la distrib, probabablidad de exito
data = bernoulli.rvs(p, size = 100)     # trandom variable samplo, p y numero de muestras
len (data)

mean, var, skew, kurt = bernoulli.stats(p, moments='mvsk')  # var o desviacion = p*(1-p)
mean, var, skew, kurt

# ax = sns.distplot(data, bins = 30, kde = False, color = 'b')         # displot = distribution plots, bins de visualizacion, kde = ver lineas o puntos
# ax.set(xlabel = 'Distribucion de bernulli', ylabel = 'Frecuencia')
# ax.axvline(x = mean, linestyle = '--', label = 'Media')
# ax.legend()

#########################################################################################
from scipy.stats import binom
p = 0.3     # parametro pob;lacional de la distrib, probabablidad de exito
N = 10        # numero de veces que queremos repetri la variable bernoulli
data = binom.rvs(n = N, p = p, size = 100)     # trandom variable samplo, p y numero de muestras
len (data)

mean, var, skew, kurt = binom.stats(p = p, n = N,  moments='mvsk')  # var o desviacion = p*(1-p)
mean, var, skew, kurt

# ax = sns.distplot(data, bins = 30, kde = False, color = 'b')         # displot = distribution plots, bins de visualizacion, kde = ver lineas o puntos

# ax.set(xlabel = 'Distribucion de Binomial', ylabel = 'Frecuencia')
# ax.axvline(x = mean, linestyle = '--', label = 'Media')
# ax.legend()


#########################################################################################
from scipy.stats import nbinom                  # Distribucion geometrica, binomial negativa
p = 0.3     # parametro pob;lacional de la distrib, probabablidad de exito
N = 10      # numero de veces que queremos repetri la variable bernoulli
data = nbinom.rvs(n = N, p = p, size = 100)     # trandom variable samplo, p y numero de muestras
len (data)

mean, var, skew, kurt = binom.stats(p = p, n = N,  moments='mvsk')  # var o desviacion = p*(1-p)
mean, var, skew, kurt

# ax = sns.distplot(data, bins = 30, kde = False, color = 'b')         # displot = distribution plots, bins de visualizacion, kde = ver lineas o puntos
# ax.set(xlabel = 'Distribucion negativa', ylabel = 'Frecuencia')
# ax.axvline(x = mean, linestyle = '--', label = 'Media')
# ax.legend()

#########################################################################################
from scipy.stats import poisson
lambda_p = 100    # distribucion de poisson, 3 exitos en unidad de tiempo
data = poisson.rvs(mu = lambda_p, size = 100)     # trandom variable samplo, p y numero de muestras
len (data)

mean, var, skew, kurt = poisson.stats(mu = lambda_p,  moments='mvsk')  # var o desviacion = p*(1-p)
mean, var, skew, kurt

# ax = sns.distplot(data, bins = 30, kde = False, color = 'b')         # displot = distribution plots, bins de visualizacion, kde = ver lineas o puntos
# ax.set(xlabel = 'Distribucion de Poisson', ylabel = 'Frecuencia')
# ax.axvline(x = mean, linestyle = '--', label = 'Media')
# ax.legend()

#########################################################################################
from scipy.stats import expon
data = expon.rvs( size = 100000000)     # cambiar el tamaño para ver mejor la continuidad de la avar
len (data)

mean, var, skew, kurt = expon.stats(moments='mvsk')         # valor de media, var = 1 
mean, var, skew, kurt

# ax = sns.distplot(data, bins = 500, kde = False, color = 'b')         # displot = distribution plots, bins de visualizacion, kde = ver lineas o puntos
# ax.set(xlabel = 'Distribucion Exponencial', ylabel = 'Frecuencia')
# ax.axvline(x = mean, linestyle = '--', label = 'Media')
# ax.legend()



#########################################################################################
from scipy.stats import norm
mean = 0   # si es estandar
var = 1     # si es estandar
data = norm.rvs( size = 100000000)     # cambiar el tamaño para ver mejor la continuidad de la avar
len (data)

mean, var, skew, kurt = norm.stats(moments='mvsk')         # valor de media, var = 1 
mean, var, skew, kurt

# ax = sns.distplot(data, bins = 500, kde = False, color = 'b')         # displot = distribution plots, bins de visualizacion, kde = ver lineas o puntos
# ax.set(xlabel = 'Distribucion Normal estandar', ylabel = 'Frecuencia')
# ax.axvline(x = mean, linestyle = '--', label = 'Media')
# ax.legend()
# Todo dato que que asumamos normal en su distribucionposee 90% de probab entre 3 desviacion a la derecha e izquei


#########################################################################################
from scipy.stats import uniform

data = uniform.rvs( size = 100000000)     # cambiar el tamaño para ver mejor la continuidad de la avar
len (data)

mean, var, skew, kurt = uniform.stats(moments='mvsk')      
mean, var, skew, kurt  # Una funcion uniforme esta distribuida entre parametros de A y B

ax = sns.distplot(data, bins = 500, kde = False, color = 'b')         # displot = distribution plots, bins de visualizacion, kde = ver lineas o puntos
ax.set(xlabel = 'Distribucion Uniforme 0-1', ylabel = 'Frecuencia')
ax.axvline(x = mean, linestyle = '--', label = 'Media')
ax.legend()


# %%            Estandarización

import pandas as pd
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('iris-data.csv', index_col = 0)
df.columns

df.tipo_flor.value_counts()  # caracteristica de cada especie y value cont para ver sus valores
y = df['lar.petalo']
y

#grafico
fig, ax = plt.subplots()
ax.set_title('Variable original')
ax.hist(y, bins = 30)  # histograma con distrib bimodal con 2 picos superiores, indica que posee 2 poblaciones diferentes
ax.axvline(x = np.mean(y), c = 'k', label= 'media', linestyle = '--')
ax.axvline(x = np.mean(y) + np.std(y), c = 'r', label= 'desviacion est.', linestyle = '--')
ax.legend()


# Estandarizar 
# 1 Centralizar la var (se resta a cada y la media y desv standar
fig, ax = plt.subplots()
ax.set_title('Variable original')
ax.hist(y -  np.mean(y), bins = 30)
ax.axvline(x = np.mean(y -  np.mean(y)), c = 'k', label= 'media', linestyle = '--')
ax.axvline(x = np.mean(y) + np.std(y), c = 'r', label= 'desviacion est.', linestyle = '--')
ax.legend()
# Reduaccion de la variable = dividir cada (y -mean/desv)
fig, ax = plt.subplots()
ax.set_title('Variable estandarizada')      # se posee una medida adimencional, media (0) y var (1)
ax.hist((y -  np.mean(y))/np.std(y), bins = 30)
ax.axvline(x = np.mean((np.mean(y -  np.mean(y))/np.std(y))), c = 'k', label= 'media', linestyle = '--')
ax.axvline(x = np.mean((np.mean(y -  np.mean(y))/np.std(y))) + np.std((y -  np.mean(y))/np.std(y)), c = 'r', label= 'desviacion est.', linestyle = '--')        # se le suma la nueva media par conservar el efecto de estandarizacion sobre la media
ax.legend()

###############################################################
# Covarianza asociada a dos variables
fig, ax = plt.subplots()
ax.scatter(df['lar.petalo'], df['lar.sepalo'], alpha = 0.7)  # diagrama de puntos de dos var
ax.set_xlabel('lar.petalo')
ax.set_ylabel('lar.sepalo')
ax.autoscale()  # garantizar las escalas que la var tiene originalmente

# covarianza
np.cov(df['lar.petalo'], df['lar.sepalo'])  # matriz de covarianza; relacion puntual entre las var son las esquina inferior izq y sup derecha

# Correlacion entre 2 var
# corr = df.corr(method = 'spearman')  # si el valor es cercano a 1 es una asocioacion fuerte, es una correlacion lineal
corr = df.corr(method = 'kendall')      # no mide correlacion lineal necesariamente. varia en su magnitud

sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns ) # escala de colores sobre cada magnitud de kendall
# correlacion con mag fuerte negativa, es inversamente proporcional
# En la mitad esta la correlacion perfecta por que se habla de la var con la mism var

#           Truco 1
# plt.subplots(figsize = (30,20))
# mask = np.zeros_like(df.corr(), dtype = np.bool)
# mask[np.triu_indices_from(mask)] = True
# sns.heatmap(df.corr(), cmap = sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center=0)

#           Truco 2
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, vmin=-1, vmax=1,
cmap='Spectral', annot=True)

#           Truco 2         agregar z
fig, axis = plt.subplots()
ax.set_title('Variable original')
z = (y-np.mean(y))/np.std(y)
axis.hist(z, bins=30)
axis.axvline(np.mean(z), c='k', linestyle='--', label='media')
axis.axvline(np.mean(z)+ np.std(z), c='r', linestyle='--', label='std')
plt.legend()
plt.show()


#%%                 Estimadores a travez de datos
import sklearn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy import stats
import seaborn as sns


from scipy.stats import norm  # distribucion normal
# Simular muestra de tamanio 1 de valor 3
x1 = 3
# Hipotesis
mu1 = 4  # Media poblacional
mu2 = 7
sigma = 1  # desv estandar

p_muestra = norm.pdf(x1, mu1, sigma)  # Probability density function
p_muestra_2 = norm.pdf(x1, mu2, sigma)
p_muestra, p_muestra_2  # sobre una media pob de 4 tener un valor de 3 es de 24% probable y 


# Muestra mas grande con muestreo de dos valores
# Prob conjunta
x1 = 3
x2 = 10

mu1 = 4
mu2 = 7
sigma = 1

p_muestra = norm.pdf(x1, mu1, sigma)*norm.pdf(x2, mu1, sigma)  # multiplicacion de ocurrencia de cada uno de los eventos
p_muestra_2 = norm.pdf(x1, mu1, sigma)*norm.pdf(x2, mu2, sigma)
p_muestra, p_muestra_2  # muestrar mu1 es poco mas problable que mu2 pero no puede hacer inferencia clara


# muestra mas grande
muestra_10 = norm.rvs( 5  , sigma, size = 10)  # 5 = fil
data1 = norm.rvs( mu1  , sigma, size = 100000)
data2 = norm.rvs( mu2  , sigma, size = 100000)

ax = sns.distplot(data1, bins = 50, color = 'blue', kde = False)     # kde = False muestra como frecuencia y no como probab
ax.set(xlabel = 'Distribucion normal mu1', ylabel = 'Frecuencia')
ax = sns.distplot(data2, bins = 50, color = 'red',  kde = False)
ax.set(xlabel = 'Distribucion normal mu2', ylabel = 'Frecuencia')



muestra_10
y = list([])  # list vacia para poblar
for i in range(10):
    y.append(3000)  # 
      
ax.scatter(muestra_10, y, c ='k')   # visualizacion de la prob de ocurrencia basado en mu1 y mu2 con diferencia grande
# la muestra puede pertenecer con mayor o menor prob a alguna de las dos Hipotesis que planteamsos como parametro poblacion mu1

#%%         DISTRIBUCIONES MUESTRALES

import matplotlib.pyplot as plt
from IPython.core.display import Image
import seaborn as sns

# from scipy.stats import t      
# data1 = t.rvs( 100,  size = 1000000 ) # 100 = grados de libertad # muestra de datos, size = muy grande para aproximar el valor continuo
# data2 = t.rvs( 4 ,size = 1000000 )
# ax = sns.distplot( data2, bins = 500,   kde = False, color = 'blue')  # kde false para frecuencias y no probabilid
# ax = sns.distplot( data1, bins = 500,   kde = False, color = 'red')

# from scipy.stats import chi2     # permite dentificar la forma de calculo de probab para varianzas
# data1 = chi2.rvs( 5,  size = 1000000 ) 
# data2 = chi2.rvs( 15 ,size = 1000000 )
# ax = sns.distplot( data2, bins = 500,   kde = False, color = 'blue')  # kde false para frecuencias y no probabilid
# ax = sns.distplot( data1, bins = 500,   kde = False, color = 'red')

from scipy.stats import f     # permite aproximar cocientes de varianzas
data1 = f.rvs( 5, 25, size = 1000000 ) # (grados de libertad de las var X y Y, size)
data2 = f.rvs( 15, 25 ,size = 1000000 )
ax = sns.distplot( data2, bins = 500,   kde = False, color = 'blue')  # kde false para frecuencias y no probabilid
ax = sns.distplot( data1, bins = 500,   kde = False, color = 'red')

# Calculo de prob con Distribucion f
f.pdf(4, 15, 25)        # pdf probability density function (probabilidad asociada de un 4 en la distribucion, grados de libertad X y Y, )
                        # la altura de la probabilidad de 4 es 0.0019
f.cdf(4, 15, 25)        # Cumulative density function, la prb acum es casi 99%, donde al tener un val 4 estamos cubriendo 99% de obtener un valor en esta funcion

f.ppf(0.99889, 15,25)       # sabel el valor que acumula la prob de 99%

f.ppf(0.5, 15,25)           # valor que acum el 50% es un valor cercano a 1, algo asi como la mediana


#%%             TEOREMA DEL LIMITE CENTRAL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline  

from scipy.stats import expon
from scipy.stats import uniform

poblacion = pd.DataFrame()      # dataframe vacio
poblacion['numbers'] = expon.rvs(40, size = 100000 ) # rvs(lamba o media(promedio), size pob;lacional)

poblacion['numbers'].hist( bins = 100) # grafica con pandas

# Tomaremos una muestra de la poblacion que hemos definido
muestra_promedio = []
tamano = 50
for i in range(0, tamano):
    muestra_promedio.append( poblacion.sample(n=100).mean().values[0])  # n=val de muestra, sobre la n se calcula la media, values = otorga el valor de la posicion 0

fig, ax = plt.subplots()
ax.hist(muestra_promedio, bins = 50, alpha = 0.5)
ax.set_xlabel('Valor promedio')
ax.set_ylabel('Frecuencia')
ax.grid()


#%%                 PRUEBA DE HIPOTESIS

import pandas as pd
import numpy as np
 
import seaborn as sns
from scipy.stats import expon
from scipy.stats import uniform

muestra = [42, 35, 29, 45, 41, 57, 54, 47, 48, 56, 47, 35, 52, 
31, 52, 55, 57, 58, 26, 29, 32, 37, 32, 34, 48, 20, 48, 51, 27, 
24, 39, 40, 31, 34, 23, 24, 41, 58, 44, 48, 31, 23, 27, 55, 43, 
47, 30, 57, 38, 51]

# Hipotesis
media, var, skew, kurt = expon.stats(scale = 30, moments = 'mvsk' )   # scale = val de hipotesis, moment = hacer los calculos poblacion de una distrib exponencial

# Paso 1: parametro lambda
# Paso 2: HP

mu = 30
mu > 30

# Paso 3. Mejor estimador
# Estimador
# PAso 4. Distribucion
promedio = np.mean(muestra)
promedio

# Paso 5 - z= estimador y valor asociado estadistico
z = (promedio - mu)/np.sqrt(var/50)   # sqr debido a que la var proviene de exponencial;  es el tamano de la muestra
z

# Paso 6; tolerancia la error

alpha = 0.05

# criterios de rechazo
# Para HP de >= se va a reviasar la cola derecha de la distribucion
from scipy.stats import norm
data_norm = norm.rvs( size =  1000000)

ax = sns.distplot(data_norm, bins = 500, kde = False, color = 'blue')
ax.set_title('Distribucion normal')

# calculo de los val criticos, punto en el que la diostribucion acumule una prob de cola derecha que este asociado al val Alpha
valor_critico = norm.ppf(1-alpha, loc = 0, scale = 1),  # ppf = percentil probabiliti function, loc = media, scale = varianza; si alpha es la prob de error, entonces 1 - aplha es la de no error o val de tolerancia
valor_critico


ax = sns.distplot(data_norm, bins = 500, kde = False, color = 'blue')
ax.set_title('Distribucion normal')
ax.axvline(x = valor_critico, linestyle = '-.', label = 'valor critico')
ax.axvline(x = z, linestyle = '--', label = 'valor estadistico', color = 'k')
ax.legend()


#%%                     ERRORES ESTADÍSTICOS TIPO 1 Y 2

import pandas as pd
import numpy as np
import seaborn as sns

muestra = [42, 35, 29, 45, 41, 57, 54, 47, 48, 56, 47, 35, 52, 31, 52, 55, 57, 58, 26, 29, 32, 37, 32, 34, 48, 20, 48, 51, 27, 24, 39, 40, 31, 34, 23, 24, 41, 58, 44, 48, 31, 23, 27, 55, 43, 47, 30, 57, 38, 51]

# se acepto que mu>=30, anerior clase
mu1 = 37
mu2 = 42

promedio = np.mean(muestra)
promedio
# desv conocida
desv = 2

# estandarizacion para tener los val estadistic
z_1 = (promedio - mu1)/desv
z_2 = (promedio - mu2)/desv

# visualizar comport de val estadisticos bajo las HP asociadas mu1 y mu2
from scipy.stats import norm
data1 = norm.rvs( loc = mu1, scale = desv , size = 1000000 )   # kde se visualiza la frecuencia en forma de probabilidad
data2 = norm.rvs( loc = mu2, scale = desv , size = 1000000 )

ax = sns.distplot( data1, bins = 500 , kde = True, color = 'blue')
ax = sns.distplot( data2, bins = 500 , kde = True, color = 'red')
ax.axvline( x = promedio, c = 'k', linestyle = '--', label = 'promedio muestral')
ax.legend()     # Prom pareciera mucho mas probable baja la HP posee un alor de 42 y no 37

# Error tipo 1 o alpha: p rechazar ho cuando esta es verdadera
p_prom_mu1= norm.cdf(z_1)
1- p_prom_mu1

# Error 2: probaiblidad de no recharzar ho cuando esta es falsa
p_prom_mu2 = norm.cdf(z_2)
p_prom_mu2


#%%                     INTERVALOS DE CONFIANZA

import pandas as pd
import numpy as np

Muestra = [4046, 2578, 3796, 3412, 3315, 3228, 3666, 3877, 3154, 4062, 4365, 3776, 3761, 2587, 2911, 3184, 3810, 4459, 3385, 3899, 3602, 2701, 2821, 2790, 2557, 2540, 4119, 2712, 2743, 2713, 4466, 3937, 3871, 4427, 3177, 2556, 2903, 3522, 4155, 4010, 4453, 3080, 3008, 3865, 3356, 2799, 3308, 2759, 4347, 2576, 4075, 3333, 2936, 3746, 3334, 3940, 4113, 4220, 3580, 3922]

from scipy.stats import norm        # funciones de prob asociadas a una norm

alpha = 0.05    # nivel de tolerancia al error

lim_inf = norm.ppf(alpha/2) 
lim_sup = norm.ppf(1-(alpha/2)) 
lim_inf,lim_sup

promedio = np.mean(Muestra)
desviacion = np.std(Muestra)
len(Muestra)

# Ajustar los lim definidos, inverso de estandarizacion
lim_inf = lim_inf*desviacion + promedio
lim_sup = lim_sup*desviacion + promedio
lim_inf,lim_sup

#%%

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm
import pyreadstat as pr

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn import metrics

arc = 'compensacion-ejecutivos.sav'
df, meta = pr.read_sav(arc)
df.columns 

# parametros
y = df['salario']
X = df.drop(['salario', 'noasiat','postgrad'], axis = 1) # drop permite eliminar columnas no deseadas, exes = que eje se encientra en la column
    # X mayuscula para identificar varias variables
# modelo de regresion

from sklearn.linear_model import LinearRegression

reg_lin = sm.OLS( y , sm.add_constant(X)).fit()  # ols = ordinary least square para optimizar los parametros de la func de regreseion; ad constant para que agreguegue una constante y no solo calc la pendiente; fir = calcula parametros de ols los mejores parametros de a y b
print(reg_lin. summary())       # summary es funcion de sumarizacion para que muestre en detalle el contenido

# Fura de los errores
fig, ax = plt.subplots()
y_pred = reg_lin.predict(sm.add_constant(X))
ax.scatter(y, y - y_pred)
plt.axhline(y=0, color = 'black', alpha = 0.8, linestyle = '--')

# # Para obtener los coeficientes directamente
# model = LinearRegression()
# fit = model.fit(X,y)
# fit.coef_

# %%                                       REGRESION LOGISTICA

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm
import pyreadstat as pr

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score      # medicion de que tan acertado es mi modelo a la hora de predecir
from sklearn import metrics

arc = 'compensacion-ejecutivos.sav'
df, meta = pr.read_sav(arc)
df.columns
df.salario.describe() # analisis prifundo

# covertir la var y en var binaria
y =  np.where( df['salario'] > 96000, 1, 0)         # where si el salario es superior
X = df.drop( ['salario'], axis = 1)
y

# plot de la distribucion asociada y como var categorica en funcncion del salario
fig, ax = plt.subplots()
ax.scatter( df.salario, y)
ax.set_xlabel('salario')
ax.set_xlabel('y')

# regresion logistica
reg_log = linear_model.LogisticRegression()
reg_log.fit(X, y)
y_estimado_1 = reg_log.predict_proba(X)[:,1]  # proba de 1, ganar sueldo mayor a 96000 se toma solo la posicion 1
y_estimado = reg_log.predict_proba(X)       # arreglo prob de clase 0 en posic 0 y 1 en la posicion 1
y_estimado_sin_prob = reg_log.predict(X)    # tener el valor puntual 1 o 0 y no como prob
y_estimado_sin_prob == y        # comparacion con el y

metrics.accuracy_score(y,reg_log.predict(X))        # calcular el acuu de y y la prediccion asociada a Y utilizando a X
# nuestro modelo puede aproximar en un 90% de los casos utilizando las variables dentro de X

#%%                 # ARBOLES DE DESCICION
# clase 23 error GraphViz's executables not found


import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Importar el classificador
from sklearn.model_selection import train_test_split # Importar funciones de particion
from sklearn import metrics #Importar las metricas de scikit-learn
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO  
from io import StringIO     # remplazo de la anterior
from IPython.display import Image  
import pydotplus


df = pd.read_csv("rating-peliculas.csv")

df.columns

df.describe()       # visualizar la composicion de las var numericas

y = df.genero       # var objetivo
y.value_counts()        # identificar cada genero
len(y)

X = df.drop(['pelicula', 'genero'], axis = 1)       # categorizar o aproximarnos a la categ, elimina columns

# dividir la muestra para probar el desempenio, entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)    # random_state = para que el sampli aleatorio sea reproducible

# classifier
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)      # la var es categorica y se usa desitiontreeclass.. ;  criterian = entrega al arbol 
                                                                        # la func de desempeno que va a optimizar para encontrar los mejores parametros
                                                                        # max.. = asociada a las ramas o los niveles de reglas que se usa
clf = clf.fit(X_train, y_train)         # calcula la funcion de regresion utilizando (....)
y_pred = clf.predict(X_test)        # calculo de valores predichos

y_pred

# visualizacion del arbol generado
dot_data = StringIO()       # permite las reglkas del arbol de desition en string
export_graphviz(clf, out_file = dot_data, filled = True, rounded = True,        #faildes = coloreado
               special_characters = True, feature_names = X.columns, 
                class_names = y.value_counts().index)       # y.value_counts().index = para traer solo los labels

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())      # def grafic, getvalue() = se pide losalinterior
graph.write_png('peliculas.png')        # guardar la imagen que se genera
Image(graph.create_png())  # Imprime

metrics.accuracy_score(y_test, y_pred)      # nivel de predict de nuestras arbol, que tanto pude esplicar el genero a travez de las var que hemos definido




# %%
