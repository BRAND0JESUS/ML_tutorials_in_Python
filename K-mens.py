"""
Iris, k-,means
"""

from abc import abstractmethod
from operator import mod
from pandas.io.formats.format import format_percentiles
from sklearn.cluster import KMeans       # libreria.particularidad; cluster = agrupamiento
from sklearn import datasets 
import pandas as pd

import matplotlib.pyplot as plt

iris = datasets.load_iris()     # toda la informacion, llamado al dataset

# Division de los datos
X_iris =iris.data       
Y_iris = iris.target

x = pd.DataFrame(iris.data, columns= ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])    # division de la informacion
# x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=['Target'])       # info del target que no se usa

print(x.head(5))

# Plot
plt.scatter(x['Petal Length'], x['Petal Width'], c = 'blue')
plt.xlabel('Petal Length', fontsize = 10)
plt.ylabel('Petal Width', fontsize = 10)
plt.show()


model = KMeans(n_clusters=3, max_iter= 1000)         # n_clusters = similitud o valor de k ,permite generar centroides, max..= como se mueve k para encontrar distancia mas cercana entre los puntos

model.fit(x)     # entrenamiento
y_labels = model.labels_        # todas la etiquetas encontradas basadas en la similitud

# pediccion de y
y_kmeans = model.predict(x)     # prediccion de y a partir de x
print('Predicciones', y_kmeans)

from sklearn import metrics

accurracy = metrics.adjusted_mutual_info_score(Y_iris, y_kmeans)
print(accurracy)        # 0.5 no es un buen modelo, se modifica k

plt.scatter(x['Petal Length'], x['Petal Width'], c = y_kmeans, s = 30)      # creacion de atributos visuales; c = kmeans -> da el valor correspondiente, s = tamanio de los puntos
plt.xlabel('Petal Length', fontsize = 10)
plt.ylabel('Petal Width', fontsize = 10)
plt.show()

#%%
import numpy as np
print(np.arange(5))
print(np.zeros(5))
# %%
