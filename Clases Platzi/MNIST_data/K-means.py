#%%

# Importar librerias para graficas
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()     # graficos de apoyo

import numpy as np  # para trabajar con estructuras de datos

from sklearn.datasets.samples_generator import  make_blobs  # Generador grupo de datos o sample generatos

from sklearn.cluster import KMeans

#graficado
X,Y = make_blobs(n_samples = 300, centers=4) # de 300 puntos tener 4 grupos
plt.scatter(X[:,0], X[:,1], s=50)

# K-Means 
kmeans = KMeans(n_clusters = 4)

# entrenamiento del algoritmo
kmeans.fit(X)      # no se le da las etquetas sino directamente los puntos para que encuentre los agrupamientos

#clusters lebels (etiquetas)
y_means =kmeans.predict(X)      #ayua a encontrar los diferentes centros 

#centroides values
centers = kmeans.cluster_centers_  # para cada uno de los clusters
print (centers)

# Grafica de valores conjuntos
plt.scatter(X[:,0], X[:,1], c = y_means, cmap='viridis') # C =  considerando valoresde prediccion, Cmap =  colores
# incluir centroides
plt.scatter(centers[:,0], centers[:,1], c = 'black', s = 200, alpha=0.5)

plt.show()

# %%
