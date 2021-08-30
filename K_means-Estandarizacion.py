"""
Para aquellos que obtuvieron una precisión del 36-37% (al menos para este caso en particular) 
es debido a que trabajaron con los datos directamente sin estandarizarlos previamente 
(estoy seguro que el tema de estandarización será tratado en clases posteriores, así que no teman). 
Haciendo uso de la estandarización pude obtener una precisión de 0.89. A continuación, dejo mi código 
para que puedan observar la estandarización de datos.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler #Necesario para la estandarizacion de los datos

# Descarga de los datos de Vinos
vinos = datasets.load_wine()
variables = np.array(vinos.feature_names)
x_vinos = vinos.data
y_vinos = vinos.target

# Normalizacion de los valores de “X”
scaler = StandardScaler()
scaler.fit(vinos.data)
x_scaled = scaler.transform(vinos.data)
x = pd.DataFrame(x_scaled, columns=variables, ) #Si deseas trabajar con los datos sin estandarizar, solo cambia x_scaled por x_vinos
y = pd.DataFrame(y_vinos, columns= ["Target"])
x.head(5)

# Elaboracion del Modelo y Metodo del Codo
wccs =[]
n = 1
acc = 0
for i in range(1, 11):
    codo = KMeans(n_clusters = i, max_iter = 1000, random_state = 0)
    codo.fit(x)
    y_kmeans = codo.predict(x)
    wccs.append(codo.inertia_)
    accuracy =  round(metrics.adjusted_rand_score(y_vinos, y_kmeans), 4)
    print(f'Cantidad de Centroides: {i} ---  Precision: {accuracy}')
    if accuracy > acc:
        acc = accuracy
        n = i
plt.plot(range(1, 11), wccs)
plt.title('Metodo del Codo')
plt.xlabel('Numero de centroides')
plt.ylabel('WCSS')
plt.show()
print(f"Se recomienda emplear una cantidad de {n} centroides, para asi garantizar una precision de {acc}")

# Ejecucion del modelo KMeans con la cantidad de centroides seleccionados
modelo = KMeans(n_clusters = n, max_iter = 1000)
modelo.fit(x)
y_labels = modelo.labels_
y_kmeans = modelo.predict(x)
print('predicciones ', y_kmeans)
y_kmeans_df = pd.DataFrame(y_kmeans, columns = ['Prediction'])

# Precision
accuracy =  metrics.adjusted_rand_score(y_vinos, y_kmeans)
print(round(accuracy, 5))

# Vizualizacion del modelo

# Concateno el dataset de entrada con el de la prediccion
Z = pd.concat([x, y_kmeans_df], axis = 1)
# Grafico
sns.pairplot(Z, hue = 'Prediction')
