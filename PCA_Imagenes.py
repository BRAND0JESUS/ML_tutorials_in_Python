#%%
from inspect import CO_VARARGS
import matplotlib.pyplot as plt
import numpy as np
import imageio
from numpy.lib.function_base import append
from numpy.lib.shape_base import hsplit
from numpy.ma.extras import clump_masked      # para graficar
import pandas as pd



# conjunto de datos con muchas caras
im = imageio.imread('imagenes/s3/3.pgm')        # cargar archivos de carpeta gurada en la misma direccion que el archivo .py

# import sklearn.datasets           # opccion de cargar archivos directamente de dataset de sklearn
# im1= sklearn.datasets.fetch_olivetti_faces()
# im2 = imageio.imread(r'D:\VS Code\imagenes\s7\7.pgm')       # cargar archivos en una direccion distinta ar archivo .py
 
im = im.astype(np.uint8)            # definicion de que tipo de numero posee
print(im)

# observar el max y min para normalizar
max = np.amax (im)
print(f'Max original: {max}')
min = np.amin (im)
print(f'Max original: {min}')
# print('Maximo original', end=' ')           # end=' ' se agraga un espacio al final

# Normalizado
im_original = im
im = im/255            # max = 204

max = np.amax (im)
print(f'Max original: {max}')
min = np.amin (im)
print(f'Max original: {min}')

# Visualizacion del la imagen
fix, ax = plt.subplots(1, 2, figsize = (12,12),         # subplots = dos graficos; 1, 2 = 1 fil, 2 col; figsize = tamanio de c/u de las figuras
                        subplot_kw = {'xticks': [], 'yticks': []})      # xticks': [] = sacarles los ejes

ax[0].imshow(im_original, cmap = 'gray')
ax[1].imshow(im, cmap = 'gray')
plt.show()

# Leer todas las imagenes del directorio
from glob import iglob
caras = pd.DataFrame([])    # dataFrame donde se va a guardar los datos de las imagenes de c/u
for path in iglob('.\\imagenes\\*\*.pgm'):       # visualizar el path del conjunto de imagnes
    im = imageio.imread(path)       # caraga de la imagen
    cara = pd.Series(im.flatten(), name = path)        # componer cara en una serie donde en lugar de pasar una matriz, se pasa un vectores; se aplana la imagen
    caras = caras.append(cara)

# Definicion de como queremos que se muestre el grafico
fig, axes = plt.subplots(5, 10, figsize = (15, 8),         # subplots = dos graficos; 5, 10 = 5 fil o sujetos, 10 col o 10 fotos
                            subplot_kw = {'xticks': [], 'yticks': []},      # sin ejes
                            gridspec_kw = dict(hspace = 0.01, wspace = 0.01))       # espacio entre cada una de ellas de 0.01

# Recorrer el conjunto
for i, ax in enumerate(axes.flat):      # enumeracion de los ejes
    ax.imshow(caras.iloc[i].values.reshape(112,92), cmap = 'gray')      # grafuicar cada imagen en caras.iloc[i]; reshape(112.92) = darle forma a la imagen
plt.show()

# Aplicación de PCA
from sklearn.decomposition import PCA

caras_pca = PCA(n_components = 0.999)         # en numero de componentes sea el necesario para quedarse con el 50% de la variacion de los datos
caras_pca.fit(caras)

filas = 3       # para que se muestre 3 filas
columnas = caras_pca.n_components_//filas       # lo que caras pca tenga en num de componentes // fillas; division entera 

# Definicion de como queremos que se muestre el grafico
fig, axes = plt.subplots(filas, columnas, figsize = (12, 6),         
                            subplot_kw = {'xticks': [], 'yticks': []},      # sin ejes
                            gridspec_kw = dict(hspace = 0.01, wspace = 0.01))       # espacio entre cada una de ellas de 0.01

# Recorrer el conjunto
for i, ax in enumerate(axes.flat):      # para cada uno de los ejes que se muestre caras pca
    ax.imshow(caras_pca.components_[i].reshape(112,92), cmap = 'gray')      # grafuicar cada imagen en caras.iloc[i]; reshape(112.92) = darle forma a la imagen
plt.show()      

# Si el resultado son 6 imagenes y n_components = 0.5, nos dice que para ver el 505 de la info nos alcanza con quedarnos 6 imagnes
print(caras_pca.n_components_)

componentes = caras_pca.transform(caras)
proyeccion = caras_pca.inverse_transform(componentes)

fig, axes = plt.subplots(5, 10, figsize=(15, 8),
                       subplot_kw = {'xticks' : [], 'yticks':[]},
                        gridspec_kw = dict(hspace = 0.01, wspace = 0.01))

for i, ax in enumerate(axes.flat):
    ax.imshow(proyeccion[i].reshape(112,92), cmap = "gray")
plt.show()
# %%
