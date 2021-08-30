#%%

import matplotlib
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image       # para leer las imagenes

plt.style.use('classic')        # usar estilo clasico

imagen = Image.open("imagen_ejemplo_frida_bredesen.jpg")

plt.imshow(imagen)
plt.show()

imagen_gr = imagen.convert('LA')        
    # convert('LA')
    # Convierte la imagen RGB a escala de grises, pero entrega una imagen con 2 canales(o bandas) L y A:
    # L tiene la información de la imagen en escala de grises.
    # A tiene la información de transparencia de la imagen, como estamos trabajando con una imagen JPG todos los valores en esta matriz son 255.

print(imagen_gr)

# Transformacion a matriz
imagen_mat = np.array(list(imagen_gr.getdata(band=0)), float)        # pasar una lista de la imagen en escala de grises que toma todos los datos de la banda 0, guarda como float
print(imagen_mat)       # se presenta solo como vector

# imagen_mat.shape = (imagen_gr.size[1], imagen_gr.size[0])       # asiganar la forma o tamanio de las dimensiones
# print(imagen_mat)
# print(imagen_mat.shape)         # muestra los pixeles, guardados en fil y col

# # graficar la matriz
# plt.imshow(imagen_mat, cmap='gray')     # se hace la conversion a escala de grises
# plt.show()

# imagen_mat_2 = imagen_mat / 10
# print(imagen_mat_2)
# plt.imshow(imagen_mat_2)     # se hace la conversion a otra escala
# plt.show()

# print(np.max(imagen_mat))
# print(np.max(imagen_mat_2))

# print(np.min(imagen_mat))
# print(np.min(imagen_mat_2))
# """
# - No es tan importante tener los valores obtenidos, sino la relacion entre ellos
# - En ML se debe optimizar y para que el algoritmo no tenga problemas de convergencia se denbe
# tener los valores entre 0 y 1.
# - Dicho Efecto se LOgra dividiendo al dataset a su maximo valor
# """

# # %%            DESCOMPOSICION SVD A UNA MATRIZ

# import numpy as np
# import matplotlib.pyplot as plt

# from PIL import Image

# plt.style.use('classic')

# imagen = Image.open("imagen_ejemplo_frida_bredesen.jpg")

# plt.imshow(imagen)
# plt.show()

# imagen_gr = imagen.convert('LA')        # conversion a escala de grises
# imagen_mat = np.array(list(imagen_gr.getdata(band=0)), float)       # conversion de imagen a una matriz
# imagen_mat.shape = (imagen_gr.size[1], imagen_gr.size[0])       # otorgar la dimension correcta

# U, D, V = np.linalg.svd(imagen_mat)
# print(imagen_mat.shape)         # (3456, 3693)
# print(U.shape)      # (3456, 3456) solo tiene la primer dimensiones cuadrada
# print(D.shape)      # (3456,) valores singulares
# print(V.shape)      # (3693, 3693)   # ratacion final, forma que requiere la imagen

# # recosntruccion de la imagen por medio de los valores U y V conseguidos
# imagen_recons = np.matrix(U[:,:1])*np.diag(D[:1])*np.matrix(V[:1,:])
#     # primer valor posee mas informacion, mayor varianza de los datos
#     # np.matrix(U[:,:1]) = primera de las filas
#     # *np.diag(D[:1]) = solo usa el primero de los valores
#     # np.matrix(V[:1,:] = primer fila y todas las columnas
# plt.imshow(imagen_recons, cmap='gray')
# plt.show()

# #                   CANTIDAD DE VALORES SINGULARES QUE NOS SIRVEN

# i = 50      # usar los i primeros valores singulares
# imagen_recons = np.matrix(U[:,:i])*np.diag(D[:i])*np.matrix(V[:i,:])
# plt.imshow(imagen_recons, cmap='gray')
# titulo = "Valores Singulares = %s" %i
# # titulo = f"valores singulares = {i}"
# # plt.title(“valores singulares = %s” % i)

# plt.title(titulo)
# plt.show() 

# """
# La imagen tenia mas de 3600 valores singulares pero solo basta con 
# 50 para reconocer que la imagen sin ninguna duda la imagen original
# """



# # %%
