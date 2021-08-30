
#%%
"""
La maldición de la dimensión: Esto dice que por cada variable que agrego en el conjunto de datos vamos a necesitar exponencialmente más muestras para poder tener la misma relevancia estadística.
Cuál es el autovector relacionado con el autovalor más grande? Es el autovalor quien define cual es la dirección que contiene más información.
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from graficarVectores import graficarVectores


np.random.seed(42)      # para poder reproducir el experimento se define la semilla
x = 3*random.rand(200)      # generacion de 200 num randomicos
y = 20*x + 2*random.rand(200)

x = x.reshape(200,1)        # agragar la forma
y = y.reshape(200,1)
print(x)

xy = np.hstack([x, y])        # hacer pares unidos para poder graficar en el eje cartesian
print(xy.shape)

# Grafico de la dispersion de puntos
plt.plot(xy[:,0], xy[:,1], '.')     # xy[:.0] = todas las filas y col 0, xy[:,1] = todas las filas y col 1
plt.show()

# Centrado de la informacion
xy_centrado = xy-np.mean(xy, axis=0)            # sirve para simpliocar el computo de los numeros, axis=0 = sobre cada eje
plt.plot(xy_centrado[:,0], xy_centrado[:,1], '.')     # xy[:.0] = todas las filas y col 0, xy[:,1] = todas las filas y col 1
plt.show()

autovalores, autovectores = np.linalg.eig(xy_centrado.T.dot(xy_centrado))
print(autovectores)         # los numeros de los autovectores son los que maximizan la funcion, el vector asociado con el autovalor mas grande seniala la direccion de max varianza

graficarVectores(autovectores.T, ['blue', 'red'])
plt.plot(xy_centrado[:,0], xy_centrado[:,1]/20, '.')       # graficar directamente la nube de puntos en el mismo plot, se /20 para visualizar de mejor manera la direccion en la cual se mueve
plt.show()

print(autovalores)

# proyeccion de los puntos con el nuevo sistema de referecia
xy_nuevo = autovectores.T.dot(xy_centrado.T)
plt.plot(xy_nuevo[0,:], xy_nuevo[1,:], '.')         # [0,:] 0 en fil y tomar todas las col, [1,:] 1 fil y tomar todas las col
plt.show()
print(xy)
print(xy_nuevo)



# %%
