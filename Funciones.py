#%%                                         FUNCIONES LINEAL

import matplotlib.pyplot as plt   # librería para graficar
import numpy as np                # librería para manejo de vectores y utilidades matemáticas

def funcion(m, x, b):
    return (m*x+b)

res = 100 

m = 10

b = 5

x =np.linspace(-10.0,10.0, num=res)     # linspace genera una serie de puntos dentro de un rango

y = funcion(m, x, b)

fig, ax = plt.subplots()
ax.plot(x,y)
ax.grid()
ax.axhline(y=0, color = 'r')
ax.axvline(x=0, color = 'r')
# plt.show()




#%%                                         FUNCIONES POLINOMICA

import matplotlib.pyplot as plt   # librería para graficar
import numpy as np                # librería para manejo de vectores y utilidades matemáticas

def funcion(x):
    return 2*x**7 - x**4 + 3*x**2 + 4

res = 100 



x =np.linspace(-10.0,10.0, num=res)     # linspace genera una serie de puntos dentro de un rango

y = funcion(x)

fig, ax = plt.subplots()
ax.plot(x,y)
ax.grid()
plt.xlim(-25,25)
plt.ylim(-25,25)
ax.axhline(y=0, color = 'r')
ax.axvline(x=0, color = 'r')
# plt.show()


#%%                                         FUNCIONES TRACIENTES

import matplotlib.pyplot as plt   # librería para graficar
import numpy as np                # librería para manejo de vectores y utilidades matemáticas

def funcion(x):
    return np.cos(x)

res = 100


x =np.linspace(-10.0,10.0, num=res)     # linspace genera una serie de puntos dentro de un rango

y = funcion(x)

fig, ax = plt.subplots()
ax.plot(x,y)
ax.grid()
# plt.xlim(-25,25)
# plt.ylim(-25,25)
ax.axhline(y=0, color = 'r')
ax.axvline(x=0, color = 'r')
# plt.show()


#%%                                         FUNCIONES EXPONENCIAL
import matplotlib.pyplot as plt   # librería para graficar
import numpy as np                # librería para manejo de vectores y utilidades matemáticas

def funcion(x):
    return np.e**x

res = 100 



x =np.linspace(-1.0,1.0, num=res)     # linspace genera una serie de puntos dentro de un rango

y = funcion(x)

fig, ax = plt.subplots()
ax.plot(x,y)
ax.grid()
ax.axhline(y=0, color = 'r')
ax.axvline(x=0, color = 'r')
# plt.show()


#%%                                         FUNCIONES LOGARITMO

import matplotlib.pyplot as plt   # librería para graficar
import numpy as np                # librería para manejo de vectores y utilidades matemáticas

def funcion(x):
    return np.log2(x)

res = 100 



x =np.linspace(0.01,256, num=res)     # 256 = corresponde al max valor de X

y = funcion(x)

fig, ax = plt.subplots()
ax.plot(x,y)
ax.grid()
ax.axhline(y=0, color = 'r')
ax.axvline(x=0, color = 'r')
# plt.show()

#%%                                         FUNCIONES SECCIONADA
# FUNCION DE HEAVISIDE

import matplotlib.pyplot as plt   # librería para graficar
import numpy as np                # librería para manejo de vectores y utilidades matemáticas

def H(x):
    y = np.zeros(len(x))
    for idx, x in enumerate (x):     # idx = indice, x = otroga el valor de X
        if x >= 0:
            y[idx] = 1.0

    return y


res = 100 

x =np.linspace(-10.0,10.0, num=res)     # linspace genera una serie de puntos dentro de un rango

y = H(x)

fig, ax = plt.subplots()
ax.plot(x,y)
ax.grid()
# ax.axhline(y=0, color = 'r')
# ax.axvline(x=0, color = 'r')
# plt.show()


# %%                                 MANIPULACION DE FUNCIONES

import matplotlib.pyplot as plt   # librería para graficar
import numpy as np                # librería para manejo de vectores y utilidades matemáticas

N = 1000

def f(x):
  return np.sin(x);

c = 10

x = np.linspace(-15,15, num=N)

y = f(x) 

# y=f(x)+c se desplaza c unidades hacia arriba.
# y=f(x)−c se desplaza c unidades hacia abajo.
# y=f(x−c) se desplaza c unidades hacia la derecha.
# y=f(x+c) se desplaza c unidades hacia la izquierda.
# y=c⋅f(x) alarga la gráfica verticalmente en un factor de c .
# y=1/c⋅f(x) comprime la gráfica verticalmente en un factor de c .
# y=f(c⋅x) comprime la gráfica horizontelmente en un factor de c .
# y=f(1/c⋅x) alarga la gráfica horizontelmente en un factor de c .
# y=−f(x) refleja la gráfica respecto al eje x.
# y=f(−x) refleja la gráfica respecto al eje y.

fig, ax = plt.subplots()
ax.plot(x,y)
ax.grid()
ax.axhline(y=0, color='r')
ax.axvline(x=0, color='r')

# %%                                COMPOSICION DE FUNCIONES
# FUNCION DENTRO DE OTRAS FUNCIONES

import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return x**3

def f(x):
    return np.sin(x)

x = np.linspace (-10, 10, num=1000)

f_o_g = f(g(x))

plt.plot(x,f_o_g)
plt.grid()



# %%                    CURVAS DE NIVEL

from matplotlib import cm # Para manejar colores
import numpy as np
import matplotlib.pyplot as plt

# grafica 3 Dimensiones
def f(x,y):
    return np.sin(x) + 2*np.cos(y)

res = 100

x = np.linspace(-4, 4, num=res)
y = np.linspace(-4, 4, num=res)

x, y = np.meshgrid(x, y)    # Combinacion de puntos x, y

z = f(x,y)

fig, ax = plt.subplots(subplot_kw = {"projection" : "3d"})     # Grafica de 3D
surf = ax.plot_surface(x,y,z, cmap = cm.cool)       # todo lo que queremos que se grafique en la superficie
fig.colorbar(surf)

# Curvas de nivel

fig2, ax2 = plt.subplots()
level_map = np.linspace(np.min(z), np.max(z), num = res)   # se crea un vector      
cp = ax2, plt.contour(x, y, z, levels = level_map, cmap = cm.cool)
# cp = ax2, plt.contourf(x, y, z, levels = level_map, cmap = cm.cool)       # sin las curvas de contorno
plt.show()
# %%
