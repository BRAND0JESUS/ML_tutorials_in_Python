#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi,100)
y = np.sin(x)
plt.plot(x,y,'m--')
plt.grid()
w = np.linspace(0, 2*np.pi,100)
v = np.sin(-w)
plt.plot(w,v,'g.')


# %%

#Librerías necesarias: Matplotlib y Numpy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

#linspace retorna valores entre los intervalos que se indican
u=np.linspace(-5,5,100)
v=np.linspace(-6,6,100)

#Funciona similar que en Matlab
[x,y]=np.meshgrid(u,v)

#Esta es la ecuación
z=np.cos(x)*np.cos(y)*np.exp(-np.sqrt(x**2+y**2)/20)

#Se crea la figura
fig = plt.figure()

#Se le indica el tamaño
fig.set_size_inches(10, 6)

#Especifica que es 3D
ax = plt.axes(projection="3d")

#Generación de la superficie
# p=ax.plot_surface(x, y, z,cmap=cm.coolwarm);
p=ax.plot_surface(x, y, z,cmap='inferno')

#Barra que indica el valor de los colores
fig.colorbar(p,ax=ax)
# %%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm



theta =  np.linspace(0,8*np.pi,100)  # angulo, de -pi a pi, 100 datos

#r = 2-4*np.cos(theta)
r = np.sin(8/5*np.cos(theta))
fig = plt.figure()
ax = fig.add_subplot(111, projection = 'polar')
ax.plot(theta,r,'r')
plt.show()
# %%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Definimos la función.
def f(x, y):
    return x**2 + y**2

# Creamos el rango de valores para X y Y.
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)

# Calculamos las coordenadas de los puntos (X, Y).
X, Y = np.meshgrid(x, y)
Z = f(X, Y) # Llamamos la función.

fig = plt.figure()  # Se crea la figura.
fig.set_size_inches(10, 6)  # Se le indica el tamaño.
ax = plt.axes(projection="3d")  # Especifica que es 3D.
p=ax.plot_surface(X, Y, Z, cmap=cm.coolwarm);  # Generación de la superficie.
fig.colorbar(p,ax=ax) # Barra que indica el valor de los colores.
# %%
