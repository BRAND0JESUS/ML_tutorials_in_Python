#%%

import seaborn as sns
vuelos = sns.load_dataset("flights")
vuelos = vuelos.pivot("month", "year", "passengers")
ax = sns.heatmap(vuelos)
# %%
import numpy as np  # numerical python
from platform import python_version
print(python_version())

escalar = 5.678
print(escalar)

escalar_python = True           # buleano
print(escalar_python)
print(type(escalar))
# Se diferencia por slo grados de libertad 
# vector 1 solo grado, 1 solo eje
# matriz, 2 ejes, x, y
# tensor, 3 grados de libertad, x,y,z es decir puede variar entre matrices

vector = np.array([1,2,3])
matriz = np.array([[1,2,3],[4,5,6],[7,8,9]])
tensor = np.array([
    [[1,2,3],[4,5,6],[7,8,9]],
    [[1,5,3],[4,8,6],[7,8,9]],
    [[1,100,1],[4,255,4],[7,7,7]]
])
print(tensor)


import matplotlib.pyplot as plt

plt.imshow(tensor, interpolation = 'nearest')
plt.show()

print(len(vector))
print(vector.shape)

print(len(matriz))
print(matriz.size)  #cantidad de elementos 3x3 = 9


print(tensor.shape)
print(tensor.size)
# %%
import numpy as np
tensor = np.array([
    [[1,2,3],[4,5,6],[6,7,8],[6,7,8]],
    [[11,12,13],[14,15,16],[17,18,19],[6,7,8]],
    [[21,22,23],[24,25,26],[27,28,29],[6,7,8]],
])
print(tensor)
print(tensor.shape)
print(tensor.size)

tensor_trans = tensor.T
print(tensor_trans)
print(tensor.shape)


# %%
import numpy as np

vector = np.array([1,2,3])
matriz = np.array([[1,2,3],[4,5,6],[7,8,9]])

A = matriz * vector
print(A)
B = matriz.dot(vector)
print(B)
C = np.dot(matriz, vector)
print(C)
# %%

vector = np.array([1,2,3])
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = A.T
# Productor interno
prod_interno = np.dot(A, vector)
print (prod_interno)
# Productor externo
prod_externo = np.outer(A, vector)
print (prod_interno)
# Producto punto
prod_punto = A @ vector
print (prod_punto)


# (A*B)TRANS = (Atrans*Btrans)
# ((A)trans)trans = A
# %%
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-5,5,1)
y_1 = 3*x+5
y_2 = 2*x+3

plt.figure()
plt.plot(x,y_1)
plt.plot(x,y_2)
plt.xlim(-5,5)
plt.ylim(-5,5)

A = np.array([[-3,1],[-2,1]])
b = np.array([5,3])

sol_1 = np.array([-2,-1])
print (sol_1)
print (A.dot(sol_1))


# %%

matriz = np.array([[1,2,3],[4,5,6],[7,8,9]])
matriz_invers = np.linalg.inv(matriz)
print (matriz.dot(matriz_invers))

# Me otroga la misma matriz salvo que sea una matriz singualar
# %%
import numpy as np
np.set_printoptions(suppress=True)      # Si el numero es muy cercano a 0 me imprime 0

A = np.array([[3,1], [2,1]])
b = np.array([[1],[1]])
inver_A = np.linalg.inv(A)
print(A)
print (inver_A)

x= inver_A.dot(b)          # Respuesta de nuestras x, y
print (x)
print(A.dot(x))             # Verificacion que nuestra x es la repuesta


sol_2 = inver_A.dot(np.array([[3],[7]]))
print (sol_2)

print (A.dot(sol_2))
# %%
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-6,6)

y_1 = 3*x + 5
y_2 = -1*x + 3
y_3 = 2*x + 1

plt.figure()
plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_3)

plt.xlim (-8,8)
plt.ylim (-8,8)

plt.axvline(x=0, color = 'grey')
plt.axhline(y=0, color = 'grey')

plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt

v1 = np.array([2,5])
v2 = np.array([3,2])

# Llamda de funciones externas al archivo
%run "..\\VS Code\graficarVectores.py"

graficarVectores([v1,v2], ['orange', 'blue'])
# Definimos los limites
plt.xlim(-1, 8)
plt.ylim(-1, 8)
# %%
import numpy as np
import matplotlib.pyplot as plt

v1 = np.array([2,5])
v2 = np.array([3,2])

# vecs: vectores.
# cols: colores.
# alpha: valor de transparencia.
def graficarVectores(vecs, cols, alpha=1):
    plt.figure()
    plt.axvline(x=0, color="grey", zorder=0)           # zorder = orden de imagnes si se sobreponen
    plt.axhline(y=0, color="grey", zorder=0)
    
    for i in range(len(vecs)):
        # El origen de los vectores inicia en el punto (0,0)
        x = np.concatenate([[0,0], vecs[i]])
        plt.quiver([x[0]],
                  [x[1]],
                  [x[2]],
                  [x[3]],
                  angles='xy', scale_units='xy', scale=1, 
                  color=cols[i], alpha=alpha)

graficarVectores([v1,v2], ['orange', 'blue'])
# Definimos los limites
plt.xlim(-1, 8)
plt.ylim(-1, 8)

#%% 
# Si deseas plotting de varias funciones en una sola ventana 
# pero de forma separadas, te paso este sencillo ejemplo:

import matplotlib.pyplot as plt

x = np.arange(1,100,0.2)
y1 = np.sin(x)
y2 = x ** 2 + 2 * x
y3 = np.log(x)
y4 = x + 2

ax1 = plt.subplot(221) # (rows|columns|index)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

ax1.plot(x,y1)
ax2.plot(x,y2,'y--')
ax3.plot(x,y3,'r')
ax4.plot(x,y4,'g')

plt.tight_layout()
plt.show()
# %%
# graficar con pyplot de forma rápida varias funciones en un solo plot

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-100,101,1)      # Array x
y1 = 25 * x ** 2 - 2 * x       # y1 = 0.5x²+2x
y2 = ((0.5 * x ** 2) // 2 * x)  # y2 = ½x²/2x
y3 = np.sin(0.25 * x) * 250000 # y3= 250K sin(¼x)

#================================================
plt.plot(x,y1,'r.')
plt.plot(x,y2,'b--')
plt.plot(x,y3,'y')
plt.show()
# %%

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

v1 = np.array([1,0,0])
v2 = np.array([2,-3,0])

fig = plt.figure()
ax = plt.axes(projection='3d')

for a in range (-10,10):
  for b in range (-10,10):
    ax.scatter(v1[0]*a+ v2[0]*b,
               v1[1]*a+ v2[1]*b,
               v1[2]*a+ v2[2]*b,
               marker='.',
               color="orange")
# %%

import numpy as np

A = np.array(
[
    [0,1,0,0],
    [0,0,1,0],
    [0,1,1,0],
    [1,0,0,1]
])

lambdas, V = np.linalg.eig(A.T) # buscar autovaor y autovectores
print(A[lambdas == 0, :])  # me dara la fila que es linealmente dependiente = suma de el resto de filas
# si posee fila linealmente dependiente, no posee inversa
# %%

# Las normas generalmente se llaman por la letra 'L'
# L0: Cantidad de elementos de nuestro vector distintos de 0
# L1: La suma de los valores absolutos de los componentes de
# nuestro vector
# L2: 'La norma que conocemos', La magnitud de un vector, o distancia al origen.

# En machine learning se usa mucho la norma L2 pero elevada al cuadrado
# es decir, L2^2. La ventaja de esto es computacional.
# Podríamos reunir muchos vectores en una matriz X, calcular X.dot(x_t)
# y tendríamos allí el cuadrado de las normas de todos esos vectores.

#L_infinito: Nos devuelve el mayor valor dentro de los valores absolutos 
# de los componentes de nuestro vector```


vector = np.array([1,2,0,5,6,0])
print(np.linalg.norm(vector, ord=0))

# L1

vector1 = np.array([1,-1,1,-1,1])
print(np.linalg.norm(vector1, ord=1))

# L2

vector2 = np.array([1,1])
print(np.linalg.norm(vector2))
print(np.linalg.norm(vector2, ord=2))

vector3 = np.array([1,2,3,4,5,6])
print(vector3)
print(np.linalg.norm(vector3))
print(np.linalg.norm(vector3, ord=2))
print(np.linalg.norm(vector3, ord=2)**2)
print(vector3.T.dot(vector3))
# L infinito

vector4 = np.array([1,2,3,-100])
print(vector4)
print(np.linalg.norm(vector4, ord=np.inf))


#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

v1 = np.array([0,0,0,3])
v2 = np.array([0,0,3,3])

plt.xlim(-2,6)
plt.ylim(-2,6)

plt.quiver([v1[0],v2[0]],
          [v1[1],v2[1]],
          [v1[2],v2[2]],
          [v1[3],v2[3]],
           angles='xy', scale_units='xy', scale=1,
           color = sns.color_palette()
          )
plt.show()

# Para los arreglos.
v1 = np.array([0,3])
v2 = np.array([3,3])

# Realizamos su producto interno.
prod_in = v1.T.dot(v2)

# Calculamos las L2.
v1_norm = np.linalg.norm(v1)
v2_norm = np.linalg.norm(v2)

# Para calcular el nos aprovechamos de la igualdad de:
# v1^t * v2 = L2(v1) * L2(v2) * cos(a).
cos_a = prod_in / (v1_norm * v2_norm)

# Despejamos el cos.
ang_rad_a = np.arccos(cos_a)

# Pasamos de radiantes a grados.
ang_gra_a = np.rad2deg(ang_rad_a)
print(ang_gra_a) # Nos devuelve 45 lo que nos dice que hay entre v1 y v2 45 grados.
# %%
# VECTORES ORTOGONLAES

import numpy as np
import matplotlib.pyplot as plt

A = np.array([0,0,2,2])
B = np.array([0,0,2,-2])

plt.quiver([A[0], B[0]],
          [A[1], B[1]],
          [A[2], B[2]],
          [A[3], B[3]],
           angles='xy', scale_units='xy', 
           scale=1,
          )
plt.xlim(-2,4)
plt.ylim(-3,3)

plt.show() # Muestra dos vectores que forman 90 grados.

v1 = np.array([2,2])
v2 = np.array([2,-2])

print(v1.dot(v2.T)) # Si nos muestra 0 es porque tienen 90 grados.

# Para que un vector sea ortonormal su normal debe ser 1.

print(np.linalg.norm(v1)) # 2.82
print(np.linalg.norm(v2)) # 2.82

# Ya que no nos da 1 su normal concluimos que no son ortonormales.
# Pero para lograr que un vector sea ortonormal lo unico que debo realizar
# Es dividir cada uno de sus elementos por su normal.

vector_ortonormal = v1 * (1/np.linalg.norm(v1))
print(np.linalg.norm(vector_ortonormal)) # Nos devuelve 1.

#%%
import numpy as np
tensor = np.array([
    [[1,2,3],[4,5,6],[7,8,9]],
    [[1,5,3],[4,8,6],[7,8,9]],
    [[1,100,1],[4,255,4],[7,7,7]]
])
print(tensor.shape)
# %%


import numpy as np

vector = np.array([-50,-25,0,25,100,-300])

v0_norm = np.linalg.norm(vector)
print (v0_norm)
# %%
import numpy as np

M = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9,10, 11,12]])

print (M.T*[3,2])
# %%

matriz = np.array([[1,2],[4,5],[5,6]])
vector = np.array([[1],[2]])
producto_interno = matriz.dot(vector)

print(producto_interno)


# %%
