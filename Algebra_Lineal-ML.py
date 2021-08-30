#%%                                       MATRICES A TRANSFORMACIONES LINEALES    

from matplotlib import scale
from numpy.lib.function_base import angle
from numpy.lib.shape_base import apply_along_axis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from graficarVectores import graficarVectores   # importe de funciones de otro archivo py. 
# from Funciones_Auxiliares-Algebra_Lineal.graficarVectores import graficarVectores       # Si se necesita de carpeta y funciones especificas


A = np.array ([[-1, 3], [2, -2] ])      # matriz
print(A)

vector = np.array([[2], [1]])
print(vector)

# %run "..\\Funciones_Auxiliares-Algebra_Lineal\\graficarVectores.py"   # Si se necesita de carpeta especifica desde colab

print(vector.flatten())     # vector que esta en columna convierte en tira o vector estirado

print(A)
print(A.flatten())      # .flatten() → convierte el vector o matriz en un vector fila

# graficar Vectores con la funcion
graficarVectores([vector.flatten()], cols = 'blue')     # llamado a la funcion
plt.xlim(-0.5, 3)
plt.ylim(-0.5, 2)
plt.show()

# Ver cual es la transformacion
vector_transformado = A.dot(vector)    # porducto interno de la matriz A aplicado al vector 
                                        # del vector [2 ,1] se cambia a [1, 2]
print(vector_transformado)

graficarVectores([vector.flatten(), vector_transformado.flatten()],     # grafica los dos vectores, original y tranformado
                    cols = ['blue', 'orange']
                    )
plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 2.5)
plt.show()

print(np.linalg.det(A))         # resul = -4; determinante de la matriz: si éste es negativo, el vector resultante (de la multiplicación matriz-vector (A.dot(vector))) se moverá en sentido antihorario; si es positivo, el vector se moverá en sentido horario.

print(np.linalg.norm(vector))
print(np.linalg.norm(vector_transformado))      # Se tiene la misma norma en los dosa casos

#%%                              AUTOVALORES Y AUTOVECTORES

# Autovector = cuando se le aplica transformaciones tiene misma direccion pero puede tener amplitud o sentido distinta
"""
- Para conseguir los autovalores y autoverctores de la matriz A, esta debe ser cuadrada (ej: 2x2, 3x3, 9x9…)
- La matriz A tendrá tantos autovalores como dimensión tenga A (ej: una Matriz de 3x3 tiene 3 autovalores, matriz de 2x2 tiene dos autovalores)
- Los autovalores pueden repetirse
- Estos autovalores son los que forman los autovectores
- Los autovectores deben ser base, es decir, que desde esos autovectores se pueda generar todo el espacio o demás vectores
"""
import numpy as np
import matplotlib.pyplot as plt

from graficarVectores import graficarVectores

orange_ligth = '#FF9A13'
blue_ligth = '#1190FF'

X = np.array([[3, 2], [4, 1]])      # definicion de la matrix

print(X)

v = np.array([[1],[1]])
print(v)

u = X.dot(v)        # tranformada del vector
print(u)

graficarVectores([u.flatten(), v.flatten()], 
                    cols=[orange_ligth, blue_ligth])
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.show()

lambda_1 = 5
print(lambda_1*v)       # Autovector = cuando se le aplica transformaciones tiene misma direccion y sentido pero puede tener amplitud distinta

# otro caso igual
s = np.array([[-1], [2]])
print(s)

t = X.dot(s)
print(t)

graficarVectores([t.flatten(), s.flatten()],
                    cols=[orange_ligth, blue_ligth])

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()

# %%                    CALCULO DE AUTOVALORES Y AUTOVECTORES

import numpy as np
import matplotlib.pyplot as plt
from graficarVectores import graficarVectores

X = np.array([[3, 2], [4, 1]])      # definicion de la matrix
print(X)

print(np.linalg.eig(X))        # eigenvalues and eigenvectors  o autovalores y autovectores

autovalores, autovectores = np.linalg.eig(X)
print(autovalores)

# autovalor asociado a cada autovector
print(autovectores[:, 0])    # contenido de la col 0 del autovector
print(autovectores[:, 1])

v = np.array([[-1], [2]])
Xv = X.dot(v)       # tranformacion del vector, producto interno
v_np = autovectores[:, 1]
print(v_np)

# grafica de los tres casos
graficarVectores([Xv.flatten(), v.flatten(), v_np], 
                    cols = ['green', 'orange', 'blue']
                )
plt.ylim(-3, 2)
plt.xlim(-2, 2)
plt.show()      # los tres son el mismo vector ya que conservan la misma direccion, y cambian su aplitud y sentido en algunos casos
                # solo cambia su autovalor asociado

"""Los autovectores encontrados por numpy son un múltiplo del propuesto en el ejercicio. Los autovectores son los mismos, lo que varía es la amplitud o el sentido"""





# %%                        DESCOMPOSICION DE MATRICES

import numpy as np
import matplotlib.pyplot as plt

# A = np.array([[3, 2], [4, 1]])
A = np.array([[3, 4], [3, 2]]) 
print(A)

autovalores, autovectores = np.linalg.eig(A)
print("autovectores")
print(autovectores)
print("autovalores")
print(autovalores)

# A_calc = autovectores.dot(np.diag(autovalores)).dot(np.linalg.inv(autovectores))      # autovectores producto interno de la diagonal de los autovales, y se hace producto interno de la inversa de los auvectores
# print(A_calc)        # Se tiene que A = A_calc

# C = np.array([[3, 2], [2, 3]])
# print(C)
# print (C == C.T)        # Matiz simetrica, matriz igual a su transpuesta

# autovalores, autovectores = np.linalg.eig(C)
# print(autovectores)
# print(autovalores)

# C_calc = autovectores.dot(np.diag(autovalores)).dot(autovectores.T)     # calcular la transpuesta es mucho mas sencillo y mas economico que una inversa 
# print(C_calc)

# """
# Descomposición de matrices
# Consiste en reescribir una matriz cuadrada X como un producto de A x B x C, es decir X = AxBxC, donde:

# A: es la matriz formada por los autovectores
# B: matriz diagonal formada por los autovalores
# C: matriz inversa de A.
# Nota: En matrices reales y simétricas, C = A.T (matriz traspuesta de los autovectores). Esta propiedad tiene menor costo computacional.

# Recordatorio: Una matriz cuadrada es aquella que tiene igual número de filas y columnas, y cuyos vectores que la componen son linealmente independientes.
# """

# # %%                    # DESCOMPOSICION DE MATRICES NO CUADRADAS

# """ '
# Se lo hace por descomposicion de valores singulares

# U
# V
# D

# matriz U y V son orgonal, es decir sus vectores son ortonormales, no deben tener las mismas dimensiones
# D es una matriz diagonal, tiene en la diagonal todos los valores singulares y 0s fuera de la diagonal
# V posee los vetores derechos singulares
# U posee vecores izquierdos singulares
# """
# import numpy as np
# # A = np.array([[1,2,3], [3, 4, 5]])
# A = np.array([[3, 4] ,[3, 2]])
# print(A)

# U, D, V = np.linalg.svd(A)      # svd ¿ calcula los valaores singulares
# print(U)
# print("D")
# print(D)
# # print(np.diag(D))       # matriz de los valores en la diagonal
# print(V)

# """ 
# Tenemos 2 matices de 2x2 y 1 de 3x3, ya que no se puede multiplicar entre ellas se debe
# aumentar una columna extra de 0’s al vector D para que ahora sea de 2x3 y realice la 
# multiplicacion y el metodo SVD queda comprobado
# """

# magicMatrix = np.array([[D[0],0,0],[0,D[1],0]])
# A_calc = U.dot(magicMatrix).dot(V)
# print(A_calc)


# # %%                        LAS TRES TRANSFORMACIONES

# import numpy as np
# import matplotlib.pyplot as plt

# from graficarMatriz import graficarMatriz       # llamda a funcion auxiliar

# A = np.array([[3, 7], [5, 2]])
# print(A)

# print('Circulo Unitario:')
# # graficarMatriz(np.array([[1,0], [0,1]]))        # se envia la matriz identidad para ver sin ninguna modificacion
# graficarMatriz(np.eye(2))
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
# plt.show()

# print('Circulo Unitario Transformado:')
# graficarMatriz(A)        # se envia la matriz identidad para ver sin ninguna modificacion
# plt.xlim(-8, 8)
# plt.ylim(-8, 8)
# plt.show()

# ###                 APLICACION DE LAS MATRICES U, V, D & SUS EFECTOS EN LAS TRANSFORMACIONES

# U, D, V = np.linalg.svd(A)
# print(U)

# print('Circulo Unitario:')
# graficarMatriz(np.eye(2))
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
# plt.show()

# print('Primer rotacion V:')     # primer transformacion efectua V, gira los dos ejes y & x
# graficarMatriz(V)
# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)
# plt.show()

# print('Escala (D):')            # aumenta o reduce el espacio (escala), no en todas las direcciones de la misma forma
# graficarMatriz(np.diag(D).dot(V))       # recosntruye la transformacion que primero hiso V y se le aplica D
# plt.xlim(-9, 9)
# plt.ylim(-9, 9)
# plt.show()

# print('Segunda rotacion U:')        # termina generr la rotacion
# graficarMatriz(U.dot(np.diag(D).dot(V)))
# plt.xlim(-8, 8)
# plt.ylim(-8, 8)
# plt.show()

# print('Circulo Unitario Transformado:')
# graficarMatriz(A)        # se envia la matriz identidad para ver sin ninguna modificacion
# plt.xlim(-8, 8)
# plt.ylim(-8, 8)
# plt.show()

# """
# La matriz V rota el espacio
# La matriz D escala el espacio.
# La matriz U rota de nuevo el espacio.
# La transformación del espacio de una matriz A es igual a la transformación de las matrices SVD (Valores singulares)
# """

# """
# Efecto de la descomposición de una matriz en vectores singulares:
# V → rota el espacio
# D → Escala el espacio (agranda o encoje los vectores, también puede cambiar su sentido)
# U → rota nuevamente los vectores

# Nota:
# La descomposición por valores singulares tiene efectos similares:

# autovectores → rota el espacio
# diag(autovalores) → escala el espacio
# inv(autovectores) → rota el espacio
# """


# # %%                            INTERPRETACION DE LOS VALORES SINGULARES

# import numpy as np
# import matplotlib.pyplot as plt
# from graficarMatriz import graficarMatriz
# from graficarVectores import graficarVectores

# # en la descomposicion svd se tiene 3 matrices la V, D y U donde D es una matriz diagonal compuesta de los valores ingulares

# A = np.array([[3, 7], [5, 2]])
# print(A)

# U, D, V = np.linalg.svd(A)

# print(D[0])     # que se tiene en cada componente de la matriz diagonal
# print(D[1])

# u1 = [D[0]*U[0,0], D[0]*U[0,1]]     # u1 se trandforma por D0
# v1 = [D[1]*U[1,0], D[0]*U[1,1]]  

# print([A[0,0], A[1,0]])
# print(u1)
# print()
# print([A[0,1], A[1,1]])
# print(v1)

# # grafica de las matrices

# graficarMatriz(A)
# graficarVectores([u1, v1], cols = ['red', 'blue'])

# plt.text(3, 5, r"$u_1$", size = 18)             # agrara texto(luagar 3, 5 es U1, atamanio 18)
# plt.text(7, 2, r"$v_1$", size = 18)

# plt.text(-5, -4, r"$D(u_1)$", size = 18, )            # son los valores singulares que modificacan a u1
# plt.text(-4, 1, r"$D(v_1)$", size = 18)

# plt.xlim(-8, 8)
# plt.ylim(-8, 8)
# plt.show()


# """
# e demuestra que esos vectores u1 y v1 no son mas que las filas del producto interno de D·U
# """



# # %%
