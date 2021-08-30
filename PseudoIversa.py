#%%                             PseudoIversa_MorePenrose
"""
Es una aplicacion directa de svd
Permite resolver en determinados momentos sistemas de ecuaciones lineales

Resumen
La pseudo inversa de Moore Penrose es utilizada cuando en un sistema de ecuaciones lineales representado por Ax = B, x no tiene inversa.
La pseudo inversa de MP es única y existe si se verifican 4 condiciones.
Para calcularla se siguen los siguientes pasos:
Calcular las matrices U, D, y V (matrices SVD) de A.
Construir D_pse: una matriz de ceros que tiene igual dimension de A, y que luego se transpone.
Reemplazar la submatriz D_pse[: D.shape[0], : D.shape[0]] por np.linalg.inv(np.diag(D))
Reconstruir pseudoinversa: A_pse = V.T.dot(D_pse).dot(U.T)

Notas
Para calcularla automaticamente por Python: np.linalg.pinv(A)
Lo que obtenemos con A_pse es una matriz muy cercana a la inversa. Cercano en el sentido de que minimiza la norma dos de estas distancias. O sea, de estos errores que estamos cometiendo.
A_pse no es conmutativa, es decir, A_pse·A ≠ A·A_pse
""" 


import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import NumpyVersion

np.set_printoptions(suppress = True)        # seteo para que no se muestre numeros muy cercanos a cero y mostrar solo cero para que no induzca errores

# A = np.array([[2, 3], [5, 7], [11, 13]])
A = np.array([[1, 2], [3, 4], [5, 6]])
print(A)

U, D, V = np.linalg.svd(A)
print ("U")
print(U)
print ("D")
print(D)
print ("V")
print(V)

D_pse = np.zeros((A.shape[0], A.shape[1])).T    # definir matriz de ceros de tamanio de la matriz original A en las dos dimensiones y trasnponerla
print(D_pse)

print('Los valores a remplazar en D-pse')
print(D_pse[:D.shape[0], :D.shape[0]])       # valores que vamos a remplazar

print('Valores que pondremos en D-pse')
print(np.linalg.inv(np.diag(D)))        # En la matriz de 2x3 los unicos valaores que se remplazaran con D_pse son los que cuadran con la diagonal con la matriz D

# Remplazo
print('D-pse')
D_pse [:D.shape[0], :D.shape[0]] = np.linalg.inv(np.diag(D))
print(D_pse)

# Reconstruir quien es la pseudo inversa
A_pse = V.T.dot(D_pse).dot(U.T)
print(A_pse)

# Validacion de la Pseudo inversa de A
A_pse_calc = np.linalg.pinv(A)      # Pseudo inversa calculada directamente 
print (A_pse_calc)

print(A_pse.dot(A))         # deberia ser cercana a la matriz identidad, el -0 

np.set_printoptions(suppress = False)
print(A_pse.dot(A))

print(A.dot(A_pse))         # deberia ser dar elvalor de la matriz identidad, y tampoco es conmutativa
print(A_pse.dot(A))

A_pse_2 = np.linalg.inv(A.T.dot(A)).dot(A.T)
print(A_pse_2)
print(A_pse)        # son iguales las pseudo inversas



# %%                        PSEUDOINVERSA PARA RESOVER UN SISTEMA SOBRETERMINADO

"""
Se quiere hallar la solucion del sistema de ecuaciones Ax = b, tal que x hace que ||Ax-b||_2 sea minima.
En principio el sistema de ecuaciones es sobredeterminado, lo que implica que las tres rectas no van a cruzarse en un mismo punto.
La solucion dada a través de la pseudo inversa es tal que obedece a los pesos de las ecuaciones. Cada ecuacion como tal ejerce un efecto de gravedad que mueve el punto hacia ella.
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)
# y_1 = -4*x + 3
# y_2 = 2*x +5
# y_3 = -3*x + 1

y_1 = 1 * x + 4
y_2 = 2 * x + 5
y_3 = -3 * x + 6

plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_3)

plt.xlim(-2, 2.5)
plt.ylim(-6, 6)

plt.show()

matriz = np.array([[-1, 1], [-2, 1], [3, 1]])        # representacion de la matriz, coeficientes al despejar las ecuaciones
print(matriz)

matriz_pse = np.linalg.pinv(matriz)         # calculo de la pseudo inversa
print(matriz_pse)

b = np.array([[4], [5], [6]])       # definicion del vector solucion, factores independientes de la ecuacion
print(b)

resultado = matriz_pse.dot(b)       # = rsultado = pseudo inversa con el producto interno con el vector b
print("resultado")
print(resultado)

# Interpretacion grafica

plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_3)

plt.xlim(-2, 2.5)
plt.ylim(-6, 6)

plt.scatter(resultado[0], resultado[1])

plt.show()


