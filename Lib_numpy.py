
import numpy as np
from numpy.core.numeric import ones

np.array([10,20,24,5,15])
a = ([10,30,10,4,30,7,2,4,40,100])
print(a[4])     # posicion 4 del arreglo a

print(a[3:])      # mostrar datos del arreglo a partir de lam posicion 3

print(a[3:7])       # intervalo

print(a[1::4])      # desde la posicion 1 toma pasando cada 4 elementos

print(np.zeros(5))        # arreglo de manera dinamica, arreglo de 5 elemntos cero
print(np.ones((4,5)))       # arreglo bidemensional de 4 filas x 3 colum de elemensy\tos 1

print(type(np.ones(())))       # que tipo de arreglo es?

print(type(np.ones([3])))      # Tipo de elementos

np.linspace(3, 10, 5)     #intervalo min, interval max, num elem flotantes

b = np.array([['x','y','z'], ['a','c','e']])
print(b)
print(type(b))

print(b.ndim)       # cuantas dimensiones se tiene

c = ([12 ,4, 10, 40, 2])
print(c)
print(np.sort(c))       # ordenar los elemntos

        # por que medio se quiere ordenar
cabeceras = [('nombre', 'S10'), ('edad', int)]        # S10 = tipo string de 10 elemnt
datos = [ ('Juan', 10), ('Maria', 70), ('Javier', 42), ('Samuel', 15) ]
usuarios = np.array(datos, dtype=cabeceras)     # integracion de datos y cabecera; dtype: info adicional 
print (usuarios)

print (np.sort(usuarios, order = 'edad'))       # ordenar(dato que queremos ordenar, por que elemento se quiere ordenar)

print(np.arange(25))            #llenado de manera dinamica
print(np.arange(5,30))      # Limite inferior y superior
print(np.arange(5,30, 5))       # Limite inferior y superior, incremento

print(np.full((3,5), 10))       # arreglo bidemenional(cuantos elemntos m x n, valor unico )

print (np.diag([0,3,9,10]))       # diagonal en arreglo bidemencional


#%%
import numpy as np

headers = [('nombre','S10'), ('edad', int), ('nacionalidad','S100')]
data = [('Amelie',21,'Francesa'), 
         ('Valerie', 30,'Canadiense'),
         ('Valentina',25,'Inglesa'),
         ('Alejandra',22,'Cubana')]
users = np.array(data,dtype=headers)
print(np.sort(users, order = 'nacionalidad'))


# %%
