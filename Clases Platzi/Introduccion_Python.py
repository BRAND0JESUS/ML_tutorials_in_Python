##          Funciones y Abstracciones

def opciones():
  print(f'Opciones para hallar raiz cuadrada \n (1) Enumeracion exhaustiva \n (2) Aproximacion de soluciones \n (3) Busqueda binaria') 
  opcion = int(input('Elija una opcion: '))
  numero = int(input('Elija un numero: '))
  if opcion==1:
    Enumeracion(numero)
  elif opcion==2:
    Aproximacion(numero)
  elif opcion==3:
    BusquedaBinaria(numero)
  else:
    print('Elija 1, 2 o 3')

def Enumeracion(objetivo):
  respuesta = 0

  while respuesta**2 < objetivo:
      print(respuesta)
      respuesta += 1

  if respuesta**2 == objetivo:
      print(f'La raiz cuadrada de {objetivo} es {respuesta}')
  else:
      print(f'{objetivo} no tiene una raiz cuadrada exacta')

def Aproximacion(objetivo):
  epsilon = 0.001
  paso = epsilon**2 
  respuesta = 0.0

  while abs(respuesta**2 - objetivo) >= epsilon and respuesta <= objetivo:
      #print(abs(respuesta**2 - objetivo), respuesta)
      respuesta += paso

  if abs(respuesta**2 - objetivo) >= epsilon:
      print(f'No se encontro la raiz cuadrada {objetivo}')
  else:
      print(f'La raiz cudrada de {objetivo} es {respuesta}')


def  BusquedaBinaria(objetivo):
  epsilon = 0.001
  bajo = 0.0
  alto = max(1.0, objetivo)
  respuesta = (alto + bajo) / 2

  while abs(respuesta**2 - objetivo) >= epsilon:
      print(f'bajo={bajo}, alto={alto}, respuesta={respuesta}')
      if respuesta**2 < objetivo:
          bajo = respuesta
      else:
          alto = respuesta

      respuesta = (alto + bajo) / 2
  print(f'La raiz cuadrada de {objetivo} es {respuesta}')

opciones()


#%%             Scope o Alcance

def func1(un_arg,una_func):
	a="Prueba Scope"
	def func2(otro_arg):
		print(a)
		return otro_arg*2

	valor = func2(un_arg)
	return una_func(valor)

un_arg = 1

def cualquier_func(cualquier_arg):
	return cualquier_arg + 5


resultado = func1(un_arg,cualquier_func)
print(resultado)

# %%   Especificaciones del código

# Que hace la instrucción.
# Que significan los parametros.
# Que es lo que devuelve nuestra instrucción.

def suma(a, b):
	"""
    Suma dos valores a y b.

	param int a cualquier entero
	param int b cualquier entero
	returns la sumatoria de a y b
	"""

	total = a + b
	return total

help(suma)




# %%        Recursividad
# Funciones dentro de fuciones
# Serie de Fibonacci

numero = int(input ('escoge un número para la secuencia fibonacci: \n'))

i = 0
    
def fibonacci(num):
    """
    Devuelve el valor del numero para la serie fibonacci
    numero int > 0
    returns secuencia fibonacci
    """
    if num == 0 or num ==1:
        return 1
    return fibonacci(num-1) + fibonacci (num-2) 

print ('Puesto', 'Valor')
while (i < numero):
    print (i,'    ', fibonacci (i))
    i = i + 1



# %%            Tipos de datos estructurados

"""
Tuplas 

son secuencias inmutables de objetos

A diferencia de las cadenas pueden contener cualquier tipo de objetos

Pueden utilizarse para devolver varios valores en una funcion
"""

#creacion de una tupla vacia
my_tuple = ()
print(type(my_tuple))

my_tuple = (1, 'dos' , True)

#se puede acceder a las duplas por el index
print(my_tuple[0])
print(my_tuple[1])
print(my_tuple[2])

#no se pueden modificar las tuplas, provoca error
# my_tuple[0] = 2

#no se pueden modificar pero si pueden apuntar a otro lugar de memoria
my_tuple = (1)

#tipo de objeto int debido a que no se uso sintaxis adecuada
print(type(my_tuple))

#los items deberan estar separados por comas
my_tuple = (1,)
print(type(my_tuple))

#tuplas pueden sumarse
#la variable no se modifica, apunta a otro lugar de memoria
my_other_tuple = (2,3,4)
my_tuple += my_other_tuple
print(my_tuple)

#se pueden asignar los items de las tuplas a otras variables
x, y, z = my_other_tuple
print (x,y,z)

#se pueden asignar el retorno de las funciones a una tupla
def coordenadas():
    return(5,4)

coordenada = coordenadas()
print(coordenada)

#asignar los items de la tupla a variables x y
x , y = coordenada

print(x, y)




# %%            Rangos

# range (comienzo, fin, pasos)
my_range = range(1,5)
print(type(my_range))

# ¿Que rango generamos?
# Vamos a colocarlo directamente en un for loop, para observar los valores que tiene

for i in my_range:
    print(i)

#Al ejecutar el codigo vamos a observar que comienza de 1 y termina en 4, no se encuentra el valor 5 (final).

my_range2 = range(0,7,2) #de 0 a 7 y que se vaya de 2 en 2 
my_other_range = range(0,8,2)

print(my_range2 == my_other_range)

for r2 in my_range2:
    print(r2)

for otr in my_other_range:
    print(otr)

print(id(my_range2))
print(id(my_other_range))

print(my_range2 is my_other_range) #Operador de igualdad de objetos "is"

#Los rangos nos generan una secuencia de enteros

for par in range(0,101,2):
    print(par)

for nones in range(1,100):
    if nones%2!=0:
        print(nones)
 #uso % (modulo), ya que lo que hace es mostrarme todos los números que al momento de dividirlos, me sobre 1, es decir sean impares. De tal manera que nones % 2 ==0 representa a los números pares, ya que al momento de dividirlo entre 2 no me sobra ningún numero.

# %%            LISTA Y MUTABILIDAD
 """
 lista.extend(iterable) #extiende la lista con valores dentro de un iterable como un range()
lista.insert(i, ‘valor’) #Agrega un valor en la posición i y recorre todos los demás. No borra nada.
lista.pop(i) #Elimina valor en la posición i de la lista.
lista.remove(‘valor’) #Elimina el primer elemento con ese valor.
lista.clear() #Borra elementos en la lista.
lista.index(‘valor’) #Retorna posición del primer elemento con el valor.
lista.index(‘valor’, start, end) #Retorna posición del elemento con el valor dentro de los elementos desde posición start hasta posición end)
lista.count(‘valor’) #Cuenta cuántas veces esta ese valor en la lista.
lista.sort() #Ordena los elementos de mayor a menor.
lista.sort(reverse = True) #Ordena los elementos de menor a mayor.
lista.reverse() #Invierte los elementos
lista.copy() #Genera una copia de la lista. También útil para clonar listas.
"""

# Crear una lista:
mylist = ['one', 20, 5.5, [10, 15], 'five']

# listas mutables:
mylist = ['one', 'two', 'three', 'four', 'five']
mylist[2] = "New item"
# Si el índice es negativo, cuenta desde el último elemento.
elem = mylist[-1]

# Recorrer una lista:
for elem in mylist:
print(elem)

# Actualizar elementos:
mylist = [1, 2, 3, 4, 5]
for i in range(len(mylist)):
    mylist[i]+=5
print(mylist)

mylist = ['one', 20, 5.5, [10, 15], 'five']
print(len(mylist))

# Cortar una lista:
mylist = ['one', 'two', 'three', 'four', 'five']
mylist[1:3] = ['Hello', 'Seven']
print(mylist)

# Insertar en una lista:
mylist = [1, 2, 3, 4, 5]
mylist.insert(1, 'Hello')
print(mylist)

# Agregar a una lista al final:
mylist = ['one', 'two', 'three', 'four', 'five']
mylist.append("new one")

mylist = ['one', 'two', 'three', 'four', 'five']
list2 = ["Hello", "new one"]
mylist.extend(list2)
print(mylist)

# Ordenar una Lista:
mylist = ['cde', 'fgh', 'abc', 'klm', 'opq']
list = [3, 5, 2, 4, 1]
mylist.sort()
list.sort()
print(mylist)
print(list)

# Invertir una lista:
mylist = [1, 2, 3, 4, 5]
mylist.reverse()
print(mylist)

# Indice de un elemento:
mylist = ['one', 'two', 'three', 'four', 'five']
print(mylist.index('two'))

# Eliminar un elemento:
mylist = ['one', 'two', 'three', 'four', 'five']
removed = mylist.pop(2)
print(mylist)
print(removed)

mylist.remove('two')
del mylist[2]

mylist = ['one', 'two', 'three', 'four', 'five']
del mylist[1:3]
print(mylist)

# Funciones agregadas:
mylist = [5, 3, 2, 4, 1]
print(len(mylist))
print(min(mylist))
print(max(mylist))
print(sum(mylist))

# Comparar listas:
mylist = ['one', 'two', 'three', 'four', 'five']
list2 = ['four', 'one', 'two', 'five', 'three']
if (mylist == list2):
     print("match")
else:
     print("No match")

# Operaciones matematicas en las listas:
list1 = [1, 2, 3]
list2 = [4, 5, 6]
print(list1 + list2)
print(list1 * 2)

# Listas y cadenas:
mystr = "LikeGeeks"
mylist = list(mystr)
print(mylist)

mystr = "LikeGeeks"
mystr = "Welcome to likegeeks website"
mylist = mystr.split()
print(mylist)

# Unir una lista:
mylist = ['Welcome', 'to', 'likegeeks', 'website']
delimiter = ' '
output = delimiter.join(mylist)
print(output)

# Aliasing:
mylist = ['Welcome', 'to', 'likegeeks', 'website']
list2 = mylist
list2[3] = "page"
print(mylist)

# %%            CLONAR LISTAS

a = [1, 2, 3]
print (a)
print (id(a))
b=a
print (id(b))
c = list(a)   # Con la funci[on list se tiene los mismo elementos con diferente objetos de memoria
print (id(c))
d = a[::]       # Misma funcion que list
print (id(d))


# %%            LIST CONMPREHENSION
my_list = list(range(100))   # 1,2,3...99

double = [i*2 for i in my_list]     # se multiplica *2
#print(double)

pares = [i for i in my_list if i % 2 == 0]
print(pares)


# %%            DICCIONARIOS

capitales = {
    'Venezuela': 'Caracas',
    'Colombia': 'Bogota',
    'Argentina': 'Cordova',
    'Canada': 'Ottawa'
}
print(capitales)
print(capitales['Venezuela']) # Nos devuelve el valor de Venzuela
print(capitales['Argentina']) # Nos devuelve el valor de Argentina

print(capitales.get('Mexico')) # Nos devuelve None
print(capitales.get('Mexico', 'Ciudad de Mexico')) # Nos devuelve Ciudad de Mexico

a = 'Buenos Aires'
capitales['Argentina'] = a # Modificamos el valor de llave Argentina y le pasamos el valor de la variable a
print(capitales)

capitales['Peru'] = 'Lima' # Le agregamos la llave Peru con el valor Lima al diccionario
print(capitales)

del capitales['Canada'] # Eliminamos la llave Canada
print(capitales)

#del capitales['Lima'] # Esto nos dara error, porque no hay una llave llamada Lima
#print(capitales)

del capitales['Peru'] # Aca eliminamos la llave Peru y por ende se elimina su valor; peru
print(capitales)

for key in capitales.keys(): # Iteramos en las todas las llaves de capitales
    print(key) # Imprimimos las llaves

for value in capitales.values(): # Iteramos en todos los valores de capitales
    print(value) # Imprimimos todos los valores de capitales

for key, value in capitales.items(): # Iteramos en las llaves y valores de capitales
    print(key, value) # Imprimimos las llaves y valores de capitales gtfrrrrrrrrrrrrrrrrrrrrrrrrx


# %%             PRUEBA DE CAJA NEGRA

import unittest

def suma(num1, num2):
    # Si hacemos un cambio en el return donde queremos obtener el numero absoluto abs(num1) esto provocara un error en el test para sumar negativos, al hacer un cambio y tener casos de prueba para validar el funcionamiento correcto esto provoca que lo solucionemos inmediatamente conjunto a prever el desconocimiento de donde ocurre o ocurra este error
    return num1 + num2
# Esta creacion del objeto crea automaticamente los testeo dentro del modulo de python
class CajaNegraTest(unittest.TestCase):
    # Las pruebas desde funciones
    def test_suma_dos_positivos(self):
        num_1 = 10
        num_2 = 5

        resultado = suma(num_1, num_2)

        # Esto funciona asi:
        # (valor, valorQuerido) = valor == valorQuerido (Nos devuelve un true o false)
        self.assertEqual(resultado, 15)
        # Esto funciona asi:
        #  (valor, valorQuerido) = valor > valorQuerido (Nos devuelve un true o false)
        self.assertGreater(resultado, 14)
        # Esto funciona asi:
        #  (valor, valorQuerido) = valor >= valorQuerido (Nos devuelve un true o false)
        self.assertGreaterEqual(resultado, 15)
        # Esto funciona asi:
        #  (valor, valorQuerido) = valor < valorQuerido (Nos devuelve un true o false)
        self.assertLess(resultado, 16)
        # Esto funciona asi:
        #  (valor, valorQuerido) = valor <= valorQuerido (Nos devuelve un true o false)
        self.assertLessEqual(resultado, 15)
    
    def test_suma_dos_negativos(self):
        num_1 = -10
        num_2 = -7

        resultado = suma(num_1, num_2)

        self.assertEqual(resultado, -17)
    

if __name__ == '__main__':
    unittest.main()




# %%            MANEJO DE ECEPCIONES
""" ImportError : una importación falla;
IndexError : una lista se indexa con un número fuera de rango;
NameError : se usa una variable desconocida ;
SyntaxError : el código no se puede analizar correctamente
TypeError : se llama a una función en un valor de un tipo inapropiado;
ValueError : se llama a una función en un valor del tipo correcto, pero con un valor inapropiado"""



"""Creamos una función en donde cada elemento de 
una lista es dividida por un divisor definido"""
def divide_elementos_de_lista(lista, divisor):
    """El programa intentara realizar la división"""
    try:
        return [i / divisor for i in lista]
    
    # En caso de error de tipo ZeroDivisionError que
    # significa error al dividir en cero, el programa
    # ejecutara la siguiente instrucción"""
    except ZeroDivisionError as e:
        return lista

lista = list(range(10))
divisor = 0

print(divide_elementos_de_lista(lista, divisor))

# %%            AFIRMACIONES

def primera_letra(lista_palabras):
    primeras_letras = []
    
    for palabra in lista_palabras:
        try:
            assert type(palabra) == str, f'{palabra} no es String'
            assert len(palabra) > 0 , 'No se permiten vacios'
            primeras_letras.append(palabra[0])
        except AssertionError as e:
            print(e)

    return primeras_letras


lista = ['Angelo',5.5, '', 2 , '43952353', 0.35]
print('Primeras letras validas son : ' , primera_letra(lista))

# %%
a=5/'Pla'
print (a)
# %%
a = True
b = False
print (a and b) 
# %%
a = '123' + '456'
print (a)
# %%            ESTADISTICA


import numpy as np
from statistics import mode,median
from collections import Counter
temps = np.array([3,35,30,37,27,31,41,20,16,26,45,37,9,41,28,
                  21,31,35,10,26,11,34,36,12,22,17,33,43,19,
                  48,38,25,36,32,38,28,30,36,39,40])
valores = np.unique(temps)
print(Counter(temps))
print('valores :',valores)
print('Media: ',round(temps.mean(),2))
print('Moda: ',mode(temps))
print('Mediana: ',median(temps))
print('Varianza: ',round(temps.var(),2))
print('ubicacion del valor superior: posicion {} , numero {}'.format(temps.argmax(),temps[temps.argmax()])) 
print('ubicacion del valor inferior: posicion {} , el numero es {}'.format(temps.argmin(),temps[temps.argmin()]))
# %%        Q-LEARNING


import numpy as np
Q = np.zeros((state_size, action_size))

import random
epsilon = 0.2

if random.uniform(0,1) < epsilon:

else:
    Q[state, action] = Q[state, action] + lr * (reward + gama * np.max(Q[new_state,:]) - Q[state, action])
# %%

import numpy as np
from sklearn.linear_model import LogisticRegression

def regresion():
    #Datos a utilizar
    xlist = [i*0.25 for i in range(2,22)]
    horas = np.array(xlist).reshape(-1,1)
    aprobado = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1])

    #Creamos la instancia de la regresion y le damos los datos
    regresion_logistica = LogisticRegression()
    regresion_logistica.fit(horas,aprobado)

    #Generamos una nueva lista de horas para hacer una prediccion y se ejecuta
    horas_prediccion = np.array([i for i in range(1,7)]).reshape(-1,1)
    prediccion = regresion_logistica.predict(horas_prediccion)
    probabilidad = regresion_logistica.predict_proba(horas_prediccion)

    #Se muestran los resultados
    np.set_printoptions(3)  #Ajustamos la visualizacion
    print('Datos de la prediccion realizada:')
    print('Horas:        {}'.format(horas_prediccion.reshape(1,-1)))
    print('Aprobado:     {}'.format(prediccion))
    print('Probabilidad: {}'.format(probabilidad[:,1]))

if __name__ == '__main__':
    regresion()
# %%            Support Vector Machine

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
X, y = make_blobs(n_samples=40, centers=2, random_state=6)  # make_blobs = divide en hiperplanos los 100 datos

# fit the model, don't regularize for illustration purposes   
clf = svm.SVC(kernel='linear', C=1000)      # clf es clasiffier
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)  # Scater = establece la dispersion de los datos

# plot the decision function
# stablecieron los ejes "x" , "y"
ax = plt.gca()      
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)      # establece los límites y escalas
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)        #e define la malla de datos para que esten ordenados
xy = np.vstack([XX.ravel(), YY.ravel()]).T    # Datos se distribuyan en la malla
Z = clf.decision_function(xy).reshape(XX.shape)   # función de decisión

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()



# %%            ALGORITMO GENETICO

import random

modelo = [1,2,3,4,5,5,4,3,2,1]
largo = 10
num = 20
pressure = 3
mutation_chance = 0.2

#Crea aleatoriamente las caracteristicas (ADN) de cada individuo
def individual(min,max):
    return[random.randint(min, max) for i in range(largo)]

#Genera la poblacion deseada (num)
def crearPoblacion(): 
    return[individual(1,9) for i in range(num)]
    
#Compara cada caracteristica del individuo con su contraparte del modelo y cuenta las coincidencias
def calcularFitness(individual):
    fitness = 0
    for i in range(len(individual)):
        if individual[i] == modelo[i]:
            fitness += 1
    return fitness


def selection_and_reproduction(population):
    #lista de tuplas (fitness, individuo)  de todos los individuos
    puntuados = [ (calcularFitness(i), i) for i in population]
    #print('Puntuados:\n{}'.format(puntuados))

    #Lista ordenada de menor a mayor fitness
    puntuados = [i[1] for i in sorted(puntuados)]
    #print('Puntuados2:\n{}'.format(puntuados))
    population = puntuados

    #seleccion de individuos con mejor puntuacion (cantidad = pressure)
    selected = puntuados[(len(puntuados)-pressure):]
    #print('selected:\n{}'.format(selected))
    
    #reproduccion: Por cada elemento restante (poblacion - selected) sucede:
    #1. se seleccionan dos individuos aleatorios entre los seleccionados
    #2. se escoge un numero aleatorio (punto) de caracteristicas del primer individuo (principio)
    #3. se toman las caracteristicas restantes del segundo individuo (final)
    #4. se reemplaza un elemento de la poblacion.
    for i in range(len(population)-pressure):
        punto = random.randint(1,largo-1)
        padre = random.sample(selected, 2)
        
        population[i][:punto] = padre[0][:punto]
        population[i][punto:] = padre[1][punto:]
        
        #print('Punto: {}\nPadres:\n{}\nNuevo individuo:\n{}'.format(punto, padre, population[i]))
    return population
    
def mutation(population):
    for i in range(len(population)-pressure):
        # Se escoge aleatoriamente quien sufre una mutación.
        if random.random() <= mutation_chance:
            #se escoge una posicion aleatoria en la lista de caracteristicas
            punto = random.randint(0,largo-1)
            #se genera una caracteristica nueva de forma aleatoria
            nuevo_valor = random.randint(1,9)

            # Si el valor obtenido es igual al valor existente en el punto de
            # mutacion se generan valores aleatorios hasta que cambie, luego se
            #inserta el nuevo valor.
            while nuevo_valor == population[i][punto]:
                nuevo_valor = random.randint(1,9)
            population[i][punto] = nuevo_valor
        
    return population
    
def main():
    print("\n\Modelo: %s\n"%(modelo))
    population = crearPoblacion()
    print("Población Inicial:\n%s"%(population))

    for i in range(100):
        population = selection_and_reproduction(population)
        population = mutation(population)
        
    print("\nPoblación Final:\n%s"%(population))
    print("\n\n")

if __name__ == '__main__':
    main()
# %%
a = 
print (a)

# %%
