#############################################################################
#                                    MEDIA                                  #
#############################################################################
import math
def media(lista):
    return sum(lista)/len(lista)

def Varianza(lista):
    acum = 0
    mu = media(lista)
    for x in lista:
        acum += (x-mu)**2
    return acum/len(lista)

def Desviacion_Estandar(lista):
    return math.sqrt(Varianza(lista))


def List (extension):
    lista = []   

    for i in range(extension):

        x = int(input(f'Ingrede el valor del dato {i+1}: '))
        lista.append(x)
    return lista

if __name__ == '__main__':
    
    n = int(input('Ingrese el numero de terminos de la lista: '))
    lista = List(n)
    mu = media(lista)
    sigma_cuadrad = Varianza(lista)
    sigma = Desviacion_Estandar(lista)
    print (f'Arreglo: {lista}')
    print (f'La media es: {mu}')
    print (f'La Varianza es: {sigma_cuadrad}')
    print (f'La desviacion Estanar es: {sigma}')


#%%
import random
import statistics

if __name__ == '__main__':

    edades = [random.randint(1,35) for i in range (20)]
    print(edades)
    print(statistics.mean(edades))

#%%


import numpy as np 

if __name__ == "__main__":
    X = np.random.randint(1 , 21 , size=20)
    mu = np.mean(X)
    Var = np.var(X)
    Sigma = np.std(X)
    print(X)
    print(f'Mean = {mu}, Variance {Var} , Std {Sigma}')