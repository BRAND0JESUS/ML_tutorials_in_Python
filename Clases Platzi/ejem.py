#%%                 ORDENAMSIENTO POR BURBUJA E INSERCION

import random

defordenamiento_de_burbuja(lista):
    n = len(lista)

    for i in range(n): #recorremos la lista n veces
        for j in range(0, n -i - 1): #recorremos la lista n-i -1
            #O(n)* O(n-i-1) = O(n*n) = O(n**2)   (Simbolo * porque tenemos la lista dentro de la lista) 

            if lista[j] > lista[j+1]:
                lista[j],lista[j+1] = lista[j+1],lista[j]

    return lista

defordenamiento_por_insercion(lista):

    for indice in range(1, len(lista)):
        valor_actual = lista[indice]
        posicion_actual = indice
       

        while posicion_actual > 0 and lista[posicion_actual - 1] > valor_actual:
            lista[posicion_actual] = lista[posicion_actual - 1]
            posicion_actual -= 1

        lista[posicion_actual] = valor_actual
    return lista

if __name__ == '__main__':
    tamaño_de_la_lista = int(input('De que tamaño quieres la lista???'))
    lista = [random.randint(0,100) for i in range(tamaño_de_la_lista)]

    
   # print(lista)
    print(f'Lista:{lista}')
    lista_burbuja = ordenamiento_de_burbuja(lista)   
    print(f'Lista ordenada burbuja:{lista_burbuja}')
    lista_inserccion = ordenamiento_por_insercion(lista)   
    print(f'Lista inserccion:{lista_inserccion}')
    

#%%                 ORDENAMIENTO POR MEZCLA
import random


def ord_por_mezcla(lista):
    #caso base
    if len(lista) > 1:
        medio = len(lista) // 2
        izquierda = lista[:medio]
        derecha = lista[medio:]
        print(izquierda, '*' * 5, derecha)

        #llamada recursiva
        izquierda = ord_por_mezcla(izquierda)
        derecha = ord_por_mezcla(derecha)

        #iteradores para recorrer las dos sublistas
        i = 0
        j = 0
        #Iterador para la lista principal
        k = 0

        while i< len(izquierda) and j < len(derecha): #mientras podamos seguir comparando
            if izquierda[i] < derecha[j]:
                lista[k] = izquierda[i]
                i += 1
            else:
                lista[k] = derecha[j]
                j += 1
            k+= 1
        #copiar los pedazos de las listas que quedaron
        while i < len(izquierda):
            lista[k] = izquierda[i]
            i+=1
            k += 1
        while j < len(derecha):
            lista[k] = derecha[j]
            j += 1
            k += 1
        print(f'izquierda  {izquierda}. derecha  {derecha}')
        print(lista)
        print('--' * 30)

    return lista

if __name__ == '__main__':
    tamano_de_lista = int(input('De que tamaño sera la lista? '))

    lista = [random.randint(0, 100) for i in range(tamano_de_lista)]

    print(lista)
    print('-' * 20)
    lista_ordenada = ord_por_mezcla(lista)
    print(lista_ordenada)        