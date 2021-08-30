def operacional_function():
    lista_xi=[]
    lista_yi=[]
    cantidad_de_xi = int(input("Hola, ingresa la cantidad de datos de la variable independiente: "))
    for i in range(cantidad_de_xi):
        xi=float(input("Ingrese el {}) valor de x: ".format(i+1)))
        lista_xi.append(xi)
        yi = float(input("Ingrese la {}) etiqueta:".format(i+1)))
        lista_yi.append(yi)
    sumax=contador(lista_xi)
    sumay=contador(lista_yi)
    mediax=promediador(sumax,len(lista_xi))
    mediay=promediador(sumay,len(lista_yi))
    productoxy=multiplicador(lista_xi,lista_yi)
    suma_cuadrados=cuadrado_de_sumas_x(lista_xi)
    
    n=len(lista_xi)
    
    
    pendiente = ((sumax*sumay)-(n*productoxy)) / (((sumax)**2)-(n*suma_cuadrados))
    print(pendiente)
    bias = mediay-(pendiente*mediax)
    print(bias)
    print("La ecuación de la recta más optimizadora es {}x + {} = y".format(pendiente,bias))
    valori = int(input("Ingrese un valor para predecir con nuestro modelo: "))
    respuesta = (pendiente*valori) + bias
    print("La prediccion es de  {}.".format(respuesta))






def contador(lista_random):
    acumulador=0
    for numero in lista_random:
        acumulador = numero + acumulador
    return acumulador

def promediador(suma,n_datos):
    media=suma/n_datos
    return media

def multiplicador(lista1,lista2):
    acumulador=0
    for i in range(len(lista1)):
        acumulador = acumulador + lista1[i]*lista2[i]
    return acumulador

def cuadrado_de_sumas_x(lista):
    acumulador = 0
    for numero in lista:
        acumulador = acumulador + numero**2
    return acumulador



if __name__ == "__main__":
    print("Hola, calcularemos una recta que se adecua a la nube de puntos de un problema de regresion lineal")
    operacional_function()    
    