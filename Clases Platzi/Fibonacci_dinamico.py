# Programacion dinamica, cambiar tiemoo por espacio
# debe haber problemas empalmados, se repite la operacion una y otra vez

import sys          # Librería para definir el limite recursivo

def fibo_dinamico(n, memo={}):
    if n <= 0 or n == 1:
        return 1
    try:
        print(f'..Consultando en dic_memo Fibo({n})')
        print()
        return memo[n]
        
    except KeyError:
        print(f'....No existe Fibo({n}) en el diccionario')
        print(f'......Calculando Fibo({n})')
        resultado = fibo_dinamico(n-1, memo) + fibo_dinamico(n-2, memo)
        memo[n] = resultado
        print(f'.........Se guardo Fibo({n})={resultado} en el diccionario')
    return resultado

if __name__ == '__main__':
    sys.setrecursionlimit(100002)       # limito recursivo
    n = 500
    num_fibo_n = fibo_dinamico(n)
    print('*'*40)
    print(f'El numero Fibo({n}) = {num_fibo_n}')
