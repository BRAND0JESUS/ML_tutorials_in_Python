import numpy as np
import matplotlib.pyplot as plt
from graficarVectores import graficarVectores


def graficarMatriz(matriz, vectorCol= ['red', 'blue']):
    
    # circulo initario
    x = np.linspace(-1, 1, 100000)      # toma valores del espacio entre -1 y 1, todos los puntos que queremos dentro de X
    y = np.sqrt(1-(x**2))       # representacion del circulo

    # Circulo unitario transformado, despues de aplicarle la matriz,
    x1 = matriz[0,0]*x + matriz[0,1]*y
    y1 = matriz[1,0]*x + matriz[1,1]*y
    x1_neg = matriz[0,0]*x - matriz[0,1]*y
    y1_neg = matriz[1,0]*x - matriz[1,1]*y

    # vectores
    u1 = [matriz[0,0], matriz[1,0]]
    v1 = [matriz[0,1], matriz[1,1]]

    graficarVectores([u1, v1], cols = [vectorCol[0], vectorCol[1]])

    plt.plot(x1, y1, 'green', alpha = 0.7)
    plt.plot(x1_neg, y1_neg, 'green', alpha = 0.7)
