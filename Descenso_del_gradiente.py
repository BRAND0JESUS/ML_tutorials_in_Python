
#%%                           DESCENSO DEL GRADIENTE
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt

# Hrafica en nuestra funcion de coste

fig, ax = plt.subplots (subplot_kw={"projection" : "3d"})

def f(x,y):
    return x**2 + y**2
    return np.sin(x) + 2*np.cos(y)

res = 100
x = np.linspace(-4, 4, num=res)
y = np.linspace(-4, 4, num=res)
X,Y = np.meshgrid(x,y)      # Vectores con series de puntos, combinacion x,y

Z = f(X,Y)      # Definicion del vextor z
suft = ax.plot_surface(X,Y,Z, cmap = cm.cool)       # Superficie

fig.colorbar(suft)
plt.show()

# Desenso de gradiente

level_map = np.linspace(np.min(Z), np.max(Z), res)      # mapa de niveles
plt.contourf(X,Y,Z, levels = level_map, cmap = cm.cool)   # agraga un contorno  
plt.colorbar()
plt.title("Descenso del Gradiente")


p = np.random.rand(2) * 8 - 4      # Definir punto aleatorio, rand = vector aleatorio entre 0 y 1, (2) = solo usa dos componentes por que es bidimensional
                                   # 8-4 porque 0*8-4=-4 y 1*8-4=4 que son los min y max de mi funcion
#plt.plot(p[0], p[1], 'o', c = 'k')      #grafica cada una de sus componentes, 'o' = sea un punto
# plt.show()

def gradient (p):       
    grad = np.zeros(2)  
    for idx, val in enumerate (p):  
        cp = np.copy(p)
        cp[idx] = cp[idx] + h

        dp = derivate(cp, p)
        grad[idx] = dp


def derivate(_p,p):
  return  (f(_p[0],_p[1]) - f(p[0],p[1])) / h

p = np.random.rand(2) * 8 - 4 # genrea dos var aleatorias, pasos pequeños para que desienda al punto min

plt.plot(p[0],p[1],'o', c='k')

lr = 0.01
h = 0.01

grad = np.zeros(2)      # inicializado en ceros de 2 componentes

for i in range(10000):
  for idx, val in enumerate(p):     # iteracion a traves de cada componente tomando en cuenta nuestro punto , idx = indice, nuestro 0 o 1
    _p = np.copy(p)

    _p[idx] = _p[idx] + h;

    dp = derivate(_p,p) 

    grad[idx] = dp

  p = p - lr * grad

  if(i % 10 == 0):
    plt.plot(p[0],p[1],'o', c='r')

plt.plot(p[0],p[1],'o', c='w')
plt.show()

#####################################################################
# El descenso del gradiente solo sirve para determinar un inimo local
# Solo sirve cuando la funcion posee un unico min local
######################################################################
# %%
