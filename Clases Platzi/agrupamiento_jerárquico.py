import math
from random import randint

class Punto():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def getData(self):
        return [self.x, self.y]

    def point_print(self):
        return 'p(' + str(self.x) + ', ' + str(self.y) + ')'

    def __repr__(self):
        return self.point_print()

    def __str__(self):
        return self.point_print()

    def isEqual(self,otro_punto):
        return (self.x == otro_punto.x) and (self.y == otro_punto.y)

def metrica_euclidiana(punto1, punto2):
    return math.sqrt((punto1.x - punto2.x)**2 + (punto1.y - punto2.y)**2)

def metrica_manhattan(punto1, punto2):
    return abs(punto1.x - punto2.x) + abs(punto1.y - punto2.y)

def metrica_chebyshev(punto1, punto2):
    return max(abs(punto1.x-punto2.x),abs(punto1.y - punto2.y))

def listar_distancias(lista_puntos, funcion_distancia):
    distancias = []
    copy_list = lista_puntos[:]
    while len(copy_list)>1:
        ref = copy_list.pop(0)
        point_list = []
        for punto in copy_list:
            dist = funcion_distancia(ref, punto)
            metric = (ref.getData(), punto.getData(), dist)
            point_list.append(metric)
        point_list.sort(key=lambda p: p[2])  # Se usa para ordenar la tupla
        distancias.append(point_list[0])
    return distancias


if __name__ == "__main__":
    num_puntos = int(input('Ingrese la cantidad de puntos a generar: '))
    dist_functions = [metrica_euclidiana,metrica_manhattan,metrica_chebyshev]
    dist_select = int(input('''
            Seleccione el tipo de medicion:
            [0] = Euclidiana
            [1] = Manhattan
            [2] = Chebyshev
            
            Seleccion: '''))
    #Generamos un rango aleatorio de puntos
    puntos = []
    for _ in range(num_puntos):
        punto = Punto(randint(0,100), randint(0,100))
        puntos.append(punto)

    print(f'Lista inicial de puntos:\n{puntos}')

    while len(puntos) > 1:
        distancias = listar_distancias(puntos, dist_functions[dist_select])
        #print(f'minimos:\n{distancias}')
        for metrica in distancias:
            p1 = Punto(metrica[0][0],metrica[0][1])
            p2 = Punto(metrica[1][0],metrica[1][1])
            new_p = Punto((p1.x + p2.x)//2,(p1.y + p2.y)//2)
            for p in puntos:
                if p.isEqual(p1):
                    puntos.remove(p)
                    puntos.append(new_p)
            for p in puntos:
                if p.isEqual(p2):
                    puntos.remove(p)
        print(f'Lista de puntos:\n{puntos}')


#%% 


import random
import math
from bokeh.plotting import figure, show


class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_coordinates(self):
        return (self.x, self.y)

    def __str__(self):
        return f'{self.x}#{self.y}'


def generate_random_vectors():
    vectors = []
    for _ in range(10):
        vector = Vector(random.randint(0, 10), random.randint(0, 10))
        vectors.append(vector)
    return vectors


# We can use euclidian distance because we are dealing with a cartesian plane
def get_euclidian_distance(vector_a, vector_b):
    return math.sqrt((vector_a.x - vector_b.x)**2 + (vector_a.y - vector_b.y)**2)


def run():

    vectors = generate_random_vectors()

    # Generating a graph of the vectors with bokeh
    x_points = [vector.x for vector in vectors]
    y_points = [vector.y for vector in vectors]

    p = figure(plot_width=400, plot_height=400)

    # add a circle renderer with a size, color, and alpha
    p.circle(x_points, y_points, size=20, color="navy", alpha=0.5)

    # show the results
    show(p)

    clusters = []

    # First, I define a first cluster formed by a first random vector
    initial_vector = random.choice(vectors)
    clusters.append([initial_vector])
    vectors.remove(initial_vector)

    # I'll implement memoization with a dict of distances
    distances = {}

    # I'll work with the first cluster of my clusters list
    cluster = clusters[-1][::]

    while vectors:
        closest_relationship = ''
        for vector_tracked in cluster:
            for vector in vectors:
                if f'{vector_tracked}*{vector}' not in distances.keys():
                    distance = get_euclidian_distance(
                        vector_tracked, vector)
                    distances[f'{vector_tracked}*{vector}'] = distance
                if closest_relationship:
                    if distances[closest_relationship] > distances[f'{vector_tracked}*{vector}']:
                        closest_relationship = f'{vector_tracked}*{vector}'
                else:
                    closest_relationship = f'{vector_tracked}*{vector}'

        next_vector_x = int(closest_relationship.split('*')[1].split('#')[0])
        next_vector_y = int(closest_relationship.split('*')[1].split('#')[1])

        for vector in vectors:
            if vector.x == next_vector_x and vector.y == next_vector_y:
                cluster.append(vector)
                vectors.remove(vector)
                clusters.append(cluster[::])
                break

    for cluster in clusters:
        for vector in cluster:
            print(vector)
        print('\n')
        print('*' * 20)
        print('\n')


if __name__ == "__main__":
    run()