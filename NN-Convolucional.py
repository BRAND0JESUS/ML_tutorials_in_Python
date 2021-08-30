"""
Red neuronal para clasificar imagenes de ropa como, tennis y camisetas
https://www.tensorflow.org/tutorials/keras/classification
"""

from os import access
import tensorflow as tf
from tensorflow import keras        # permite elementos para deep learning

import numpy as np      # trabajar con arreglos
import matplotlib.pyplot as plt
from tensorflow.python.keras import activations
from tensorflow.python.keras.backend import softmax
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers.core import Dense

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()        # divide el dataset en prueba y test

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']        # etiquetas

print(train_images.shape)        # cuantos datos se tiene; output (60000, 28, 28) son 60 mil datos dividido en 28x28 pixeles

plt.figure()
plt.imshow(train_images[100])        # plt.imshow(lo que se quiere mostrar[posicion])
plt.grid()
plt.show()

train_images = train_images / 255.0     # division del train y test entre 255 se hace para “normalizar” los pixels, de manera que esten en el rango 0 a 1.
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))

for i in range (25):        # numero de imagenes que va,os a tener
  plt.subplot(5, 5, i + 1)      # imagnes en secciones
  plt.xticks([])        # da espacio en x, sin etiqueta
  plt.yticks([])
  plt.grid('off')       # que no posee datos y muestre solo infomarcion
  plt.imshow(train_images[i], cmap = plt.cm.binary)     #  cmap = informacion que quiere mostrar
  plt.xlabel(class_names[train_labels[i]])      # etiquetas
plt.show()  

model = keras.Sequential([keras.layers.Flatten(input_shape = (28,28)), 
                            keras.layers.Dense(128, activation= tf.nn.relu), 
                            keras.layers.Dense(10, activation = tf.nn.softmax)])   
                                                            # Se crea la secuencia es division de elementos que permite entrenar
                                                            # La primera capa de esta red, tf.keras.layers.Flatten, transforma el formato de las imagenes de un arreglo bi-dimensional (de 28 por 28 pixeles) a un arreglo uni dimensional (de 28*28 pixeles = 784 pixeles)
                                                            # input_shape = (tamanio de nuestra imagen)
                                                            # la secuencia consiste de dos capastf.keras.layers.Dense. Estas estan densamente conectadas, o completamente conectadas. 
                                                                # La primera capa Dense tiene 128 nodos (o neuronas). La segunda (y ultima) capa es una capa de 10 nodos softmax que devuelve un arreglo de 10 probabilidades que suman a 1. 
                                                                # Cada nodo contiene una calificacion que indica la probabilidad que la actual imagen pertenece a una de las 10 clases.

model.compile(optimizer= tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics = ['accuracy'])   # Loss function —Esto mide que tan exacto es el modelo durante el entrenamiento. Quiere minimizar esta funcion para dirigir el modelo en la direccion adecuada.
                                                                                                                 # Optimizer — Esto es como el modelo se actualiza basado en el set de datos que ve y la funcion de perdida.
                                                                                                                 # Metrics — Se usan para monitorear los pasos de entrenamiento y de pruebas. El siguiente ejemplo usa accuracy (exactitud) , la fraccion de la imagenes que son correctamente clasificadas.

model.fit(train_images, train_labels, epochs = 5)    # epochs = num de iteraciones para hacer el entrenamiento, por que es muy grande los datos y se div en fracmentos
                                                     # se hace una iteracion y hace la comparacion de las 60000 imagenes, y en cada iteracion aprende algo en particular

# test_loss, test_acc = model.evaluate(test_images, test_labels)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('Accuracy', test_acc)
"""entrenamiento con 1875 y no 60000,
Ese numero no te esta indicando que se entrena con 1875 imagenes. Ese número indica el número de batch con el que se entrena (cada batch consta de 32 imagenes por default). Se está entrenando con el algoritmo de Adam (una variación avanzada de Batch Gradient Descent). 1875*32=60,000 imagenes."""

