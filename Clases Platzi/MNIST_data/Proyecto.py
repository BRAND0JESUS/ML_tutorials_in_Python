#%%

# Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NumPy (Numeric Python) le da a Python capacidades de cálculo similares a los de otros software como MATLAB.
# SciPy (Scientific Python) es una librería fundamental para computación científica.

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix  # para la evaluacion del modelo
from sklearn.ensemble import RandomForestClassifier

# funcion para evaluarel calsificador
def evaluate_classifier (clf, test_data, test_lebels):  #clasificador, datos de prueba, etiquetas de datos de rueba
   pred = clf.predict(test_data)
   MC = confusion_matrix(test_lebels, pred) # definir la Matriz de confusion (valores reales del data set, valores de prediccion)
   return MC


# importar del módulo mnist from mnist import **MNIST**
from mnist import MNIST
#importar el dataset mndata = MNIST('MNIST_data')
mndata = MNIST(r'\Users\Brando\Desktop\VS Code\MNIST_data')
# cargas los daros (features -> datos y labels -> etiquetas) datos, labels = mndata.load_training()
datos, labels = mndata.load_training()


for i in range(25):     # mostrar imagenes en 5 fil y 5 col = 25
   plt.subplot(5,5,i+1)
    
   d_image = datos[i]      # imagen que viene de los datos
   d_image = np.array(d_image, dtype='float')
    
   pixels = d_image.reshape((28, 28))     # forma de la imagen mostrada en una matriz
    
   plt.imshow(pixels, cmap='gray')
   plt.title(labels[i])
   plt.axis('off')
plt.show()

# todos los datos de training... deben ser divididos
# Datos de Entremamiento 70%  (features, labels)
# Datos de Testing  30%   (features y labels)

# sklearn.model_selection.train_test_split(*arrays, **options)

# train_data, test_data, train_labels, test_labels
# 70% del total de los datos serán para el training set

train_data, test_data, train_labels, test_labels = train_test_split(datos, labels, test_size=0.3, random_state=42)

# Modelos ML

# Definior el calsificador de arbol de desiciones

clf_dt = DecisionTreeClassifier()

# Entrenamiento del clasificador

clf_dt.fit(train_data, train_labels)

# Evaluacion del calsificador con todo el training y testing set
MC = evaluate_classifier(clf_dt, test_data,test_labels)  # (calsificador de desition tree)
# los valores de la diagonal de MC son los valores bine clasificados
print (MC)
# Score
score = MC.diagonal().sum()*100./MC.sum()
print (score)

##########################################################################
#                             Random Forest
# n_estimators numero de arboles que se desa en la red
# min_sample_aplit numero min de filas que utiliza
# min_sample_leaf numero min de samples que tenemos al final del arbol

clf_rf = RandomForestClassifier(n_estimators=150, min_samples_split=2)

# entrenar le calsificador RF
clf_rf.fit(train_data, train_labels)

# Vamos a evaluar el clasificador con el tarining y testing del set
# Evaluar random forest

MC_rf = evaluate_classifier(clf_rf, test_data, test_labels) # con el calsificador y set de prueba genera nuevas resultados y estos son comparados con las etiquetas
print (MC_rf)

Score_rf = MC_rf.diagonal().sum()*100./MC_rf.sum()  
print(Score_rf)


###############################################################################
#                          SET  DE DATOS DE PRUEBA
################################################################################

# Carga de datos de prueba de MNIST
test_data, test_labels = mndata.load_testing()
# Aplicamos el calsificador al todo el data set de evaluacion y obtrnemos el acurracy
# p <algoritmo selecionado>.predict(evaluate_data)
predicted = clf_rf.predict(test_data)

# evaluamos los resultos con la mastatriz de confusion para el data set deevaluacion
# MC.Evaluacion
MC_proyect = evaluate_classifier(clf_rf, test_data, np.array(test_labels))
print (MC_proyect)
Score_proyect = MC_proyect.diagonal().sum()*100./MC_rf.sum()  
print (Score_proyect)

# see digit extra
digit = test_data[7]

digit_float = np.array(digit, dtype="float")
pixels = digit_float.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
# predecir que numero sera
print(clf_rf.predict([test_data[7]]))



# %%
