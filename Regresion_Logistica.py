import numpy as np
from numpy import diag
import pandas as pd     
from sklearn import metrics     # para la validacion del modelo
from sklearn.model_selection import train_test_split    # para el entrenamiento y division de la informacion
from sklearn.linear_model import LogisticRegression     # modelo regresion logistica
import matplotlib.pyplot as plt         # para visualizavion
import seaborn as sns


diabetes = pd.read_csv('diabetes.csv')
# print(diabetes.head(5))      # colum "Outcome =1 o 0" son de pacientes que previamente fueron diagnosticados con o sin diabetes
# print(diabetes.shape)

feature_cols = ['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'] # atributos
x = diabetes[feature_cols]  # se mada a llamar target, atributos para aprender
y = diabetes.Outcome    # para saber si tiene debetes o no

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=0)

logreg = LogisticRegression()   # llamda del metodo
logreg.fit(X_train, Y_train)    # Entrenamiento
y_pred = logreg.predict(X_test)    # X_test =  que queremos predecir

cfn_matrix = metrics.confusion_matrix(Y_test, y_pred)      #matriz de condusion (informacion para predecir, prediccion)
# print(cfn_matrix)       # solo me imprime arreglo en 2 dimensiones
class_names = [0 , 1]        # 1 = diabetico, 0 no diabetico
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))  #np.arrange = muestra info en la grafica; 
plt.xticks(tick_marks, class_names)     # para mosgtrar en las cabezceras y laterales si tiene diabetes o no
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cfn_matrix), annot = True, cmap="Blues_r", fmt="g")        # grafia heatmat se cra con pandas a travez de dataframe; muestra matriz de confusion, cmap =muestra en azul intenso los que tienen val superiores
ax.xaxis.set_label_position("top")      # colocar informacion en la parte superior
plt.tight_layout()      # generacion del heatmat
plt.title("Matriz de confusion", y=1.1)  # y = posicion
plt.ylabel("Etiqueta Actual")
plt.xlabel("Etiqueta de prediccion")
plt.show()

print("exactitud", metrics.accuracy_score(Y_test, y_pred))