
from matplotlib.colors import cnames
import pandas as pd     # para trabajar con dataset csv
import matplotlib.pyplot as plt     # para visualizar
import seaborn as sns       # visualizaciones adicioanles 
import re
import numpy as np      # trabajar con arreglos sin loops
from sklearn import tree        
from sklearn.model_selection import train_test_split        # permite generar datos de entrenamiento y prueba

test_df = pd.read_csv('titanic-test.csv')
train_df = pd.read_csv('titanic-train.csv')
print(train_df.head())

print(train_df.info())      # ver que datos tenemos, datos nulos

# train_df.Sex.value_counts().plot(kind = 'bar', color = ['b', 'r'])      # Cuatas personas hay en el barco dado si es H o M

train_df[ train_df['Survived'] == 1 ]['Sex'].value_counts().plot(kind='bar', color=['b','r'])       # Cuantas personas sobrevivieron dado si es H o M
plt.title('Distribucion de sobrevivientes')
plt.show()

from sklearn import preprocessing       # se divide el dataset en atributos utiles
label_encoder = preprocessing.LabelEncoder()    

encoder_Sex = label_encoder.fit_transform(train_df['Sex'])      # tratar info numerica categorica
# train_df.head()
train_df['Age'] = train_df['Age'].fillna(train_df["Age"].median())      # fillna = llenar datos nulos en datos para analizar y operar
train_df['Embarked'] = train_df['Embarked'].fillna('S')      # remplazar dartos vacios con S

train_predictors = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)        # Eliminacion de atributos no utiles, axis 1 = columnas

# Encuentra las columnas que contiene información no numérica.
categorical_cols = [cname for cname in train_predictors.columns if      # tomar datos categoricos, cname = corresponde informacion anterior que se rrecore
                        train_predictors[cname].nunique() < 10 and  # si se cumple datos unicos (nunique), object = var categorica
                        train_predictors[cname].dtype == 'object' 
                   ]

print(categorical_cols, '\n')

# Encuentra las columnas con datos numéricos.
numerical_cols = [cname for cname in train_predictors.columns if
                   train_predictors[cname].dtype in ['int64', 'float64']        # ver si es de tipo nummerico, int 64 o flot 64 = entero y flotante
                 ]
print(numerical_cols, '\n')

my_cols = categorical_cols + numerical_cols     # se tiene en una sola var
print(my_cols)

# train_predictors = train_predictors[my_cols]
# # print(train_predictors)

# dummy_encoded_train_predictors = pd.get_dummies(train_predictors)       # dummies = convierte var catergoricas en indicadoras como 0,1,2...
# print(train_df['Pclass'].value_counts())       # Obtener clases para ver como esta dividida la info

# y_target = train_df['Survived'].values      # pediccion
# x_features_one = dummy_encoded_train_predictors.values      # ca racteristicas para trabajar

# x_train, x_validation, y_train, y_validation = train_test_split(x_features_one, y_target, test_size = .25, random_state = 1)        # division de info, random_... 1 = cada ves que se corra el modelo se toma una seccion diferente

# tree_one = tree.DecisionTreeClassifier()    # creación del árbol
# tree_one = tree_one.fit(x_features_one, y_target)   # entrenamiento

# tree_one_accuracy = round(tree_one.score(x_validation, y_validation), 4)    # validación de que tan cercano es la predicción
# print('Accuracy: %0.4f' % (tree_one_accuracy))    # presición de la información

# # archivo png, generacion del camino del arbol para la toma de desiciones

# from io import StringIO     # 
# from IPython.display import Image, display
# import pydotplus    # para generar cada uno de los caminos

# out = StringIO()    # generar archivo
# tree.export_graphviz(tree_one, out_file = out)    # lo que exportamos

# graph = pydotplus.graph_from_dot_data(out.getvalue())   # generar cada uno de las ramas
# graph.write_png('titanic.png')    # creacion de png



