#%%

import pandas as pd
import numpy as np
import seaborn as sns
import os

os.chdir('/Users/Brando/Desktop/VS Code')
os.getcwd()
# print(os.listdir('.'))      # que elementos estan contenidos
df = pd.read_csv('train.csv')

df.columns
# descripcion para definir la calidad de var
df.isna().sum(axis=0)/len(df)   # conteo de las var NA

df = df.drop(['Cabin'], axis = 1)   #Se elimina de la base las col que tengas var incompletas

df = df[df.Age.notna()]     # Asiganar a df unicamente aquellas entradas que no tienen val perdidos sobre la edad
# var discretas

var_discr = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
# iterar para entender el contenido de cada una de ellas
for i in var_discr:
    df[i].value_counts()  # cuales son ordinales y categoricas
# variables categoricas
var_cat = ['Pclass', 'Sex', 'Embarked']  # No poseen sentido ordinal

# one hot encoding = proceso en q la var catgorica se expresa como 1 o 0
def one_hot():
    for i in var_cat:
        categorias = df[i].value_counts().index       # index para tener solo los nombres de las categorias
        # print(categorias)

        # poner 0 u 1 si toma o no el val de la categoria
        for k in categorias:
            name = 'is-' + str(i) + '-' + str(k)
            print(name)
            # asignarle una var binaria
            df[name] = np.where(df[i] == k, 1, 0) 

    # eliminar las var originales del dataframe para no tener var repetidas o colineales
        df = df.drop([i], axis=1)

# identificacion de outliers

df.columns
   


    



# y = df['Survived']
# X = df[['Age', 'SibSp','Parch',  'Fare', 'is-Pclass-1', 'is-Pclass-2', 'is-Sex-male','is-Embarked-S', 'is-Embarked-C', 'family_size']]

# %%