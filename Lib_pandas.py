from numpy import inf
import pandas as pd

series = pd.Series([5,10,15,20,25])
print (series)      # se crea a los diccionarios o excel con columnas e indices

print(type(series))

print(series[3])

cad = pd.Series(['H', 'o', 'l', 'a'])       
print(cad)

# DATA FRAME, Se usa en ML y analisis de datos

lst = ['Hola', 'mundo', 'robótico']
df = pd.DataFrame(lst)      # arreglo lst transformado a dataframe
print(df)

data = { 'Nombre': ['Juan', 'Ana', 'Jose', 'Arturo' ],  
        'Edad' : [25, 18, 23, 27], 
        'Pais': ['MX', 'CO', 'BR', 'MX'] }
df = pd.DataFrame(data)
print(df)       # Imprime todo df
print(df[['Nombre', 'Pais']])       # imprime col definidas

#trabajo con archivos csv

import pandas as pd
data = pd.read_csv('canciones-2018.csv')
print(data.head(5))

artista = data.artists          # trae columna del df
print (artista[5])              # imprime dato en particular de la coolum artistsa
info = data.iloc[15]            # describe la iformation en la posicion 15
print(info)
print(data.info())              # esumen de las variables y el tipo de dato en cada una de ellas.

print(data.tail())                # infomacion que esta al final

print(data.shape)               # sabe la amplitud del archivo fil x col           
print(data.columns)             # describe las columnas que se posee

print(data['tempo'].describe())         # describe, elementos, media, mediana, var....

data.sort_index(axis=0, ascending=False)       # Odenar (axis=desde la posicion 0, sea de manera descedete)

subset = data[['name','tempo','duration_ms']]
subset.sort_values(by='tempo',axis=0 , ascending=True)          # ordenar definiendo una columna


"""
al cargar el archivo usando pd.read_csv. Hay algunos parametros que son de interés por si cargan otro tipo de archivos.

delimiter = tipo de separador de los datos en el csv
decimal = notacion decimal de los datos ( , o .)
encoding = puede que se tengan archivos cuya codificacion no se utf-8 y se necesite cambiar a por ejemplo latin1"""


# df = pd.read_csv('nombre.csv', delimiter = ',' , decimal = '.' , encoding= 'utf-8')



# # Ordenando según la clase
# data[['name', 'artists', 'tempo', 'duration_ms']].sort_index(axis=0, level='name', ascending=True)

# #Ordenando según columna 'tempo'
# data[['name', 'artists', 'tempo', 'duration_ms']].sort_values('tempo')
