# %%                  Clase 8

#               Strings ->  Cadenas 

# Concatenar = union de varios carateres
a = 'Hip '*3 + '' + 'Hurra'
b = f'{"Hip " * 3} Hurra'  # f = cadena de forma

# len = longitud de la cadena
# Indexing = acceder a cada uno de los elementos de la cadena
# slicing = dividir la cadena, sub cadenas o sub string

my_stg = 'Brando'
len (my_stg)            # = 6
my_stg [0]              # = B
my_stg.count('d')    # = 1  cuantas veces se repite la letra en la cadena

# my_string [comienzo:final:pasos]
my_stg[2:]        # = ando
my_stg[:3]        # = Bra
my_stg[:-1]       # = Brand
my_stg[::2]       # = Bad
my_stg.replace('o','i')         # = Brandi
my_stg.split('n')               # = ['Bra', 'do']

'Yo me llamo ' + my_stg         # = Yo me llamo Brando
f'Yo me llamo {my_stg}'         # = Yo me llamo Brando
f'Yo me llamo {my_stg}, ' *3        # = Yo me llamo Brando, Yo me llamo Brando, Yo me llamo Brando,


#       Entradas -> Imputs

# imput = solo ingresa cadenas
Nombre = input ('Cual es tu nombre: ')
print(f'Tu nombre es {Nombre}')
# Si se desa ingresar un numero se requiere encapsular con int o float, float, bool
Numero = int(input('Donos un Numero: '))
c=type(Numero)
print(c)


# %%

# %%                  Clase 9

#               Ramificaciones

num_1 = int(input('escriba un numero entero: '))
num_2 = int(input('escriba un numero entero: '))

if num_1 > num_2:
    print('El primer numero es mayor que el segundo numero')
elif num_1 < num_2:
    print('El primer numero es menor que el segundo numero')
else:
    print('Los numeros son iguales')
    

# %%
import torch
# %%
