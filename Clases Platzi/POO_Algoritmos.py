# %%                Programación Orientada a Objetos

class Coordenada:
    def __init__(self, x, y):  #hasta aquí lo que esta después de self, solo son parametros.
        self.x = x
        self.y = y  
		# Aquí estamos inicializando las variables de instancia

    def distancia(self, otra_coordenada):  #el parametro otra_coordenada es una instancia, que hara uso del molde en la primer clase de inicializacion, la cual usaremos despues, tomar atencion.
        x_diff = (self.x - otra_coordenada.x)**2
        y_diff = (self.y - otra_coordenada.y)**2

        return (x_diff + y_diff)**0.5
	#estos solo son una representación matemática de lo que hará el programa (lo explicare después)

    #mi caso yo prefiero usar comillas dobles, depende de ustedes.
if __name__ == '__main__': 
    coord_1 = Coordenada(3, 30)
    coord_2 = Coordenada(4, 8)
	#estas dos expresiones son instancias que usan el molde que es el primer método de inicializacion.
    
    # print (coord_1.distancia(coord_2))
    #"coord_1" hace uso de la primer instancia, mientras que coord_2 al estar dentro del metodo distancia, ocupa el lugar de otra_coordenada. (entender esta parte es muy importante)

    # Determina si la corrdenada es instancia
    print (isinstance(coord_1, Coordenada))  
# %%             DECOMPOSICION

print("\n\n")

class Automovil:
    def  __init__(self, modelo, marca, color):
        self.modelo = modelo
        self.marca = marca
        self.color = color 
        self._estando = "en_reposo"
        self._motor = Motor(cilindros = 4) #esto es una  variable privada, por eso se empieza con _
        self._seguridad = AirBag ()

    def acelerar (self, tipo ):
        if tipo == "rapida":
            self._motor.inyectaGasolina(10)
            self._motor.temperatura(12)
        else: 
            self._motor.inyectaGasolina(3)
            self._motor.temperatura(7)
        self._estado = "EnMovimiento"

    def desAcelerar (self, tipo ):

        if tipo == "brusca":
            self._seguridad.activar()
        else:
            pass
            
class Motor:
    def __init__(self, cilindros, tipo = 'gasolina', nivelGasolina = 46000, temperatura = 0 ): #tipo es un parametro ya definido, se le llama default keyword, se entiende comoo un parametro por defecto.
        self.cilindros = cilindros
        self.tipo = tipo 
        self.nivelGasolina = nivelGasolina
        self.estadoTemperatura = temperatura

    def inyectaGasolina(self, cantidadGasolina):
        self.nivelGasolina -= cantidadGasolina
    
    def temperatura (self, grados ):
        self.estadoTemperatura += grados

    def informacion (self): #Esta funcion es temporal, solo para revisar que todo esta funcionanndo :v xd 
        print("\n")
        print(f"nivelGasolina = {self.nivelGasolina} y temperatura = {self.estadoTemperatura}")
        print("\n")

class AirBag:

    def __init__(self, estado = "optimo"):
        self.estado = estado

    def activar (self):
        print("SISTEMA DE SEGURAD DE CHOQUES ACTIVADO")
        self.estado = "inhalitado"

if __name__ == "__main__":

    car1 = Automovil("AAFF","toyota", "rojo")
    car1._motor.informacion() 
    car1.acelerar("lenta")
    car1._motor.informacion()
    car1.desAcelerar("brusca")
    
print("\n\n")

# %%
class petroleo:
    def __init__(self, Pr, Pwf):
        #self.Q = Q
        self.Pr = Pr
        self.Pwf = Pwf
        self.API= API(densidad_rel_oil = 8.2)  # todas las variables privadas requieren ser definidas posteriomente como funcion

          
class API:
    def __init__(self, densidad_rel_oil):
        self.densidad_rel_oil = densidad_rel_oil
    
    def Api(self):
        cal_API = 141.5/densidad_rel_oil-131.5
        print (cal_API)

if __name__ == "__main__":
    Datos = petroleo(5000, 2000)


# %%






# %%
