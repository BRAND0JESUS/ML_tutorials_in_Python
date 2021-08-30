import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))

# prediccion de un nuevo dato

nuevo_dato1 = np.array([[0.199713, 4.323565]])

# Ahora encontramos la etiqueta o clase de este nuevo individuo
label_pred_load21 = model.predict(nuevo_dato1)

#etiqueta_nueva_dato1 = 
print("Etiqueta nuevo dato: ", label_pred_load21)