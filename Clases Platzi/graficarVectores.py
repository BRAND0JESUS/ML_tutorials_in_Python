#%%

import numpy as np
import matplotlib.pyplot as plt

# vecs: vectores.
# cols: colores.
# alpha: valor de transparencia.
def graficarVectores(vecs, cols, alpha=1):
    plt.figure()
    plt.axvline(x=0, color="grey", zorder=0)
    plt.axhline(y=0, color="grey", zorder=0)
    
    for i in range(len(vecs)):
        # El origen de los vectores inicia en el punto (0,0)
        x = np.concatenate([[0,0], vecs[i]])
        plt.quiver([x[0]],
                  [x[1]],
                  [x[2]],
                  [x[3]],
                  angles='xy', scale_units='xy', scale=1, 
                  color=cols[i], alpha=alpha)


# %%
