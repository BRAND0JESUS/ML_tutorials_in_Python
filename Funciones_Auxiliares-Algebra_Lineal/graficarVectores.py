def graficarVectores (vecs, cols, alpha = 1):        # alpha = nivel de transparencia
    plt.axvline(x = 0, color = 'grey', zorder = 0)      # eje hororizontal (cruce en X=0, orden z = 0)
    plt.axhline(y = 0, color = 'grey', zorder = 0)

    for i in range (len(vecs)):        # para cada uno de los vectores
        x = np.concatenate([[0, 0], vecs[i]])        # x es la concatenacion con el 0,0 que es el p de origen y se le concatena el vector[i]
        plt.quiver([x[0]],      # grafica las coordenadas
                    [x[1]],
                    [x[2]],
                    [x[3]],
                    angles = 'xy', scale_units = 'xy',      # angles = estan expresados en xy, scale... = debe ser xy
                    scale = 1,
                    alpha = alpha)