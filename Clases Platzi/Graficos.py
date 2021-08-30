from bokeh.plotting import figure, output_file, show


if __name__ == '__main__':
    output_file('Graficado_Simple.html')
    fig = figure()   # type o hep para investigar 
    total_Valv = int(input('Cuantos valores quieres graficar? '))

    x_vals = list(range(total_Valv))
    y_vals = []

    for x in x_vals:
        val = int (input(f'Valor "y" para {x+1} '))
        y_vals.append(val)

    fig.line(x_vals,y_vals, line_width=2)
    show(fig)


#%%                     Con base de datos externo


from bokeh.plotting import figure, output_file, show
import csv

def Leer_CSV(ruta):
    
    fecha = []
    hospitalizados_cdmx = []
    intubados_cdmx = []

    with open(ruta, newline='') as File:
        reader = csv.reader(File)
        data = list(reader)
        
        for i in range(len(data)):
            for j in range(len(data[i])):
                if j == 0:
                    fecha.append(data[i][j])
                elif j == 1:
                    hospitalizados_cdmx.append(int(data[i][j]))
                else:
                    intubados_cdmx.append(int(data[i][j]))
    #print('Fechas: \n', fecha)
    #print('Hospitalizados: \n', hospitalizados_cdmx)
    #print('Intubados: \n', intubados_cdmx)
    Crear_grafico(fecha, hospitalizados_cdmx, intubados_cdmx)

def Crear_grafico(fecha, lista1, lista2):
    output_file('Covid_cdmx.html')
    fig1 = figure(x_range = fecha, plot_height=800, plot_width = 1800, title="Casos de Hospitlalizacion por COVID en CDMX y Edo. Méx.")

    fig1.vbar(x=fecha, top=lista1, width=0.9)
    fig1.y_range.start = 0
    fig1.xaxis.major_label_orientation = 1.2

    show(fig1)

    
if __name__ == "__main__":
    Leer_CSV('personas-hospitalizadas-covid19.csv')