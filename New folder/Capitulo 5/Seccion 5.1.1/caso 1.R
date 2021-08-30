# # # # # # # # # # # # # # # # # # #
# Curva de supervivencia del caso de estudio #1
# capitulo 5
# Seccion 5.1.1 
# # # # # # # # # # # # # # # # # # #

###############################################
## Librerias
###############################################
library(survival)
library(readxl)
library("survminer")

##############################################

# cargar datos
data1 <- read_excel("C:/Users/alexh/Desktop/Anexo A/Codigos de programacion/Capitulo 5/Seccion 5.1.1/data caso estudio 1.xlsx")
View(data1)

# analisis descriptivo y exploratorio grafico
summary(data1)

# Histograma ciclos
hist(data1$ciclos, freq = FALSE, breaks = 9, col = "lightblue", border = "black")

# grafica de densidad
plot(density(data1$ciclos))

###########################################
# analisis de supervicencia

# Estimador de kaplan y Meier:

km1<-survfit(Surv(data1$ciclos,data1$falla)~1)

# Valores resumidos obtenidos con el estimador de Kaplan y Meier
print(km1)

# tabla de datos de probabilidad de la curva de supervicencia
summary(km1)

# curva de supervivencia
plot(km1,xlab="Ciclos",ylab="Probabilidad de Supervivencia", main="Estimador de Kaplan y Meier caso de estudio # 1")

# curva de supervivencia marcando puntos en observaciones 
plot(km1, xlab="Ciclos" ,mark="|", conf.int=T,  ylab="Probabilidad de Supervivencia", main="Estimador de Kaplan y Meier caso de estudio # 1")

# vamos a mejorar los graficos con la libreria survminer:

# Gráfico del estimador de Kaplan y Meier:

ggsurvplot(km1, data = data1, palette= 'blue', ylab="Probabilidad de Supervivencia", xlab="Tiempo en ciclos", title ="                                     Estimador de Kaplan y Meier caso de estudio #1", legend="none", surv.median.line="hv", font.title= c(20, "bold", "black"),font.x= c(18, "plain", "black"),font.y= c(18, "plain", "black"),font.tickslab= c(14, "plain", "black"),fun = "pct" )


# Graficamos la historia de ocurrencia de los eventos:

ggsurvplot(km1, data = data1, fun = "event", palette= 'darkorange', legend="none", font.title= c(28, "bold", "black"),font.x= c(24, "plain", "black"),font.y= c(24, "plain", "black"),font.tickslab= c(22, "plain", "black"),ylab="Eventos acumulados", xlab="Tiempo en ciclos", title ="              Historia ocurrencia de eventos caso de estudio #1",)

# Graficamos la función de riesgo acumulada:

ggsurvplot(km1, data = data1, fun = "cumhaz", legend="none", font.title= c(28, "bold", "black"),font.x= c(24, "plain", "black"),font.y= c(24, "plain", "black"),font.tickslab= c(22, "plain", "black"),ylab="Riesgo acumulado", xlab="Tiempo en ciclos", title ="              Función de riesgo acumulada caso de estudio #1",)

##########################################################

# prueba para validar si los datos se ajustan a una distribucion Weibull

# extraer los estimadores de supervicencia y tiempo

survest<- km1$surv
survest

surtime<- km1$time
surtime

loglogsurvest<- log(-log(survest))
logsurvtime<- log(surtime)

loglogsurvest<- loglogsurvest[35:126]

logsurvtime<- logsurvtime[35:126]

# grafica
plot(loglogsurvest~logsurvtime)
result.lm<- lm(loglogsurvest ~ logsurvtime)
abline(result.lm)

result.lm
# conclusion estos datos no se ajusta a una distribucion de Weibull


