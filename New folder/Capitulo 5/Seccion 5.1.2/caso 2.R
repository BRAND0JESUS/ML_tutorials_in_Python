# # # # # # # # # # # # # # # # # # #
# Curva de supervivencia del caso de estudio #2
# capitulo 5
# Seccion 5.1.2 
# # # # # # # # # # # # # # # # # # #

###############################################
## Librerias
###############################################
library(survival)
library(readxl)
library("survminer")
##############################################

# cargar datos
data1 <- read_excel("C:/Users/alexh/Desktop/Anexo A/Codigos de programacion/Capitulo 5/Seccion 5.1.2/data caso estudio 2.xlsx")
View(data1)

# analisis descriptivo y exploratorio grafico
summary(data2)

# Histograma tiempo medio entre fallas MTBF
hist(data1$mtbf, freq = FALSE, breaks = 9, col = "lightblue", border = "black")

# grafica de densidad
plot(density(data1$mtbf))

# validar si la variable modelo es tipo factor 
is.factor(data2$modelo)

# convertir variable a factor
modelo<- as.factor(data2$modelo)
data3<- data.frame(modelo=I(modelo),data2)

View(data3)

is.factor(data3$modelo.1)
data4 <- data3[,-4]
View(data4)
is.factor(data4$modelo)
View(data4)
summary(data4)

# Estimador de kaplan y Meier:

km1<-survfit(Surv(data4$mtbf,data4$falla)~1)

# Valores resumidos obtenidos con el estimador de Kaplan y Meier
print(km1)

# tabla de datos de probabilidad de la curva de supervicencia
summary(km1)

# curva de supervivencia
plot(km1,xlab="Ciclos",ylab="Probabilidad de Supervivencia", main="Estimador de Kaplan y Meier caso de estudio # 2")
plot(km1, fun = "pct", xlab="Tiempo: dias" ,mark="|", conf.int=T,  ylab="Probabilidad de Supervivencia", main="Estimador de Kaplan y Meier caso de estudio # 2")

# Graficos con la libreria survminer:

# vamos a mejorar los graficos con la libreria survminer:
# Gráfico del estimador de Kaplan y Meier:

ggsurvplot(km1, fun = "pct", data = data4, palette= 'blue', ylab="Probabilidad de Supervivencia", xlab="Tiempo en dias (MTBF)", title ="                                     Estimador de Kaplan y Meier caso de estudio #2", legend="none", surv.median.line="hv", font.title= c(20, "bold", "black"),font.x= c(18, "plain", "black"),font.y= c(18, "plain", "black"),font.tickslab= c(14, "plain", "black") )


# Grafica de la historia de ocurrencia de los eventos:

ggsurvplot(km1, data = data4, fun = "event")

# Gráfica de la función de riesgo acumulada:
ggsurvplot(km1, data = data4, fun = "cumhaz")

# Variantes: Estimador de Fleming y Harrington:

fh <- survfit(Surv(data4$mtbf,data4$falla)~1, type ="fleming-harrington", data = data4)
fh.2 <- survfit(Surv(data4$mtbf,data4$falla)~1, type ="fh2", data =data4)
plot(km1,xlab="Dias", conf.int= FALSE,ylab="Supervivencia", main="Gráfico No. 2. Estimadores de la función de supervivencia")
lines(fh, lty=2, conf.int= FALSE)
lines(fh.2, lty=3, conf.int= FALSE)
legend(75,1,legend=c("KM","FH", "FH2"),lty=c(1,2,3))

# el comportamiento de los dos estimadores es similar

# Comparación de curvas de supervivencia por modelo:

Km2 <- survfit(Surv(data4$mtbf,data4$falla)~data4$modelo, data = data4)
plot(Km2,xlab="Dias",ylab="Supervivencia", main="Gráfico No. 2. Estimador de Kaplan y Meier por modelo de maquina",lty=c(1,2),mark.time=FALSE)
legend(75,0.9,legend=c("model1","model2", "model3", "model4" ),lty=c(1,2))

# tablas de datos de probabilidad de la curva de supervicencia por modelo
summary(Km2)

# datos de mediana e intervalo de confianza por modelo
Km2

# Prueba de equivalencia para comparacion de grupos
attach(data4)
survdiff(Surv(data4$mtbf,data4$falla)~modelo) 

# Se cumple el supuesto que los datos por modelo son estadisticamente diferentes

# Gráfico del estimador de Kaplan y Meier por modelo:

ggsurvplot(Km2, fun = "pct", data = data4, palette = c("#E7B800","#2E9FDF", "#CC79A7", "#D55E00" ), ylab="Probabilidad de Supervivencia", xlab="Tiempo en dias (MTBF)", title ="                                    Curvas de supervivencia por modelo",legend.title = "Modelos:",legend.labs = c("modelo1","modelo2", "modelo3", "modelo4"), legend=c(0.7,0.7), surv.median.line="hv", font.legend= c(18, "plain", "black"), font.title= c(20, "bold", "black"),font.x= c(18, "plain", "black"),font.y= c(18, "plain", "black"),font.tickslab= c(14, "plain", "black"))

# tablas de datos del estimado por modelo 
summary(Km2)
print(Km2)
print(Km2)


###############################################

# Modelo semiparametrico de Cox
modelAll2.coxph <- coxph(Surv(data4$mtbf,data4$falla) ~ data4$modelo + data4$edad + data4$errores)
summary(modelAll2.coxph) 


# Obtener los coeficientes del modelo

weib.coef.all <- model.pharm.weib$coef 
weib.coef.all
weib.coef <- weib.coef.all[2:6]
weib.coef

# Obtener los estimados proporcionales de riesgo del modelo weibull
weib.coef.ph <- -weib.coef/model.pharm.weib$scale

# Extraer los coeficientes del modelo de cox para compararlos

coxph.coef <- modelAll2.coxph$coef 
data.frame(weib.coef.ph, coxph.coef)


