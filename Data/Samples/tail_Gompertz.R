library(ismev)  #Donde viene el método de exceso sobre un umbral
library(fitdistrplus) #El conjunto de datos
library(mgcv)
library(nlme)
library(MASS)
library(survival)

# Calculo de la proyeccion 
set.seed(1)
#setwd("~/Escritorio/Proyecto SNI 2022/Generalized Gompertz/Data/Samples")
file1='NUEVO LEÓN_kde_sample_omicron_trun.csv'
data<- read.csv(file = file1)
#data<-subset(data, tiempo>umbral)

tiempo=unlist(data['tiempo'])
hist(tiempo,type="l") # Grafica para valorar independencia

tiempo=sort(tiempo,decreasing=FALSE)
#tiempo=tiempo[tiempo<40]
umbral=20
tiempo=tiempo[tiempo>umbral]

E=function(u,y){
  z=y[y>u]
  zz=length(z)
  e=(1/zz)*sum(y[y>u]-u)
  return(e)
}
N=length(tiempo)
EE=seq(0.0,1.0,by=1.0/(N-1))
for(i in 1:N){
  EE[i]=E(tiempo[i],tiempo)[1]
}

plot(tiempo,EE,type="l",col="red",xlim=c(umbral,umbral+120)) #Funcion Media de Excesos
plot(tiempo,EE/tiempo,type="l",col="blue",xlim=c(umbral,umbral+120)) #Identificar un posible dominio de atracción.

umbral_excess=21 # Nuevo Leon
u_proportion=length(tiempo[tiempo>umbral_excess])/N #Candidato para estimar los parámetros de la DGVE
u_size=length(tiempo[tiempo>umbral_excess])

max(tiempo)
gpd.fitrange(tiempo,umbral_excess-1,umbral_excess+1) #Calcular los estimadores de la DGVE utilizando Pickands-Haan
#En lo anterior buscamos estabilidad del estimador

estimadores=gpd.fit(tiempo,umbral_excess) #Para estimar dado el umbral
gpd.diag(estimadores) #La graficas de diagnostico

