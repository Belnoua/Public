library(tseries)
library(ggplot2)
library(patchwork)
library(cowplot)

# Fonction permettant une analyse visuelle des résidus d'une modélisation par quelques graphiques adaptés
ggcheckupRes = function(Res,col,i){
  
  n=length(Res)
  mu=mean(Res)
  Sn=sd(Res)
  
  data=data.frame(Res)
  
  # Série des résidus
  a=ggplot(data,aes(1:n,Res))+geom_line(color=col)+ggtitle(paste("Graphiques des résidus du modèle",i))+theme(axis.title.x = element_blank(),axis.title.y = element_text(size=9),plot.title = element_text(size = 12,hjust = 0.5))
  
  # ACF/PACF
  b=ggAcf(Res,ylim=c(-1,1))+theme(axis.title.x = element_blank(),axis.title.y = element_text(size=9),plot.title= element_blank())
  c=ggPacf(Res,ylim=c(-1,1))+theme(axis.title.x = element_blank(),axis.title.y = element_text(size=9),plot.title= element_blank())
  
  data2=data.frame(Res[1:n-1])
  # Nuage de points avec décalage de 1 dans le temps
  d=ggplot(data2,aes(x=Res[1:(n-1)],y=Res[2:n]))+geom_point(size = 1,color=col)+xlab("Res[i-1]") + ylab("Res[i]")+ggtitle("Autocorrélation d'ordre 1")+
    theme(axis.title.x = element_text(size=9),axis.title.y = element_text(size=9),plot.title = element_text(size = 9))
  
  
  # Histogramme
  e=ggplot(data, aes( x=Res) )+ geom_histogram(aes(y=..density..),color="black", fill=col)+  geom_function(fun = dnorm,colour = "black", args = list(mean = mu, sd = Sn),size=1)+
    theme(axis.title.x = element_blank(),axis.title.y = element_text(size=9),plot.title = element_text(size = 9))+ggtitle("Histogramme")

  # QQ plots
  f=ggplot(data, aes(sample=Res),color=col)+ stat_qq(size=1,color=col) + stat_qq_line(color="black")+
    theme(axis.title.x = element_blank(),axis.title.y = element_blank(),  plot.title = element_text(size = 9))+ggtitle("QQ plot") 

  # Nuage de points standardisé
  Z=(Res-mu)/Sn
  g=ggplot(data,aes(1:n,Z))+geom_point(size=1,color=col)+geom_hline(yintercept = c(-2,2), linetype="dashed", color = "black",size=1)+
    theme(axis.title.x = element_blank(),axis.title.y = element_blank(),plot.title = element_text(size = 9))+ggtitle("Série centrée réduite")
  
  return(a/(b+c+d)/(e+f+g))
  
}

res=rnorm(100)
G=ggcheckupRes(res,"blue",1)
G
