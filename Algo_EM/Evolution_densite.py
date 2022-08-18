#!/usr/bin/env python3

import numpy as np
import numpy.random as npr
import numpy.linalg as lg

from math import sqrt
import matplotlib.pyplot as plt

def phi(y,mu,S):
    return 1/(sqrt(2*np.pi*S))*np.exp(-(y-mu)**2/(2*S))

def g(y,m,theta):
    G=0
    for i in range (m):
        G+=theta['pi'][i]*phi(y,theta['mu'][i],theta['S'][i])
    return G

### SIMULATION DES DONNEES ###


# Simulation données completes
# Pour chaque individus on a sa classe et sa valeur observée
def simu_Z(theta,n):
    p=np.cumsum(theta['pi'])
    Z=[]
    for i in range(n):
        U=npr.random()
        k=np.where(p>U)[0][0]
        Y_k=npr.normal(theta['mu'][k],theta['S'][k]**(1/2))
        Z.append([Y_k,k])
    Z=np.array(Z).reshape((n,2))
    return Z


### INITIALISATION DES PARAMETRES A ESTIME ###

# Methode initialisation 1 : mu[d]=0 et S[d]=1 pour tout d dans {1,...,m}
def init_1(m):
    theta={}
    theta['pi']=1/m*np.ones(m)
    theta['mu']=np.zeros(m)
    theta['S']=np.ones(m)
    return theta

#Mehtode initialisation 2:
    #On tri Y puis on le divise en m groupe
    # initialise mu[d] avec la moyenne du groupe d
def init_2(Y,m):
    Y_s=np.copy(Y)
    Y_s.sort()
    theta={}
    theta['pi']=1/m*np.ones(m)
    theta['mu']=np.array([np.mean(Y_s[m*i:m*(i+1)]) for i in range(m)])
    theta['S']=np.ones(m)
    return theta

### ALGORITHME EM ###

# matrice M de taille n*m
# coeff M[i,j]=pi[j]*phi(Y[i],mu[j],S[j])/sum(pi[k]*phi(Y[i],mu[k],S[k]))
# Probabilité que Y[i] appartienne au groupe j
def E_step(Y,m,theta):
    n=len(Y)
    M=np.zeros((n,m))
    for i in range(n):
        for k in range (m):
            M[i,k]=theta['pi'][k]*phi(Y[i],theta['mu'][k],theta['S'][k])
        M[i,:]=M[i,:]/sum(M[i,:])
    return M

# calcul des estimations de pi,mu et S
def M_step(Y,m,theta):
    n=len(Y)
    theta_p={}
    pi=[]
    mu=[]
    S=[]
    M=E_step(Y,m,theta)
    for i in range (m):
        Sum=sum(M[:,i])
        pi.append(1/n*Sum)
        mu.append(np.dot(M[:,i],Y)/Sum)
        S.append(np.dot(M[:,i],(Y-mu[i])**2)/Sum)
    theta_p['pi']=np.array(pi)
    theta_p['mu']=np.array(mu)
    theta_p['S']=np.array(S)
    return theta_p

# on renormalise la norme inf de u-v par la norme de u
# pour prendre ne compte les differents ordre de grandeurs entre pi, mu et theta
def Norm(u,v):
    return lg.norm(u-v,np.inf)/lg.norm(v,np.inf)

def Algo_EM(Y,m,eps,init):
    theta=init(Y,m)
    tol=eps+1
    nb_iter=0
    while tol>eps: # Seuil de tolerance fixé
       theta_p=M_step(Y,m,theta)
       tol=Norm(theta_p['pi'],theta['pi'])+Norm(theta_p['mu'],theta['mu'])+Norm(theta_p['S'],theta['S'])
       theta=theta_p
       nb_iter+=1
    print(f"Nombre d'iteration : {nb_iter}")
    return theta


### CLASSIFICATION ###

# Calcul de la matrice du E_step avec l'approximation de theta obtenu par l'algo
# Pour chaque ligne i, on cherche l'indice de la colonne j ou la valeur est la plus grande
# La valeur j correspond au groupe auquel l'individu i appartient avec la plus grande proba
# Return un vecteur où la i-eme coordonnée correspond au groupe d'appartenance estime pour l'individu i
def groupe(Y,m,theta):
    n=len(Y)
    M=E_step(Y,m,theta)
    Z=np.array([np.where(M[i,:]==max(M[i,:]))[0][0] for i in range (n)])
    return Z

def Matrice_confusion(Z,Z_estim,m):
    n=len(Z)
    M=np.zeros((m,m))
    for i in range(n):
        M[int(Z[i,1]),Z_estim[i]]+=1
    taux_err=(n-np.sum(np.diag(M)))/n*100
    return M,taux_err

### REPRESENTATION GRAPHIQUE ###

color=('b','g','r','c','m','y','k','w')
hatch=("/","\\","","+",":","-")

def graph_reel(Z,theta,m):
    Y=Z[:,0] 
    for i in range(m):
        G=np.where(Z[:,1]==i) # Individus estimé comme appartenant au groupe i
        hist,bins=np.histogram(Y[G]) # Histogramme du groupe i estimé
        I=bins[1]-bins[0]
        A=I*np.sum(hist) # Aire de l'histogramme, utilisé pour avoir un histogramme en densité
        plt.hist(Y[G],bins=bins,weights=theta['pi'][i]/(A)*np.ones(len(Y[G])),color='w',hatch=hatch[i],edgecolor='k',label=f"G{i+1}")
    
def graph(Z,Z_estim,theta,theta_estim,m,nb_iter):
    plt.clf()
    Y=Z[:,0]
    graph=graph_reel(Z,theta,m)
    x=np.linspace(min(Y),max(Y),200)
    for i in range(m):
        plt.plot(x,theta_estim['pi'][i]*phi(x,theta_estim['mu'][i],theta_estim['S'][i]),color='k',linestyle="--")
    plt.plot(x,g(x,m,theta_estim),'r',label='g',linewidth=2)
    for i in range(m):
        graph
        G=np.where(Z_estim==i) # Individus estimé comme appartenant au groupe i
        if len(G[0])!=0:
            hist,bins=np.histogram(Y[G]) # Histogramme du groupe i estimé
            I=bins[1]-bins[0]
            A=I*np.sum(hist) # Aire de l'histogramme, utilisé pour avoir un histogramme en densité
            plt.hist(Y[G],bins=bins,weights=theta_estim['pi'][i]/(A)*np.ones(len(Y[G])),color=color[i],alpha=.4,label=f"Groupe {i+1} estime") 
    plt.title(f"Estimation apres {nb_iter} iteration")
    plt.legend(loc="upper left")
    plt.pause(0.2)
    
def graph_err(Z,Z_estim,theta,m):
    bon=np.where(Z_estim==Z[:,1])# Individus bien classés
    err=np.where(Z_estim!=Z[:,1]) # Individus mal classés
    plt.scatter(Z[:,0][bon],Z[:,1][bon]+1,c=Z[:,1][bon]) 
    plt.scatter(Z[:,0][err],Z_estim[err]+1,c=Z[:,1][err])
    plt.xlabel("Valeur observé")
    plt.ylabel("groupe")
    plt.yticks(range(1,m+1))
    plt.title("Classification des individus")

def affiche_parametre(theta):
    print(f"pi : {theta['pi']}\nmu : {theta['mu']}\nsigma^2 : {theta['S']}\n")

def evolution_algo(Z,m,theta,eps,init):
    Y=Z[:,0]
    if init==init_1:
        theta_estim=init(m)
    else :
        theta_estim=init(Y,m)
    tol=eps+1
    nb_iter=0
    while tol>eps and nb_iter<=300: # Seuil de tolerance fixé
       theta_p=M_step(Y,m,theta_estim)
       tol=Norm(theta_p['pi'],theta_estim['pi'])+Norm(theta_p['mu'],theta_estim['mu'])+Norm(theta_p['S'],theta_estim['S'])
       theta_estim=theta_p
       Z_estim=groupe(Y,m,theta_estim)
       if nb_iter%10==0 :
           graph(Z,Z_estim,theta,theta_estim,m,nb_iter)
       nb_iter+=1

    return Z_estim,theta_estim

if __name__=='__main__':
    #Initialisation de Z
    print("Exemple : ")
    x=input()
    init=init_2
    
    if x=='1' or x=='0':
        pi=[1/6,1/2,1/3]
        mu=[-5,1,7]
        S=[1,4,1]
        if x=='0':
            init=init_1
        else:
            init=init_2
        
    elif x=='2':
        pi=[1/3,1/2,1/6]
        mu=[-3,1,5]
        S=[1,4,1]
        init=init_2
    
    elif x=='3':
        pi=[1/3,1/2,1/6]
        mu=[-3,0,2]
        S=[1,2,1]
        init=init_2
          
    n=1000
    tol=10**-3
    m=3
    theta={'pi':np.array(pi), 'mu':np.array(mu),'S':np.array(S)}
    Z=simu_Z(theta,n)
    
    
    Z_estim,theta_estim=evolution_algo(Z,m,theta,tol,init)
    print("Parametres reels: ")
    affiche_parametre(theta)
    print("Parametres estimes: ")
    affiche_parametre(theta_estim)
    M,taux_err=Matrice_confusion(Z,Z_estim,m)
    print (f"Matrice de confusion:\n{M} \ntaux_d'erreur d'erreur : {taux_err}%")
    plt.show()