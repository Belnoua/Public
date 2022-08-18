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

def Algo_EM(Y,m,eps):
    theta=init_2(Y,m)
    tol=eps+1
    nb_iter=0
    while tol>eps: # Seuil de tolerance fixé
       theta_p=M_step(Y,m,theta)
       tol=Norm(theta_p['pi'],theta['pi'])+Norm(theta_p['mu'],theta['mu'])+Norm(theta_p['S'],theta['S'])
       theta=theta_p
       nb_iter+=1
    return theta


### REPRESENTATION GRAPHIQUE ###

color=('b','g','r','c','m','y','k','w')

def graph_obs(Y,theta,m):
    plt.hist(Y,bins=20,density=True,color="lightgrey",edgecolor='k',label="Densite observee")

def graph(Y,theta,theta_estim,m,nb_iter): 
    plt.clf()
    x=np.linspace(min(Y),max(Y),300)
    y_max=graph_obs(Y,theta,m)
    plt.ylim=(0,y_max)
    for i in range(m):
        plt.plot(x,theta_estim['pi'][i]*phi(x,theta_estim['mu'][i],theta_estim['S'][i]),color=color[i],linestyle="--",label=fr'$\phi_{i+1}$')
    plt.plot(x,g(x,m,theta_estim),'r',label='g',linewidth=2)
    plt.title(f"Estimation apres {nb_iter} iteration")
    plt.legend(loc='upper left')
    plt.pause(0.2)


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
       if nb_iter%10==0:
           graph(Y,theta,theta_estim,m,nb_iter)
       nb_iter+=1
    graph(Y,theta,theta_estim,m,nb_iter)
    return theta_estim

if __name__=='__main__':
    #Initialisation de Z
    print("Exemple : ")
    x=input()
    
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
    
    
    theta_estim=evolution_algo(Z,m,theta,tol,init)
    print("Parametres reels: ")
    affiche_parametre(theta)
    print("Parametres estimes: ")
    affiche_parametre(theta_estim)
    plt.show()