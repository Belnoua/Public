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
    while tol>eps: # Seuil de tolerance fixé
       theta_p=M_step(Y,m,theta)
       tol=Norm(theta_p['pi'],theta['pi'])+Norm(theta_p['mu'],theta['mu'])+Norm(theta_p['S'],theta['S'])
       theta=theta_p
    return theta

### BIC ###

def BIC_m(Y,m):
    theta=Algo_EM(Y,m,10**-3)
    L=0
    for y in Y :
        l=0 
        for i in range(m):
            l+=theta['pi'][i]*phi(y,theta['mu'][i],theta['S'][i])
        L+=np.log(l)
    Bic=-2*L+(3*m-1)*np.log(len(Y))
    print(f'BIC m={m} : {Bic}')
    return Bic,theta

# Determination du nombre de groupe minimisant le BIC
# Renvoie les differentes valeurs du BIC calculer sur l'interval voulu
# Ainsi que le m optimal trouvé et la valeur de theta pour ce parametre

def BIC(Y,m_min,m_max):
    m_opti=m_min
    BIC_min,theta_opti=BIC_m(Y,m_min)
    stock=[BIC_min]
    for m in range (m_min+1,m_max+1):
        Bic,theta=BIC_m(Y,m)
        if Bic<BIC_min:  # si la valeur du BIC est inferieur a celle stockée BIC_min
            m_opti=m     # la valeur de m courante est meilleur que celle contenu dans m_opti
            theta_opti=theta # conserve la valeur de theta pour le nombre de groupe optimal
            BIC_min=Bic
        stock.append(Bic)
    return stock,m_opti,theta_opti

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

# Calcul de la matrice de confusion
# Ligne groupe d'appartenance reel
# Colonne groupe d'appartenance estimé
def Matrice_confusion(Z,Z_estim,m,m_opti):
    n=len(Z)
    M=np.zeros((m,m_opti))
    for i in range(n):
        M[int(Z[i,1]),Z_estim[i]]+=1
    taux_err=(n-np.sum(np.diag(M)))/n*100
    return M,taux_err

### REPRESENTATION GRAPHIQUE ###

color=('b','g','r','c','m','y','k','w')

def graph_estim_densite(Y,theta,m):
    x=np.linspace(min(Y),max(Y),200)
    for i in range(m):
        plt.plot(x,theta['pi'][i]*phi(x,theta['mu'][i],theta['S'][i]),color=color[i],linestyle="--",label=f"fonction densite groupe {i+1}")
    plt.plot(x,g(x,m,theta),'r',label='g',linewidth=2)
    plt.hist(Y,bins=20,density=True,color="lightgrey",edgecolor='k',label="Densite de Y")
    plt.title("Estimation de la fonction de densite g")
    plt.legend(loc="upper right")

def graph_estim_group(Y,Z_estim,theta,m):
    for i in range(m):
        G=np.where(Z_estim==i) # Individus estimé comme appartenant au groupe i
        hist,bins=np.histogram(Y[G]) # Histogramme du groupe i estimé
        I=bins[1]-bins[0]
        A=I*np.sum(hist) # Aire de l'histogramme, utilisé pour avoir un histogramme en densité
        plt.hist(Y[G],bins=bins,weights=theta['pi'][i]/(A)*np.ones(len(Y[G])),color=color[i],alpha=.4,label=f"Groupe {i+1}")
    plt.title("Estimation de la densite de chaque groupe")
    plt.legend(loc="upper right")

def graph_reel(Z,theta,m):
    Y=Z[:,0]
    for i in range(m):
        G=np.where(Z[:,1]==i) # Individus estimé comme appartenant au groupe i
        hist,bins=np.histogram(Y[G]) # Histogramme du groupe i estimé
        I=bins[1]-bins[0]
        A=I*np.sum(hist) # Aire de l'histogramme, utilisé pour avoir un histogramme en densité
        plt.hist(Y[G],bins=bins,weights=theta['pi'][i]/(A)*np.ones(len(Y[G])),color=color[i],alpha=.4,label=f"Groupe {i+1}")
    plt.title("Densite reel de chaque groupe")
    plt.legend(loc="upper right")
    

def graph_bic(B,bic_min, bic_max):
    n=np.arange(bic_min,bic_max+1)
    plt.scatter(n,B)
    plt.plot(n,B)
    plt.title("Variation de la Valeur du BIC")
    plt.xlabel("Nombre de groupe")
    plt.ylabel("Valeur du BIC")

def affiche_parametre(theta,m):
    print(f"m : {m}\npi : {theta['pi']}\nmu : {theta['mu']}\nsigma^2 : {theta['S']}\n")

def EM_complet(Z,theta,m,tol,bic_min,bic_max):
    print("Parametres réels: ")
    affiche_parametre(theta,m)
    Y=Z[:,0]
    bic,m_opti,theta_opti=BIC(Y,bic_min, bic_max)
    print("\nParametres extimés: ")
    affiche_parametre(theta_opti,m_opti)
    Z_estim=groupe(Y,m_opti,theta_opti)
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1=graph_bic(bic,bic_min, bic_max)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2=graph_estim_densite(Y,theta_opti,m_opti)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3=graph_reel(Z,theta,m)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4=graph_estim_group(Y,Z_estim,theta_opti,m_opti)
    plt.show()
    return Z_estim,m_opti



if __name__=='__main__':
    #Initialisation de Z
    print("Exemple : ")
    x=input()
    
    if x=='1':
        pi=[1/6,1/2,1/3]
        mu=[-5,1,7]
        S=[1,4,1]
        
    elif x=='2':
        pi=[1/3,1/2,1/6]
        mu=[-3,1,5]
        S=[1,4,1]
    
    elif x=='3':
        pi=[1/3,1/2,1/6]
        mu=[-3,0,2]
        S=[1,2,1]

    m=3
    theta={'pi':np.array(pi), 'mu':np.array(mu),'S':np.array(S)}
    n=1000
    tol=10**-3
    
    Z=simu_Z(theta,n)
    
    bic_min=1
    bic_max=4
    
    Z_estim,m_opti=EM_complet(Z,theta,m,tol,bic_min,bic_max)
    
    M,taux_err=Matrice_confusion(Z,Z_estim,m,m_opti)
    print (f"Matrice de confusion:\n{M} \ntaux_d'erreur d'erreur : {taux_err}%")
