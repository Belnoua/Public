
""" Méthodes d'échantillonnages """

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt 
from Map import Map, Source, Valeur

def Ech_rdm(X, Y, Z, nb_ech):
    """
    Echantillonnage avec sélection aléatoire des sites d'observation
    """
    xx,yy = np.meshgrid(X,Y)
    pts = np.c_[xx.ravel(),yy.ravel(),Z.ravel()] # Matrice des points de la grille
    nb_pt = pts.shape[0] # Nombre de points dans la grille
    ind = npr.choice(nb_pt, nb_ech) # Indices des sites d'observation
    return pts[ind] 

# def Ech_rdm2(X, Y, Z, nb_ech):
#     """
#     Autre méthode
#     """
#     xx,yy = np.meshgrid(X,Y)
#     pts = np.c_[xx.ravel(),yy.ravel(),Z.ravel()] # Matrice des points de la grille
#     sites = np.c_[pts, Z.ravel()]
#     np.random.shuffle(sites)
#     return sites[:nb_ech,:]

def Ech_grid(X,Y,Z,nb_g):
    """ 
    Echantillonnage sur grille
    """
    nx = len(X)
    ny = len(Y)
    ind_i = np.linspace(0, nx, nb_g+2, dtype = int)[1:-1]
    ind_j = np.linspace(0, ny, nb_g+2, dtype = int)[1:-1]
    xx,yy = np.meshgrid(X[ind_i],Y[ind_j])
    zz = np.array([[Z[i,j] for j in ind_j] for i in ind_i])
    return np.c_[xx.ravel(),yy.ravel(),zz.ravel()]
    
if __name__=="__main__":

    nb_source = 50  # Nombre de sources de diffusion de chaleur
    nb_pts = 100    # Nombre de points par axe de la grille
    nb_ech = 100    # Nombre de points d'"chantillonage
    nb_g = 10       # Nombre de points par axe de la grille d'echantillonage

    Xlim = [0,10]   # Limite axe des X
    Ylim = [0,10]   # Limite axe des Y
    Zlim = [-10,10] # Intervales des valeurs des sources

    Src = Source(Xlim, Ylim, Zlim, nb_source)
    X,Y,Z = Map(Src, Xlim, Ylim, nb_pts)

    E = Ech_rdm(X, Y, Z, nb_ech)
    plt.scatter(E[:,0],E[:,1],c=E[:,2],cmap="coolwarm")
    plt.title("Echantillonage Aléatoire")
    plt.colorbar()
    plt.show()

    E = Ech_grid(X,Y,Z,nb_g)
    plt.scatter(E[:,0],E[:,1],c=E[:,2],cmap="coolwarm")
    plt.title("Echantillonage sur grille")
    plt.colorbar()
    plt.show()
