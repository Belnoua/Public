import numpy as np 
import multiprocessing as mp
import matplotlib.pyplot as plt
import itertools
from numpy.linalg import norm, inv, LinAlgError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Map import Map, Source
from Echantillonage import Ech_rdm, Ech_grid
from Modeles import *
from SaveMap import *
import argparse
from Krigeage import Krigeage

if __name__=="__main__":

    nb_source = 50  # Nombre de sources de diffusion de chaleur
    nb_pts = 100    # Nombre de points par axe de la grille


    """     Initialisation des paramètres de la Map, par défaut ou par utilisateur """      
    parser = argparse.ArgumentParser()
    parser.add_argument('-mth', help="grid for distribution on grid or rdm for a random distribution")
    parser.add_argument('-nb_ech', type = int, help = "Taille de l'échantillon")
    parser.add_argument('-parametre', help = "Paramètre à faire varier")
    parser.add_argument('-a', type = float, help = "Minoration du paramètre à faire varier")
    parser.add_argument('-b', type = float, help = "Majoration du paramètre à faire varier")
    parser.add_argument('-func', help = "Fonction au choix : gaussien, puissance, expo, lin_p (linéaire avec palier)")
    args = parser.parse_args()

    if args.mth == "grid" or args.mth == "rdm": mth = args.mth
    else: mth = "rdm"

    if args.nb_ech != None : nb_ech = args.nb_ech
    else:
        if mth == "grid" : nb_ech = 10
        else : nb_ech = 100

    if args.parametre != None : param = args.parametre
    else: param = "c0"
    
    if args.a != None : u = args.a
    else: u = 0.1
    
    if args.b != None : v = args.b
    else: v = 10

    if args.func == "expo": f = expo 
    elif args.func == "lin_p": f = lin_p 
    elif args.func == "puissance": f = puissance 
    else: f = gaussien 

    """ Création ou récupération des données de la Map """
    Xlim = [0,10]   # Limite axe des X
    Ylim = [0,10]   # Limite axe des Y
    Zlim = [-10,10] # Intervales des valeurs des sources

    Src = Source(Xlim,Ylim,Zlim,nb_source)
    # X,Y,Z = Map(Src,Xlim,Ylim,nb_pts)
    X,Y,Z = recup_Map("data/Map.csv")


    """ Création ou récupération des données d'échantillonnage, i.e. les sites d'observation """
    # if mth == "grid" :
    #     E = Ech_grid(X,Y,Z,nb_ech)   # Echantillonnage aléatoire
    # else:
    #     E = Ech_rdm(X,Y,Z,nb_ech)   # Echantillonnage sur grille

    # S, S0 = train_test_split(E,test_size = 0.2) # Partage en 2 échantillons train et test (resp. 80% et 20%)
    
    path_Ech = mth + f"_{nb_ech}.csv"
    S,S0 = recup_Ech("data/train_"+path_Ech,"data/test_"+path_Ech)
    # save_Ech(S, S0, path_Ech)


    """ Variations des différents paramètres """
    Z0 = []
    variations = np.linspace(u,v,5)
    for p in variations:
        if param == "c0":
            Z0.append(Krigeage(X,Y,S,f,p,5,2))
        elif param == "c":
            Z0.append(Krigeage(X,Y,S,f,1.5,p,2))
        elif param == "a":
            Z0.append(Krigeage(X,Y,S,f,1.5,5,p))
        elif param == "m":
            Z0.append(Krigeage(X,Y,S,puissance,1.5,p,2))
        elif param == "k":
            Z0.append(Krigeage(X,Y,S,puissance,1.5,5,p))


    """ Représentation graphique """
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize = (24,4))
    C1 = ax1.contourf(X, Y, Z0[0], levels = 30, cmap = "coolwarm")
    ax1.set_title(f"{param} = {variations[0]}", fontsize = 20)
    fig.colorbar(C1, ax = ax1)

    C2 = ax2.contourf(X, Y, Z0[1], levels = 30, cmap = "coolwarm")
    ax2.set_title(f"{param} = {variations[1]}", fontsize = 20)
    fig.colorbar(C2, ax = ax2)
    
    C3 = ax3.contourf(X, Y, Z0[2], levels = 30, cmap = "coolwarm")
    ax3.set_title(f"{param} = {variations[2]}", fontsize = 20)
    fig.colorbar(C3, ax = ax3)
    
    C4 = ax4.contourf(X, Y, Z0[3], levels = 30, cmap = "coolwarm")
    ax4.set_title(f"{param} = {variations[3]}", fontsize = 20)
    fig.colorbar(C4, ax = ax4)
    
    C5 = ax5.contourf(X, Y, Z0[4], levels = 30, cmap = "coolwarm")
    ax5.set_title(f"{param} = {variations[4]}", fontsize = 20)
    fig.colorbar(C5, ax = ax5)

    plt.show()
