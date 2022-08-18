
""" Affinage de la sélection des paramètres par validation croisée sur le modèle gaussien """

import numpy as np 
import multiprocessing as mp
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from Map import Map, Source
from Echantillonage import Ech_rdm, Ech_grid
from SaveMap import *
import argparse, itertools
from Krigeage import Krigeage, Estimation
from Modeles import *

def fit_param(S,S0,f):
    """ 
    Retourne le modèle gaussien minimisant la MSE, par recherche des paramètres c0, c et a
    """
    best_param = {"MSE": np.inf, "f":f}

    for c0,a,c in itertools.product(np.linspace(0,2,5), np.linspace(1,10,10), np.linspace(0,10,10)):
        try:
            estim = Estimation(S,S0,f,c0,c,a) 
            MSE = mean_squared_error(S0[:,2],estim) # Calcul de l'erreur test
            if MSE < best_param["MSE"]:
                best_param["c0"] = c0
                best_param["c"] = c
                best_param["a"] = a
                best_param["MSE"] = MSE
        except LinAlgError or ValueError: 
            pass
    return best_param 

class Fit_fun(mp.Process):
    """ 
    Recherche du meilleur semi-variogramme par parallélisation
    """
    def __init__(self, S, S0, f, q):
        mp.Process.__init__(self)
        self.S = S
        self.S0 = S0
        self.f = f
        self.q = q

    def run(self) :
        param = fit_param(self.S, self.S0, self.f)
        self.q.put(param, block = False)


if __name__=="__main__":

    nb_source = 50  # Nombre de sources de diffusion de chaleur
    nb_pts = 100    # Nombre de points par axe de la grille

    """     Initialisation des paramètres de la Map, par défaut ou par utilisateur """      
    parser = argparse.ArgumentParser()
    parser.add_argument('-mth', help="grid for distribution on grid or rdm for a random distribution")
    parser.add_argument('-KF', type = int, help = "paramètre KFold")
    parser.add_argument('-nb_ech', type = int, help = "Taille de l'échantillon")
    parser.add_argument('-func', help = "Fonction au choix : gaussien, puissance, expo, lin_p (linéaire avec palier)")
    args = parser.parse_args()

    if args.mth == "grid" or args.mth == "rdm": mth = args.mth
    else: mth = "rdm"

    if args.nb_ech != None : nb_ech = args.nb_ech
    else:
        if mth == "grid" : nb_ech = 10
        else : nb_ech = 100

    if args.KF != None : k = args.KF
    else: k = 5

    if args.func == "expo": f = expo 
    elif args.func == "lin_p": f = lin_p 
    elif args.func == "puissance": f = puissance 
    else: f = gaussien 

    """ Création ou récupération des données de la Map """
    Xlim = [0,10]   # Limite axe des X
    Ylim = [0,10]   # Limite axe des Y
    Zlim = [-10,10] # Intervales des valeurs des sources

    Src = Source(Xlim,Ylim,Zlim,nb_source)
    X,Y,Z = Map(Src,Xlim,Ylim,nb_pts)
    #X,Y,Z = recup_Map("data_diapo/Map.csv")


    """ Création ou récupération des données d'échantillonnage, i.e. les sites d'observation """
    if mth == "grid" :
        E = Ech_grid(X,Y,Z,nb_ech)   # Echantillonnage aléatoire
    else:
        E = Ech_rdm(X,Y,Z,nb_ech)   # Echantillonnage sur grille
    
    #param = mth+f"_{nb_ech}.csv"
    #S,S0 = recup_Ech("data_diapo/train_"+param,"data_diapo/test_"+param)
    #E = np.concatenate((S,S0))
    
    """ Recherche des meilleurs paramètres du semi-variogramme gaussien en multi-processing et cross-validation """
    q = mp.Queue()

    cv = []
    KF = KFold(n_splits = k)
    for train_index, test_index in KF.split(E):
        cv.append([E[train_index], E[test_index]])

    tasks = [Fit_fun(ech[0],ech[1],f,q) for ech in cv]

    for i in tasks:
        i.start()

    liste_model = [q.get() for i in tasks]

    for i in tasks:
        i.join()
    

    """ Moyennisation des paramètres optimaux """
    c = sum([model['c'] for model in liste_model])/k
    a = sum([model['a'] for model in liste_model])/k
    c0 = sum([model['c0'] for model in liste_model])/k
    
    best_model = {'f':gaussien, 'c0':c0, 'c':c, 'a': a}
    

    """ Application du meilleur modèle sur l'ensemble des points de la grille """
    Z0 = Krigeage(X, Y, E, best_model["f"],c0,c,a)

    MSE = mean_squared_error(Z, Z0)
    best_model["f"]=best_model["f"].__name__
    print(best_model)
    best_model["MSE totale"] = MSE
    print(f"L'erreur de Krigeage est de {MSE}.")


    """ Représentation graphique """
    fig = plt.figure(figsize = (16,8))

    ax1 = plt.subplot(221)
    C1 = ax1.contourf(X, Y, Z, levels = 30, cmap = "coolwarm")
    ax1.scatter(E[:,0], E[:,1], marker = '+', color = 'k', label = "Sites d'observation")
    ax1.set_title('Map')
    fig.colorbar(C1, ax = ax1)

    ax2 = plt.subplot(222)
    C2 = ax2.contourf(X, Y, Z0, levels = 30, cmap = "coolwarm")
    ax2.set_title('Krigeage')
    fig.colorbar(C2, ax = ax2)

    ax3 = plt.subplot(235)
    C3 = ax3.contourf(X, Y, abs(Z-Z0), levels = 30, cmap = "Reds")
    ax3.scatter(E[:,0], E[:,1], marker = '+', color = 'k')
    ax3.set_title('Erreur', fontsize = 20)
    fig.colorbar(C3, ax = ax3)

    fig.legend(fontsize = "x-large")
    plt.show()

    """ Sauvegarde des résultats """
    #save_Krigeage(X, Y, Z0, "CV")
