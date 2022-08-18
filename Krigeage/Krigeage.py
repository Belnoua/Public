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

def gamma_0(s0, S, f, c0, c, a):
    """ 
    Calcul de gamma_0 
    """
    n = len(S)
    R = [norm(s-s0) for s in S[:,:2]]
    g0 = np.array([f(r,c0,c,a) for r in R]).reshape(n,1)
    return g0

def Gamma(S, f, c0, c, a):
    """ 
    Calcul de Gamma
    """
    n = len(S)
    G = c0/2*np.eye(n)
    for i in range(n-1):
        for j in range (i+1,n):
            r = norm(S[i,:2]-S[j,:2])
            G[i,j] = f(r, c0, c, a)
    G += G.T
    return G

def Estim_z0(g0, uG, uGu, GZ, u):
    """ 
    Estimation du site z0 par Krigeage ordinaire
    en fonction de gamma0, Gamma, vecteur unitaire
    """
    A = g0 + (1 - uG@g0)/uGu*u
    Z_estim = np.dot(A.T,GZ)[0]
    return Z_estim

def Estimation(S, S0, f, c0, c, a):
    """ 
    Estimation d'un échantillon test
    """
    Z0_estim = []
    n = len(S)

    #Calcul des composantes invariables de la formule
    u = np.ones((n,1))
    G = Gamma(S, f, c0, c, a)
    G_inv = inv(G)
    uG = u.T @ G_inv
    uGu = uG @ u
    if uGu==0 : raise ValueError
    GZ = G_inv @ S[:,2]

    for s0 in S0 :
        g0 = gamma_0(s0[:2], S, f, c0, c, a)
        Z0_estim.append(Estim_z0(g0, uG, uGu, GZ, u))
    return Z0_estim

def fit_param(S,S0,f,dico_model):
    """ 
    Retourne le modèle minimisant la MSE
    """
    best_param = {"MSE": np.inf, "f":f,"fonction":f.__name__}
    if f==puissance: 
        for k,m,c0 in itertools.product(dico_model["k"],dico_model["m"],dico_model["c0"]):
            try:
                estim = Estimation(S,S0,f,c0,m,k) 
                MSE = mean_squared_error(S0[:,2],estim) # Calcul de l'erreur test
                if MSE < best_param["MSE"]:
                    best_param["k"] = k
                    best_param["c0"] = c0
                    best_param["m"] = m
                    best_param["Median"] = np.median((S0[:,2]-estim)**2)
                    best_param["MSE"] = MSE
            except (LinAlgError,ValueError): 
                pass
    else:
        for a,c,c0 in itertools.product(dico_model["a"],dico_model["c"],dico_model["c0"]):
            try:
                estim = Estimation(S,S0,f,c0,c,a) 
                MSE = mean_squared_error(S0[:,2],estim) # Calcul de l'erreur test
                if MSE < best_param["MSE"]:
                    best_param["a"] = a
                    best_param["c0"] = c0
                    best_param["c"] = c
                    best_param["Median"] = np.median((S0[:,2]-estim)**2)
                    best_param["MSE"] = MSE
            except LinAlgError: 
                pass
    return best_param 

def Krigeage(X, Y, S, f, c0, c, a):
    """ 
    Retourne la Map "krigée"
    """
    nx = len(X)
    ny = len(Y) 
    xx,yy = np.meshgrid(X,Y)
    S0 = np.c_[xx.ravel(),yy.ravel()] # Vecteur des points de la grille
    Z0 = np.array(Estimation(S, S0, f, c0, c, a)).reshape(nx,ny)
    return Z0

class Fit_fun(mp.Process):
    """ 
    Recherche du meilleur semi-variogramme par parallélisation
    """
    def __init__(self, S, S0, f, dico_model, q):
        mp.Process.__init__(self)
        self.S = S
        self.S0 = S0
        self.f = f
        self.dico_model = dico_model
        self.q = q

    def run(self) :
        param = fit_param(self.S, self.S0, self.f, self.dico_model)
        self.q.put(param, block = False)

if __name__=="__main__":

    nb_source = 50  # Nombre de sources de diffusion de chaleur
    nb_pts = 100    # Nombre de points par axe de la grille


    """     Initialisation des paramètres de la Map, par défaut ou par utilisateur """      
    parser = argparse.ArgumentParser()
    parser.add_argument('-mth', help="grid pour une distribution sur grille ou rdm pour une distribution aléatoire")
    parser.add_argument('-nb_ech', type = int, help = "Taille de l'échantillon")
    parser.add_argument('-a', type = float, help = "Portée")
    parser.add_argument('-c', type = float, help = "Palier")
    parser.add_argument('-c0', type = float, help = "Pépite")
    parser.add_argument('-m', type = float, help = "Facteur echelle")
    parser.add_argument('-k', type = float, help = "Puissance")
    args = parser.parse_args()

    if args.mth == "grid" or args.mth == "rdm": mth = args.mth
    else: mth = "rdm"

    if args.nb_ech != None : nb_ech = args.nb_ech
    else:
        if mth == "grid" : nb_ech = 10
        else : nb_ech = 100

    dico_model = {}    

    if args.a != None : dico_model["a"] = [args.a]
    else: dico_model["a"]= np.linspace(1,10,20)

    if args.c != None : dico_model["c"] = [args.c]
    else: dico_model["c"]= np.linspace(0,10,20)

    if args.c0 != None : dico_model["c0"] = [args.c0]
    else: dico_model["c0"] = np.linspace(0,10,20)

    if args.m != None : dico_model["m"] = [args.m]
    else: dico_model["m"] = np.linspace(0,10,20)

    if args.k != None : dico_model["k"] = [args.k]
    else: dico_model["k"] = np.linspace(0,2,20)

    #path_Ech = mth + f"_{nb_ech}.csv"

    """ Création ou récupération des données de la Map """
    Xlim = [0,10]   # Limite axe des X
    Ylim = [0,10]   # Limite axe des Y
    Zlim = [-10,10] # Intervalles des valeurs des sources

    Src = Source(Xlim,Ylim,Zlim,nb_source)
    #save_Source(Src)

    X,Y,Z = Map(Src,Xlim,Ylim,nb_pts)
    #save_Map(X,Y,Z)

    #X,Y,Z = recup_Map("data/Map.csv")


    """ Création ou récupération des données d'échantillonnage, i.e. les sites d'observation """

    if mth == "grid" :
        E = Ech_grid(X,Y,Z,nb_ech)   # Echantillonnage aléatoire
    else:
        E = Ech_rdm(X,Y,Z,nb_ech)   # Echantillonnage sur grille

    S, S0 = train_test_split(E,test_size = 0.2) # Partage en 2 échantillons train et test (resp. 80% et 20%)
    #save_Ech(S, S0, path_Ech)

    #S,S0 = recup_Ech("data/train_" + path_Ech,"data/test_" + path_Ech)
    
    """ Recherche du meilleur semi-variogramme en multi-processing """
    liste_F = [lin_p, expo, gaussien, puissance] # Modèles de semi-variogramme à tester
    q = mp.Queue()

    tasks = [Fit_fun(S,S0,f,dico_model,q) for f in liste_F]

    for i in tasks:
        i.start()

    for i in tasks:
        i.join()

    df = pd.DataFrame()
    while not q.empty():
        d = q.get(block = True)
        df = df.append(d,ignore_index=True)

    df.sort_values(by=["MSE"],inplace=True)
    df=df.reindex(columns=["fonction","c0","a","c","m","k","MSE","Median","f"])
    print(df[df["fonction"]!="puissance"].drop(["f",'k','m'],axis=1),"\n")
    print(df[df["fonction"]=="puissance"].drop(["f",'a','c'],axis=1),"\n")
    #df.to_csv("Semi_vario_" + path_Ech)

    """ Sélection du meilleur modèle parmi les fonctions optimisées """
    best_model = pd.DataFrame.copy(df.iloc[0])
    
    """ Application du meilleur modèle sur l'ensemble des points de la grille """
    if best_model["f"] == puissance :
        Z0 = Krigeage(X,Y,S,best_model["f"],best_model["c0"],best_model["m"],best_model["k"])
    else : 
        Z0 = Krigeage(X,Y,S,best_model["f"],best_model["c0"],best_model["c"],best_model["a"])


    if best_model["f"]==puissance:
        best_model.drop(['a','c','f'],inplace=True)
    else :
        best_model.drop(['k','m','f'],inplace=True)
    print(f"Semi-variogramme le plus adapté : \n{best_model}.")

    MSE = mean_squared_error(Z, Z0)
    print(f"L'erreur de Krigeage est de {MSE}.")


    """ Représentation graphique """
    fig = plt.figure(figsize = (16,10))

    ax1 = plt.subplot(221)
    C1 = ax1.contourf(X, Y, Z, levels = 20, cmap = "coolwarm")
    ax1.scatter(S[:,0], S[:,1], marker = 'x', color = 'k', label = "Train")
    ax1.scatter(S0[:,0], S0[:,1], marker = 'o',color = 'g', label = "Test")
    ax1.set_title('Map', fontsize = 20)
    fig.colorbar(C1, ax = ax1)
    
 
    ax2 = plt.subplot(222)
    C2 = ax2.contourf(X, Y, Z0, levels = 20, cmap = "coolwarm")
    ax2.set_title('Krigeage', fontsize = 20)
    fig.colorbar(C2, ax = ax2)

   
    ax3 = plt.subplot(235)
    C3 = ax3.contourf(X, Y, abs(Z-Z0), levels = 20, cmap = "Reds")
    ax3.scatter(S[:,0], S[:,1], marker = 'x', color = 'k')
    ax3.scatter(S0[:,0], S0[:,1], marker = 'o',color = 'g')
    ax3.set_title('Erreur', fontsize = 20)
    fig.colorbar(C3, ax = ax3)

    fig.legend(fontsize = "x-large")
    plt.show()
