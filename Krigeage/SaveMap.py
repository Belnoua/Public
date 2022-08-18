import numpy as np 
import pandas as pd

def save_Source(Src):
    """ 
    Sauvegarde des sources
    """
    df_Map = pd.DataFrame(Src)
    df_Map.to_csv("data/Source.csv", index=False)

def recup_Source(Src):
    """ 
    Récupération des sources
    """
    df_Src = pd.read_csv(Src)
    Src = np.array(df_Src)
    return Src

def save_Map(X, Y, Z):
    """ 
    Sauvegarde des données de la Map
    """
    df_Map = pd.DataFrame(np.c_[X,Y,Z])
    df_Map.to_csv("data/Map.csv", index=False)

def recup_Map(Map):
    """ 
    Récupération des données de la Map
    """
    df_Map = pd.read_csv(Map)
    X = np.array(df_Map.iloc[:,0])
    Y = np.array(df_Map.iloc[:,1])
    Z = np.array(df_Map.iloc[:,2:])
    return X,Y,Z

def save_Ech(S, S0, path):
    """ 
    Sauvegarde des échantillons d'entraînement et de test
    """
    path_train = "data/train_"+ path
    path_test = "data/test_"+ path
    train = pd.DataFrame(S)
    test = pd.DataFrame(S0)
    train.to_csv(path_train, index = False)
    test.to_csv(path_test, index = False)

def recup_Ech(train,test):
    """ 
    Récupération des échantillons d'entraînement et de test
    """
    train = pd.read_csv(train)
    test = pd.read_csv(test)
    S = np.array(train)
    S0 = np.array(test)
    return S,S0
    
def save_result(param, p, mth, nb_ech, best_model,  MSE):
    """ 
    Sauvegarde du meilleur modèle et de son erreur 
    """
    path = "data/best_modele"
    with open(path,'a') as f:
        f.write(f"\nParametre echantillonage: {mth},\t{nb_ech}\n")
        if not np.isnan(p):
            f.write(f"Parametre fixe : " + param + f"={p}\n")
        f.write(f"Meilleur modèle : {best_model}\n")
        f.write(f"MSE : {MSE}\n")

def save_Krigeage(X, Y, Z, path):
    """ 
    Sauvegarde des données de la Map "Krigée"
    """
    file_out ="data/Krigeage_" + path
    df_Kri = pd.DataFrame(np.c_[X,Y,Z])
    df_Kri.to_csv(file_out, index = False)
