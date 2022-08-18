#!/usr/bin/env python3

import sys
from time import time

class Accessoire :
    """ Un accessoire est représenté par une liste de type LIFO """

    def __init__(self) :
        self.file = []

class Pic(Accessoire):
    """ Un pic peut embrocher un post-it par-dessus les post-it déjà présents
        et libérer le dernier embroché. """

    def embrocher(self,postit):
        self.file.append(postit)
        if verbose>1 :
            print(f"[{self.__class__.__name__}] post-it '{postit}' embroché\tTemps : {time()-t0:.5g} sec")
        if verbose>2 :
            print(f"[{self.__class__.__name__}]  état={self.file}\tTemps : {time()-t0:.5g} sec")
        
    def liberer(self):
        if verbose>2 :
            print(f"[{self.__class__.__name__}]  état={self.file}\tTemps : {time()-t0:.5g} sec")
        postit = self.file.pop()
        if verbose>1 :
            print(f"[{self.__class__.__name__}] post-it '{postit}' libéré\tTemps : {time()-t0:.5g} sec")
        return postit

class Bar(Accessoire):
    """ Un bar peut recevoir des plateaux, et évacuer le dernier reçu """
    def recevoir(self,plateau):
        self.file.append(plateau)
        if verbose>1 :
            print(f"[{self.__class__.__name__}] '{plateau}' reçu\tTemps : {time()-t0:.5g} sec")
        if verbose>2 :
            print(f"[{self.__class__.__name__}]  état={self.file}\tTemps : {time()-t0:.5g} sec")

    def evacuer(self):
        if verbose>2 :
            print(f"[{self.__class__.__name__}]  état={self.file}\tTemps : {time()-t0:.5g} sec")
        plateau = self.file.pop()
        if verbose>1 :
            print(f"[{self.__class__.__name__}] '{plateau}' évacué\tTemps : {time()-t0:.5g} sec")
        return plateau

class Serveur:
    def __init__(self,pic,bar,commandes):
        self.pic = pic
        self.bar = bar
        self.commandes = commandes
        print(f"[{self.__class__.__name__}] prêt pour le service !\tTemps : {time()-t0:.5g} sec")

    def prendre_commande(self):
        """ Prend une commande et embroche un post-it. """
        while self.commandes :
            commande = self.commandes.pop()
            print(f"[{self.__class__.__name__}] je prends commande de '{commande}'\tTemps : {time()-t0:.5g} sec")
            self.pic.embrocher(commande)

        print(f"[{self.__class__.__name__}] il n'y a plus de commande à prendre\tTemps : {time()-t0:.5g} sec")

    def servir(self):
        """ Prend un plateau sur le bar. """
        while self.bar.file :
            plateau = self.bar.evacuer()
            print(f"[{self.__class__.__name__}] je sers '{plateau}'\tTemps : {time()-t0:.5g} sec")

        if verbose>1 : 
            print(f'{self.bar.__class__.__name__} est vide\tTemps : {time()-t0:.5g} sec')

class Barman:
    def __init__(self,pic,bar):
        self.pic = pic
        self.bar = bar
        print(f"[{self.__class__.__name__}] prêt pour le service !\tTemps : {time()-t0:.5g} sec")
    
    def preparer(self):
        """ Prend un post-it, prépare la commande et la dépose sur le bar. """
        while self.pic.file :
            commande = self.pic.liberer()
            print(f"[{self.__class__.__name__}] je commence la fabrication de '{commande}'\tTemps : {time()-t0:.5g} sec")
            print(f"[{self.__class__.__name__}] je termine la fabrication de '{commande}'\tTemps : {time()-t0:.5g} sec")
            self.bar.recevoir(commande)
        
        if verbose > 1 : 
            print(f'{self.pic.__class__.__name__} est vide\tTemps : {time()-t0:.5g} sec')

if __name__ == "__main__" :
    global t0,verbose
    t0 = time() # Temps de départ de l'algorithme
    verbose = 1 # Niveau minimum de verbosité

    pic = Pic()
    bar = Bar()

    commandes = sys.argv[1:]

    if len(commandes)>0 :
        if "=" in commandes[-1] :
            verb = commandes.pop().split('=')[-1] # on récupère le niveau de verbosité
            try :
                verbose = int(verb)
            except ValueError :
                verbose = int(input("Veuillez saisir un niveau de verbosité correct (1, 2 ou 3) : "))
            while verbose not in [1,2,3] :
                verbose = int(input("Veuillez saisir un niveau de verbosité correct (1, 2 ou 3) : "))        

    if len(commandes)==0 :
        print(f"Aucune commande passée\tTemps : {time()-t0:.5g} sec")
        while True :
            c = input("Passez une commande : ")
            if c!="" :
                commandes.append(c)
            else :
                break 

    barman = Barman(pic,bar)
    serveur = Serveur(pic,bar,commandes)

    serveur.prendre_commande()
    if verbose>1 : 
        print(f"plus de commande à prendre\tTemps : {time()-t0:.5g} sec")
    barman.preparer()
    serveur.servir()