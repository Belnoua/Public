#!/usr/bin/env python3

import sys
from time import time
import asyncio

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

    async def prendre_commande(self):
        """ Prend une commande et embroche un post-it. """
        while self.commandes :
            commande = self.commandes.pop()
            print(f"[{self.__class__.__name__}] je prends commande de '{commande}'\tTemps : {time()-t0:.5g} sec")
            self.pic.embrocher(commande)
            await asyncio.sleep(0)

        print(f"[{self.__class__.__name__}] il n'y a plus de commande à prendre\tTemps : {time()-t0:.5g} sec")
        global fin_commande
        fin_commande = True
        if verbose>1 : 
            print(f"plus de commande à prendre \t {time()-t0:.5g} sec")

    async def servir(self):
        """ Prend un plateau sur le bar. """
        while True :
            try :
                plateau = self.bar.evacuer()
                print(f"[{self.__class__.__name__}] je sers '{plateau}'\tTemps : {time()-t0:.5g} sec")
                await asyncio.sleep(0)
            except IndexError :
                if verbose>1 : 
                    print(f'{self.bar.__class__.__name__} est vide\tTemps : {time()-t0:.5g} sec')
                if fin_preparation :
                    break
                else :
                    while not self.bar.file :
                        await asyncio.sleep(0)

class Barman:
    def __init__(self,pic,bar):
        self.pic = pic
        self.bar = bar
        print(f"[{self.__class__.__name__}] prêt pour le service !\tTemps : {time()-t0:.5g} sec")
    
    async def preparer(self):
        """ Prend un post-it, prépare la commande et la dépose sur le bar. """
        while True :
            try :
                commande = self.pic.liberer()
                print(f"[{self.__class__.__name__}] je commence la fabrication de '{commande}'\tTemps : {time()-t0:.5g} sec")
                await asyncio.sleep(0) # simule le temps de réalisation d'une boisson                
                print(f"[{self.__class__.__name__}] je termine la fabrication de '{commande}'\tTemps : {time()-t0:.5g} sec")
                self.bar.recevoir(commande)
                await asyncio.sleep(0)
            except IndexError :
                if verbose > 1 : 
                    print(f'{self.pic.__class__.__name__} est vide\tTemps : {time()-t0:.5g} sec')
                if fin_commande :
                    global fin_preparation
                    fin_preparation = True
                    break
                else :
                    while not self.pic.file :
                        await asyncio.sleep(0)

async def main() :
    global t0,verbose,fin_commande,fin_preparation
    t0 = time() # Temps de départ de l'algorithme
    verbose = 1 # Niveau minimum de verbosité

    # Gestion de la fermeture du bar
    fin_commande = False
    fin_preparation = False

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

    await asyncio.gather(serveur.prendre_commande(),barman.preparer(),serveur.servir())

asyncio.run(main())