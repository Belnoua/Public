
""" Modèles de semi-variogramme principaux """

import numpy as np 

""" Portée exacte  """
# Pépite c0, palier c0+c, portée a
def lin_p(r,c0,c,a):
    if r <= a: return c0+c/a*r
    else : return c0+c 

"""  Portée asymptotique  """
# Pépite c0, palier c0+c, portée pratique 3*a
def expo(r, c0, c, a):
    return c0+c*(1-np.exp(-r/a))

# Pépite c0, palier c0+c, portée pratique a*sqrt(3)
def gaussien(r, c0, c, a):
    return c0+c*(1-np.exp(-(r/a)**2))

""" Modèles sans palier """ 
# Pépite c0, facteur échelle m
def lin_sp(r, c0, m):
    return c0+m*r

def puissance(r, c0, m, k):
    return c0+m*r**k
