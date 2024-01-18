#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:07:32 2023

@author: andro
"""

import numpy as np

def construire_sous_array(A, centre_B):
    h, w = A.shape
    i, j = centre_B
    
    # Déterminer la taille de B en fonction de la position du centre
    if i == 0 or i == h - 1:
        if j == 0 or j == w - 1:
            taille_B = (2, 2)  # Coin, taille 2*2
        else:
            taille_B = (2, 3)  # Bord vertical, taille 2*3
    elif j == 0 or j == w - 1:
        taille_B = (3, 2)  # Bord horizontal, taille 3*2
    else:
        taille_B = (3, 3)  # Centre, taille 3*3
    
    print('taille de B', taille_B)
    # Calculer les indices de début et de fin pour extraire B
    i_debut, j_debut = max(0, i - taille_B[0] // 2), max(0, j - taille_B[1] // 2)
    i_fin, j_fin = min(h, i_debut + taille_B[0]), min(w, j_debut + taille_B[1])
    
    # Extraire le sous-array B
    B = A[i_debut:i_fin, j_debut:j_fin]
    print(B)

    return B

# Exemple d'utilisation
A = np.arange(20).reshape((4, 5))

centre_B = (0, 2)  # Coordonnées du centre de B

B = construire_sous_array(A, centre_B)
print(B)

#%%
found = False
for i in range(3):
    for j in range(3):
        if i > 0:
            my_tuple = (i, j)
            found = True
            break
    if found:
        break


found = False
i = 0
while i < 3 and not found:
    j = 0
    while j < 3 and not found:
        if i > 1:
            my_tuple = (i, j)
            found = True
        j += 1
    i += 1
