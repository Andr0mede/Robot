# -*- coding: utf-8 -*-

cible = (10, 20)

position = (0, 0)

class reachTarget:
    def __init__(self, position):
        self.y = position[0]
        self.x = position[1]
        self.chemin = [(self.y, self.x)]
        self.directions = []
        
    def liste_move(self, cible):
        y_cible = cible[0]
        x_cible = cible[1]
        
        if (x_cible >= self.x) and (y_cible <= self.y): # en haut a droite
            while((self.chemin[-1][0] != x_cible) and (self.chemin[-1][1] != y_cible)):
                self.chemin.append((self.chemin[-1][0] + 1, self.chemin[-1][1] - 1))
            if (x_cible == self.chemin[-1][0]):
                while (y_cible != self.chemin[-1][1]):
                    self.chemin.append((self.chemin[-1][0], self.chemin[-1][1] - 1))
            if (y_cible == self.chemin[-1][1]):
                while (x_cible != self.chemin[-1][0]):
                    self.chemin.append((self.chemin[-1][0] + 1, self.chemin[-1][1]))
        
        if (x_cible <= self.x) and (y_cible <= self.y): # en haut a gauche
            while((self.chemin[-1][0] != x_cible) and (self.chemin[-1][1] != y_cible)):
                self.chemin.append((self.chemin[-1][0] - 1, self.chemin[-1][1] - 1))
            if (x_cible == self.chemin[-1][0]):
                while (y_cible != self.chemin[-1][1]):
                    self.chemin.append((self.chemin[-1][0], self.chemin[-1][1] - 1))
            if (y_cible == self.chemin[-1][1]):
                while (x_cible != self.chemin[-1][0]):
                    self.chemin.append((self.chemin[-1][0] - 1, self.chemin[-1][1]))
        
        if (x_cible <= self.x) and (y_cible >= self.y): # en bas a gauche
            while((self.chemin[-1][0] != x_cible) and (self.chemin[-1][1] != y_cible)):
                self.chemin.append((self.chemin[-1][0] - 1, self.chemin[-1][1] + 1))
            if (x_cible == self.chemin[-1][0]):
                while (y_cible != self.chemin[-1][1]):
                    self.chemin.append((self.chemin[-1][0], self.chemin[-1][1] + 1))
            if (y_cible == self.chemin[-1][1]):
                while (x_cible != self.chemin[-1][0]):
                    self.chemin.append((self.chemin[-1][0] - 1, self.chemin[-1][1]))
        
        if (x_cible >= self.x) and (y_cible >= self.y): # en bas a droite
            while((self.chemin[-1][0] != x_cible) and (self.chemin[-1][1] != y_cible)):
                self.chemin.append((self.chemin[-1][0] + 1, self.chemin[-1][1] + 1))
            if (x_cible == self.chemin[-1][0]):
                while (y_cible != self.chemin[-1][1]):
                    self.chemin.append((self.chemin[-1][0], self.chemin[-1][1] + 1))
            if (y_cible == self.chemin[-1][1]):
                while (x_cible != self.chemin[-1][0]):
                    self.chemin.append((self.chemin[-1][0] + 1, self.chemin[-1][1]))
        
        for i in range(1, len(self.chemin)):
            trajectoire = (self.chemin[i][0] - self.chemin[i-1][0], self.chemin[i][1] - self.chemin[i-1][1])
            if trajectoire == (-1, -1):
                self.directions.append(5)
            elif trajectoire == (0, -1):
                self.directions.append(3)
            elif trajectoire == (1, -1):
                self.directions.append(6)
            elif trajectoire == (-1, 0):
                self.directions.append(1)
            elif trajectoire == (1, 0):
                self.directions.append(2)
            elif trajectoire == (-1, 1):
                self.directions.append(7)
            elif trajectoire == (0, 1):
                self.directions.append(4)
            elif trajectoire == (1, 1):
                self.directions.append(8)
            
        return self.chemin, self.directions
                
                
    
    
my_class = reachTarget(position)
chemin, directions = my_class.liste_move(cible)
print(chemin)
print(directions)