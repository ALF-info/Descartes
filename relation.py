import numpy as np
from matplotlib import pyplot as plt

#%% Définition du tirage au sort de la valeur d'une variable

"""
    Renvoie une valeur x aléatoire de la variable X d'incertitude-type u(X)
        X = [x, u(X)] (loi normale)
"""

def Alea(X):
    tirage = np.random.normal()   
    return X[0]+X[1]*tirage

#%% Procedure Regression Linéaire; tableaux np X et Y (méthode des moindres carrés)
    
def RegLin(X,Y):
    N = len(X)
    moyX = sum(X)/N
    moyY = sum(Y)/N
    pente = sum((X-moyX)*(Y-moyY))/(sum((X-moyX)**2))   # calcule la pente de la droite de régression
    ordor = moyY - pente*moyX                           # calcule l'ordonnée à l'origine de la droite de régression
    return [pente,ordor]

#%% Entrées
    

incertitude_inverse_distance = 0.001                              # Estimation de l'incertitude-type sur l'inverse d'une mesure de distance (cm^-1)     
OA = [-9.0, -10.0, -11.0, -12.0, -13.0, -15.0, -20.0, -25.0, -30.0, -40.0,-50.0] # Liste contenant les distances objet-lentille mesurées (cm) !Attention au signe!
dist_lentille_ecran = [70.2, 41.1,30.0, 25.3, 20.0, 17.1, 13.2, 11.9, 11.1, 10.3, 9.8] # Liste contenant les tailles di'mage mesurées sur l'écran (cm) 

#%% Préparation des listes avec incertitudes

Dist_objet = []
for k in range(len(OA)):
    Dist_objet.append([1/OA[k], incertitude_inverse_distance])        # Remplit une liste de listes contenant les inverses des distances objet-lentille assorties de leur incertitude 
    
Dist_lentille_ecran = []
for k in range(len(dist_lentille_ecran)):
    Dist_lentille_ecran.append([1/dist_lentille_ecran[k], incertitude_inverse_distance ])  # Remplit une liste de listes contenant les inverses des distances lentille-écran assorties de leur incertitude 
    
#%% Méthode de Monte Carlo pour déterminer les incertitudes sur la pente et l'ordonnée à l'origine de la régression linéaire
    
LPente = []     # Crée une liste vide pour stocker les valeurs de la pente de la droite de régression issues de la simulation
LOrdor = []     # Crée une liste vide pour stocker les valeurs de l'ordonnée à l'origine de la droite de régression issues de la simulation

iterations = 100000     # Nombre d'essais de la simulation

for i in range(iterations):
    

    Alea_dist_objet = []            # Crée une liste vide pour stocker les valeurs de la distance objet-smartphone issues de la simulation
    Alea_dist_ecran = []
    for k in range(len(dist_lentille_ecran)):
       
        Alea_dist_objet.append(Alea(Dist_objet[k]))                                         # Remplit la liste  Alea_dist_objet avec des valeurs tirées au hasard (loi normale) de la distance objet-lentille
        Alea_dist_ecran.append(Alea(Dist_lentille_ecran[k]))                                # Remplit la liste  Alea_dist_ecran avec des valeurs tirées au hasard (loi normale) de la distance ecran-lentille
    Pente = RegLin(np.array(Alea_dist_objet),np.array(Alea_dist_ecran))[0]            # Calcule la pente de la droite de régression pour chaque itération
    OrdOr = RegLin(np.array(Alea_dist_objet),np.array(Alea_dist_ecran))[1]            # Calcule l'ordonnée à l'origine de la droite de régression pour chaque itération
    LPente.append(Pente)                                                                    # Remplit la liste LPente avec les valeurs calculées de la pente de la droite de régression pour chaque itération
    LOrdor.append(OrdOr)                                                                    # Remplit la liste LOrdor avec les valeurs calculées de l'ordonnée à l'origine de la droite de régression pour chaque itération
    
MoyPente = np.sum(LPente)/iterations                                                        # Calcule la moyenne des valeurs simulées de la pente 
MoyOrdOr = np.sum(LOrdor)/iterations                                                        # Calcule la moyenne des valeurs simulées de l'ordonnée à l'origine

incertitude_type_Pente = np.std(np.array(LPente))                                           # Calcule l'incertitude-type sur la pente de la droite de régression
incertitude_elargie_Pente = 2*incertitude_type_Pente                                        # Calcule l'incertitude élargie sur la pente de la droite de régression
incertitude_type_OrdOr = np.std(np.array(LOrdor))                                           # Calcule l'incertitude-type sur l'ordonnée à l'origine de la droite de régression
incertitude_elargie_OrdOr = 2*incertitude_type_OrdOr                                        # Calcule l'incertitude élargie sur l'ordonnée à l'origine de la droite de régression

#%% Affichage

print ('Pente de la droite de régression:', MoyPente, 'cm^-1')
print('Incertitude élargie sur la pente :',incertitude_elargie_Pente, 'cm^-1' )
plt.hist(LPente, range = (0.8, 1.2), bins = 50, color = 'orange', edgecolor = 'black')        # Affiche l'histogramme de répartion des valeurs simulées de la pente
plt.xlabel('Pente (cm^-1)')
plt.ylabel('effectif')
plt.title('Pour 100 000 iterations')
plt.show()

print("Ordonnée à l origine :",MoyOrdOr)
print("Incertitude élargie sur l'ordonnée à l origine:",incertitude_elargie_OrdOr)
plt.hist(LOrdor, range = (0.1, 0.15), bins = 50, color = 'blue', edgecolor = 'black')            # Affiche l'histogramme de répartion des valeurs simulées de l'ordonnée à l'origine
plt.xlabel("Ordonnée à l'origine")
plt.ylabel('effectif')
plt.title('Pour 100 000 iterations')
plt.show()    
