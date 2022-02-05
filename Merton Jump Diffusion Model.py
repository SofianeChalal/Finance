# -*- coding: utf-8 -*-
"""
@author: Sofiane_C

Merton Jump Diffusion Model

"""

import numpy as np
import matplotlib.pyplot as plt


def Merton_model(T, S_0, r, sigma, lam, m, sj, steps):
    dt = T/steps #Pas de temps
    
    pois = np.multiply(np.random.poisson( lam*dt, size=steps),
                         np.random.normal(m,sj, size=steps)).cumsum(axis=0)
    
    geo = np.cumsum(((r -  0.5*(sigma**2) - lam*(m  + 0.5*(sj**2)))*dt +\
                              sigma*np.sqrt(dt) * \
                              np.random.normal(size=steps)), axis=0)
    
    return S_0*np.exp(geo+pois)

def Geo_brownian(T, S_0, sigma, mu, steps):
    dt = T/steps
    A = np.cumsum(mu - 0.5*sigma**2  + sigma*np.sqrt(dt)*np.random.normal(size=steps), axis = 0)
    return S_0*np.exp(A)



T = 1 # Temps de maturité.
S_0 = 50 # Prix de l'action initial.
steps = 365 #  Nombres de jours.
r = 0.1 # Taux sans risque.
lam = 1 # Intensité des sauts.
m = 0.5 # Nombre moyen de sauts.
mu = 0.0 # Moyenne d'une normal.
sj = 0.1 # Standard deviation d'un saut.
sigma = 0.1 # Deviation standard pour un processus Gaussien.

Merton_model = Merton_model(T, S_0, r, sigma, lam, m, sj, steps)
Geo_brownian = Geo_brownian(T,S_0,sigma,mu, steps)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(Merton_model, color = 'blue', label = 'Merton Saut-Diffusion')
ax.plot(Geo_brownian, color = 'red', label = 'Mvt Brownien Geometrique')
plt.legend()
plt.xlabel('Jours')
plt.ylabel('Prix de l\'action $S_t$')
plt.grid(True)
plt.title('Comparaison entre un processus de diffusion et un processus Saut-diffusion')
