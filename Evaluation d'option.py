# -*- coding: utf-8 -*-
"""
Created on Sun May 4 01:48:37 2020

@author: Sofiane_C
"""

import scipy.stats as st
import numpy as np

#Implementation du modèle d'évaluation d'option CRR.

def OptionValue(N,S,K,T,sig,r):
    t = T/N
    up = np.exp(np.sqrt(t)*sig)
    down = 1/up
    p = (np.exp(r*t)-down)/(up-down)
    call = np.zeros((N+1,N+1))
    price  = np.zeros((N+1,N+1))
    for k in range(N+1):
        for i in range(k+1):
            price[i,k] = (down**i)*(up**(k-i))*S
   
    for k in range(N+1,0,-1):
        for i in range(k):
            if k == (N+1):
                call[i,k-1] = max(price[i,k-1] - K, 0)
            else:
                call[i,k-1] = np.exp(-r*t)*(p*call[i,k] + (1-p)*call[i+1,k])
    return (price, call)

def Call_BS(S,K,T,sig,r):
    down = (T*(sig**2)*0.5 - r + np.log(K/S))*(1/sig*np.sqrt(T))
    e = down - np.sqrt(T)*sig
    call  = S*(1 - st.norm.cdf(e)) - np.exp(-r*T)*K*(1 - st.norm.cdf(down))
    return call

   
S = 50
K = 50
T = 1
r = 0.3
sig = 0.1
N = [10,20,30,40,50,100]

print("Call_BS ", Call_BS(S,K,T,sig,r))
for i in range(0,6):
    print("Cox-Ross-Rubinstein model N = ", N[i], " ", OptionValue(N[i],S,K,T,sig,r)[1][0][0])



