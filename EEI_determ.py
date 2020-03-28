# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:51:45 2020

@author: Ming Cai
"""


#Deterministic Growth Model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import root
import time 

font = {'family':'DejaVu Sans',
        'weight':'normal',
        'size'   : 30}

mpl.rc('font', **font) 

#options 
simulate = 1
plot_pol = 1
plot_sim = 1

#max iter
max_iter = 300

#global parameters 
beta = 0.96
delta = 0.08 

#utility function 
sigma = 1

if sigma == 1: 
    def u(c):
        if c == 0:
            return -np.inf
        else: 
            return np.log(c)
else: 
    def u(c): 
        if sigma >= 1 and c == 0:
            return -np.inf
        else:
            return (np.power(c,np.float(1-sigma))-1)/(1-sigma)

def u1(c):
    return np.power(c,-np.float(sigma))

#production function 
alpha = 1/3
def f(k): 
    return np.power(k,np.float(alpha))
def f1(k): 
    if k <= 0:
        return np.inf
    else:
        return alpha*np.power(k,np.float(alpha-1))

# borrowing constraint 
kmin = 0

r = 0.03 # asset return 
R = 1+r
A = 1
kmax = 30
prec = 1
num = prec*(kmax-kmin)
gridpar = 0.4

k = np.linspace(0,1, num=np.int(num))
k = np.power(k, 1/gridpar)
k = kmin + (kmax-kmin)*k

conguess = np.zeros(k.shape[0])

for ia in range(k.shape[0]):
    conguess[ia] = A*f1(k[ia])-delta*k[ia]

emuc = u1(conguess)
c = conguess

i = 0 
maxdiff = 1 

def FOC(y):
    if cash-kmin <= y:
        return u1(cash - kmin) - beta*(A*f1(kmin)+1-delta)*np.interp(kmin, k, emuc)
    else:
        return u1(cash-y) - beta*(A*f1(y)+1-delta)*np.interp(y, k, emuc)

k1 = np.zeros(k.shape[0])

#core calculation process 
print("calculating")
a = time.time()
while i <= max_iter: 
    i += 1 
    for ia in range(k.shape[0]):
        cash = A*f(k[ia])+(1-delta)*k[ia]
        if u1(cash - kmin) > beta*(A*f1(kmin)+1-delta)*np.interp(kmin, k, emuc): #has to be >= not =
            k1[ia] = kmin
        else:
            k1[ia] = root(FOC, 0.5*cash).x
        c[ia] = cash - k1[ia]
    emuc = u1(c)
    #print("iter #%d"%(i))
b = time.time()
print("calculation completed, time taken: %.2f s"%(b-a))
sav = k1-k

#simulation
if simulate == 1:

    T = 100
    z = 5 # different starting points
    ksimmin = 1
    ksimmax = 9

    ksimmin = np.max([kmin, ksimmin])
    k0sim = np.linspace(ksimmin, ksimmax, num=z)
    ksim = np.zeros([z, T])
    csim = np.zeros([z, T])

    ksim[:,0] = k0sim
    csim[:,0] = np.interp(ksim[:,0], k, c)

    print("Simulation begins")
    a = time.time()
    for t in range(1, T):
        ksim[:, t] = np.interp(ksim[:,t-1], k, k1)
        csim[:, t] = np.interp(ksim[:,t], k, c)
    b = time.time()
    print("Simulation complete, time taken: %.2f s"%(b-a))

print("Steady State Estimate: c = %.4f, k = %.4f"%(csim[0,T-1], ksim[0, T-1]))

#plot
"""
plt.plot(k, c, color = 'red', linewidth = 5, label = 'Saddle Path')
plt.plot(4.535, 1.292, 'ro', markersize=15, label = 'Steady State')
plt.xlabel("Assets")
plt.ylabel("Consumption")
plt.grid(which = 'both')
plt.xlim([0,10])
plt.ylim([0,3])
plt.legend()
plt.title('Growth Model Saddle Path')
"""

def plot(plot_sim, plot_pol):
    if plot_pol == 1 and plot_sim == 0:
        fig, ax = plt.subplots(1, 2)
        plt.suptitle("Deterministic Growth Model")
        ax[0].plot(k, c, color = 'black', linewidth = 5, label = 'Saddle Path')
        ax[0].plot(ksim[0, T-1], csim[0, T-1], 'ro', markersize=15, label = 'Steady State')
        ax[0].set_xlabel("Capital")
        ax[0].set_ylabel("Consumption")
        ax[0].grid(which = 'both')
        ax[0].set_xlim([0,10])
        ax[0].set_ylim([0,3])
        ax[0].legend()
        ax[0].set_title('Consumption Policy/Saddle Path')
        
        ax[1].plot(k, sav, color = 'black', linewidth = 5, label = 'Change in Capital')
        ax[1].set_xlabel("Capital")
        ax[1].set_ylabel("Change in Capital")
        ax[1].grid(which = 'both')
        ax[1].plot(ksim[0, T-1], 0, 'ro', markersize=15, label = 'Steady State')
        ax[1].set_xlim([0,10])
        ax[1].set_ylim([-1,0.5])
        ax[1].set_title('Policy for Change in Capital')
        ax[1].legend()
    elif plot_sim == 1 and plot_pol == 0:
        fig, ax = plt.subplots(1, 2)
        plt.suptitle("Convergence Path") 
        for i in range(z):
            ax[0].plot(csim[z-1-i], linewidth = 5, label = 'k0 = %.1f'%(ksim[z-1-i, 0]))
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Capital")
        ax[0].grid(which = 'both')
        ax[0].set_title('Consumption Convergence Path')
        ax[0].set_xlim([0,60])
        ax[0].legend()
        for i in range(z):
            ax[1].plot(ksim[i], linewidth = 5, label = 'k0 = %.1f'%(ksim[z-1-i, 0]))
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Capital")
        ax[1].grid(which = 'both')
        ax[1].set_title('Capital Convergence Path')
        ax[1].set_xlim([0,60])
        ax[1].legend()
    else:   
        fig, ax = plt.subplots(2,2)
        plt.suptitle("Deterministic Growth Model")
        ax[0,0].plot(k, c, color = 'black', linewidth = 5, label = 'Saddle Path')
        ax[0,0].plot(ksim[0, T-1], csim[0, T-1], 'ro', markersize=15, label = 'Steady State')
        ax[0,0].set_xlabel("Capital")
        ax[0,0].set_ylabel("Consumption")
        ax[0,0].grid(which = 'both')
        ax[0,0].set_xlim([0,10])
        ax[0,0].set_ylim([0,3])
        ax[0,0].legend()
        ax[0,0].set_title('Consumption Policy/Saddle Path')
        
        ax[0,1].plot(k, sav, color = 'black', linewidth = 5, label = 'Change in Capital')
        ax[0,1].set_xlabel("Capital")
        ax[0,1].set_ylabel("Change in Capital")
        ax[0,1].plot(ksim[0, T-1], 0, 'ro', markersize=15, label = 'Steady State')
        ax[0,1].grid(which = 'both')
        ax[0,1].set_xlim([0,10])
        ax[0,1].set_ylim([-1,0.5])
        ax[0,1].set_title('Policy for Change in Capital')
        ax[0,1].legend()
        
        for i in range(z):
            ax[1,0].plot(csim[z-1-i], linewidth = 5, label = 'k0 = %.1f'%(ksim[z-1-i, 0]))
        ax[1,0].set_xlabel("Time")
        ax[1,0].set_ylabel("Capital")
        ax[1,0].grid(which = 'both')
        ax[1,0].set_title('Consumption Convergence Path')
        ax[1,0].set_xlim([0,50])
        ax[1,0].legend()
        for i in range(z):
            ax[1,1].plot(ksim[z-1-i], linewidth = 5, label = 'k0 = %.1f'%(ksim[z-1-i, 0]))
        ax[1,1].set_xlabel("Time")
        ax[1,1].set_ylabel("Capital")
        ax[1,1].grid(which = 'both')
        ax[1,1].set_title('Capital Convergence Path')
        ax[1,1].set_xlim([0,50])
        ax[1,1].legend()
        
plot(plot_sim, plot_pol)