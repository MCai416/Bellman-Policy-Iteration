# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:55:49 2020

@author: Ming Cai
"""

#Aiyagari Model with Markov Process

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

#iteration number
num_iter = 2

#global parameters 
beta = 0.95
delta = 0.08 

#utility function 
sigma = 2

if sigma == 1: 
    def u(c):
        if c == 0:
            return -np.inf
        else: 
            return np.log(c)
else: 
    def u(c): 
        if sigma > 1 and c == 0:
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
    if k == 0:
        return np.inf
    else:
        return alpha*np.power(k,np.float(alpha-1))

# borrowing constraint 
kmin = 0

r = 0.03 # asset return 
R = 1+r
y = np.array([0.8, 1, 1.2])
rho = 0.7
P = np.array([[rho, 1-rho, 0],[(1-rho)/2, rho, (1-rho)/2],[0, 1-rho, rho]])
Pt = P.transpose()
kmax = 30
prec = 1
num = prec*(kmax-kmin)
gridpar = 0.4

k = np.linspace(0,1, num=np.int(num))
k = np.power(k, 1/gridpar)
k = kmin + (kmax-kmin)*k

conguess = np.zeros([y.shape[0], k.shape[0]])

for i in range(y.shape[0]):
    conguess[i] = r*k + y[0]

emuc = np.matmul(Pt, u1(conguess))

c = conguess

i = 0 
max_iter = 200
maxdiff = 1 

def FOC(y):
    if cash-kmin <= y:
        return u1(cash - kmin) - beta*R*np.interp(kmin, k, emuc[iy])
    else:
        return u1(cash-y) - beta*R*np.interp(y, k, emuc[iy])

k1 = np.zeros([y.shape[0], k.shape[0]])

#core calculation process 
print("calculating")
a = time.time()
while i <= max_iter: 
    i += 1 
    for ia in range(k.shape[0]):
        for iy in range(y.shape[0]):
            cash = R*k[ia] + y[iy]
            if u1(cash - kmin) > beta*R*np.interp(kmin, k, emuc[iy]): #has to be >= not =
                k1[iy, ia] = kmin
            else:
                k1[iy, ia] = root(FOC, 0.5*cash).x
            c[iy, ia] = cash - k1[iy, ia]
    emuc = np.matmul(P, u1(c))
    #print("iter #%d"%(i))
b = time.time()
print("calculation completed, time taken: %.2f s"%(b-a))
sav = k1-k

#simulation
if simulate == 1:
    T = 100
    N = 10000
    k0 = 0

    ksim0 = np.zeros([N,T])
    csim0 = np.zeros([N,T])
    ksim1 = np.zeros([N,T])
    csim1 = np.zeros([N,T])
    
    dist0 = np.zeros([T, y.shape[0]])
    dist1 = np.zeros([T, y.shape[0]])
    dist0[0, 0] = 1
    dist1[0, y.shape[0]-1] = 1
    for t in range(1, T):
       dist0[t] = np.matmul(Pt, dist0[t-1])
       dist1[t] = np.matmul(Pt, dist1[t-1])
    
    print("simulating")
    a = time.time()
    
    state_sim0 = np.random.choice(np.arange(0,y.shape[0]), p = dist0[0])
    state_sim1 = np.random.choice(np.arange(0,y.shape[0]), p = dist1[0])
    for i in range(N):
        csim0[i,0] = np.interp(k0, k, c[state_sim0])
        ksim0[i,0] = np.interp(k0, k, k1[state_sim0])
        csim1[i,0] = np.interp(k0, k, c[state_sim1])
        ksim1[i,0] = np.interp(k0, k, k1[state_sim0])

        for t in range(1, T): 
            state_sim0 = np.random.choice(np.arange(0,y.shape[0]), p = dist0[t])
            state_sim1 = np.random.choice(np.arange(0,y.shape[0]), p = dist1[t])
            csim0[i,t] = np.interp(ksim0[i, t-1], k, c[state_sim0])
            ksim0[i,t] = np.interp(ksim0[i, t-1], k, k1[state_sim0])
            csim1[i,t] = np.interp(ksim1[i, t-1], k, c[state_sim1])
            ksim1[i,t] = np.interp(ksim1[i, t-1], k, k1[state_sim1])
    
    b = time.time()
    print("simulation completed, time taken: %.2f s"%(b-a))
    
    kdist = ksim0[:, T-1]
    Kt0 = np.mean(ksim0, axis = 0)
    Kt1 = np.mean(ksim1, axis = 0)
    Ct = np.mean(csim0, axis = 0)
    kss_guess = Kt0[T-1]
    css_guess = Ct[T-1]
    print("Steady state estimate: c = %.2f, k = %.2f"%(css_guess, kss_guess))
    
def plot(plot_sim, plot_pol):
    if simulate == 1 and plot_pol == 1 and plot_sim == 1:
        plot_all = 1
    else:
        plot_all = 0
    if plot_pol == 1 and plot_all == 0:
        fig, ax = plt.subplots(1, 2)
        plt.suptitle("Euler Equation Iteration Markov Process Income")
        ax[0].set_title("Consumption Policy")
        ax[0].plot(k, c[y.shape[0]-1], linewidth = 5, color = 'red', label = 'highest income')
        ax[0].plot(k, c[0], linewidth = 5, color = 'blue', label = 'lowest income')
        ax[0].set_xlabel("Assets")
        ax[0].set_ylabel("Consumption")
        ax[0].legend()
        ax[0].grid(which='both')
        
        ax[1].set_title("Policy for Change in Assets")
        ax[1].plot(k, sav[y.shape[0]-1], linewidth = 5, color = 'red', label = 'highest income')
        ax[1].plot(k, sav[0], linewidth = 5, color = 'blue', label = 'lowest income')
        ax[1].set_xlabel("Assets")
        ax[1].set_ylabel("Change in Assets")
        ax[1].grid(which='both')
        ax[1].legend()
    if plot_sim == 1 and plot_all == 0:
        fig1, ax1 = plt.subplots(1, 2)
        plt.suptitle("Asset distribution and convergence")
        ax1[0].hist(kdist, bins = 50, weights = np.ones(N)/N)
        ax1[0].set_xlabel("Asset")
        ax1[0].set_ylabel("Probability")
        ax1[0].set_title("Stationary Assset Distribution")
        ax1[0].grid(which='both')
        ax1[1].plot(Kt1, color = 'red', linewidth = 5, label = 'highest income path')
        ax1[1].plot(Kt0, color = 'blue', linewidth = 5, label = 'lowest income path')
        ax1[1].set_xlabel("Time")
        ax1[1].set_ylabel("Mean Asset")
        ax1[1].set_title("Mean Asset Convergence")
        ax1[1].grid(which='both')
        ax1[1].legend()
        
    if plot_all == 1:
        fig, ax = plt.subplots(2, 2)
        plt.suptitle("Euler Equation Iteration Markov Process Income")
        ax[0, 0].set_title("Policy for Consumption")
        ax[0, 0].plot(k, c[y.shape[0]-1], linewidth = 5, color = 'red', label = 'highest income')
        ax[0, 0].plot(k, c[0], linewidth = 5, color = 'blue', label = 'lowest income')
        ax[0, 0].set_xlabel("Assets")
        ax[0, 0].set_ylabel("Consumption")
        ax[0, 0].legend()
        ax[0, 0].grid(which='both')
        
        ax[0, 1].set_title("Policy for Change in Assets")
        ax[0, 1].plot(k, sav[y.shape[0]-1], linewidth = 5, color = 'red', label = 'highest income')
        ax[0, 1].plot(k, sav[0], linewidth = 5, color = 'blue', label = 'lowest income')
        ax[0, 1].set_xlabel("Assets")
        ax[0, 1].set_ylabel("Change in Assets")
        ax[0, 1].grid(which='both')
        ax[0, 1].legend()
        
        ax[1, 0].hist(kdist, bins = 50, weights = np.ones(N)/N)
        ax[1, 0].set_xlabel("Assets")
        ax[1, 0].set_ylabel("Probability")
        ax[1, 0].set_title("Stationary Assset Distribution")
        ax[1, 0].grid(which='both')
        
        ax[1, 1].plot(Kt1, color = 'red', linewidth = 5, label = 'highest income path')    
        ax[1, 1].plot(Kt0, color = 'blue', linewidth = 5, label = 'lowest income path')
        ax[1, 1].set_xlabel("Time")
        ax[1, 1].set_ylabel("Mean Assets")
        ax[1, 1].set_title("Mean Asset Convergence")
        ax[1, 1].legend()
        ax[1, 1].grid(which='both')
        
plot(plot_sim, plot_pol)