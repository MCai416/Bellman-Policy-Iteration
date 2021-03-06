# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:43:58 2020

@author: Ming Cai
"""

#Aiyagari Model Demand Side, Kaplan's Code 

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

if plot_pol == 1 and plot_sim == 1:
    plot_all = 1
else:
    plot_all = 0

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
kmin = -0.3

r = 0.03 # asset return 
R = 1+r
y = np.array([0.6133, 0.8066, 1, 1.1934, 1.3867])
p = np.array([0.0735, 0.2409, 0.3712, 0.2409, 0.0735])
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

emuc = np.matmul(p, u1(conguess))

c = conguess

i = 0 
max_iter = 200
maxdiff = 1 

def FOC(y):
    if cash-kmin <= y:
        return u1(cash - kmin) - beta*R*np.interp(kmin, k, emuc)
    else:
        return u1(cash-y) - beta*R*np.interp(y, k, emuc)

k1 = np.zeros([y.shape[0], k.shape[0]])

print("calculating")
a = time.time()
while i <= max_iter: 
    i += 1 
    for ia in range(k.shape[0]):
        for iy in range(y.shape[0]):
            cash = R*k[ia] + y[iy]
            if u1(cash - kmin) > beta*R*np.interp(kmin, k, emuc): #has to be >= not =
                k1[iy, ia] = kmin
            else:
                k1[iy, ia] = root(FOC, 0.5*cash).x
            c[iy, ia] = cash - k1[iy, ia]
    emuc = np.matmul(p, u1(c))
    #print("iter #%d"%(i))
b = time.time()
print("calculation completed, time taken: %.2f"%(b-a))
sav = k1-k

if plot_pol == 1 and plot_all == 0:
    fig, ax = plt.subplots(1, 2)
    plt.suptitle("Euler Equation Iteration IID Income")
    ax[0].set_title("Policy for Consumption")
    ax[0].plot(k, c[y.shape[0]-1], linewidth = 5, color = 'red', label = 'highest income')
    ax[0].plot(k, c[0], linewidth = 5, color = 'blue', label = 'lowest income')
    ax[0].set_xlabel("Assets")
    ax[0].set_ylabel("Consumption")
    ax[0].legend()
    ax[0].grid(which='both')
    
    ax[1].set_title("Policy for Change in Assets")
    ax[1].plot(k, sav[0], linewidth = 5, color = 'blue', label = 'lowest income')
    ax[1].plot(k, sav[y.shape[0]-1], linewidth = 5, color = 'red', label = 'highest income')
    ax[1].set_xlabel("Assets")
    ax[1].set_ylabel("Savings")
    ax[1].grid(which='both')
    ax[1].legend()

#wealth distribution 
if simulate == 1:
    T = 100
    N = 10000
    k0 = 0.1

    ksim = np.zeros([N,T])
    csim = np.zeros([N,T])
    ssim = np.zeros([N,T])
    print("simulating")
    a = time.time()
    for i in range(N):
        state_sim = np.random.choice(np.arange(0,y.shape[0]), size = T, p = p)
        csim[i,0] = np.interp(k0, k, c[state_sim[0]])
        ksim[i,0] = np.interp(k0, k, k1[state_sim[0]])

        for t in range(1, T): 
            csim[i,t] = np.interp(ksim[i, t-1], k, c[state_sim[t]])
            ksim[i,t] = np.interp(ksim[i, t-1], k, k1[state_sim[t]])
    b = time.time()
    print("simulation completed, time taken: %.2f s"%(b-a))
    
    kdist = ksim[:, T-1]
    Kt = np.mean(ksim, axis = 0)
    Ct = np.mean(csim, axis = 0)
    kss_guess = Kt[T-1]
    css_guess = Ct[T-1]
    print("Steady state estimate: c = %.2f, k = %.2f"%(css_guess, kss_guess))
    
    if plot_sim == 1 and plot_all == 0:
        fig1, ax1 = plt.subplots(1, 2)
        plt.suptitle("Asset distribution and convergence")
        ax1[0].hist(kdist, bins = 50, weights = np.ones(N)/N)
        ax1[0].set_xlabel("Asset")
        ax1[0].set_ylabel("Probability")
        ax1[0].set_title("Stationary Assset Distribution")
        ax1[0].grid(which='both')
        ax1[1].plot(Kt, color = 'black', linewidth = 5)
        ax1[1].set_xlabel("Time")
        ax1[1].set_ylabel("Mean Asset")
        ax1[1].set_title("Mean Asset Convergence")
        ax1[1].grid(which='both')
    
if plot_all == 1:
    fig, ax = plt.subplots(2, 2)
    plt.suptitle("Euler Equation Iteration IID Income")
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
    ax[0, 1].set_ylabel("Savings")
    ax[0, 1].grid(which='both')
    ax[0, 1].legend()
    
    ax[1, 0].hist(kdist, bins = 50, weights = np.ones(N)/N)
    ax[1, 0].set_xlabel("Assets")
    ax[1, 0].set_ylabel("Probability")
    ax[1, 0].set_title("Stationary Assset Distribution")
    ax[1, 0].grid(which='both')
    
    ax[1, 1].plot(Kt, color = 'black', linewidth = 5)
    ax[1, 1].set_xlabel("Time")
    ax[1, 1].set_ylabel("Mean Assets")
    ax[1, 1].set_title("Mean Asset Convergence")
    ax[1, 1].grid(which='both')