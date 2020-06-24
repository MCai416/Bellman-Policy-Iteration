# Bellman-Policy-Iteration
also known as Euler Equation Iteration (Moll), Time Iteration (Sargent), based on Kaplan's codes, written using Python 3

There are 3+1 .py files, which has slightly different models

EEI_AG_IID looks at the consumption/savings policy (supply side of the capital market) of an income fluctuation model, which is the capital supply side of the Aiyagari model, with iid income distribution, allows savings with a constant rate of return, code written based on Kaplan's Matlab codes 

EEI_AG_Markov modifies the iid model to a finite state Markov process, which allows for responses to be conditional on past states, more lags are also possible by making a larget matrix, but it takes a lot of computational power. For example, the iid setting takes 4 seconds to compute, while Markov process takes more than 15 seconds. Meanwhile there isn't a lot of information added using Markov process. 

EEI_GM_Deter looks at the classical social planner's deterministic growth model, which is different from the income fluctuation model, you can compare the results between policy iteration in EEI_GM_Deter.png and bisection/shooting algorithm in the "Discrete Ramsey" repository. The picture results have the same axes settings.  

Lastly, although irrelevant, VFI_IID is the python code for Bellman Value Function Iteration under the same setup as EEI_IID. This algorithm is based on the contraction mapping of Bellman Operator. Although it is the easiest in theory, the actual performance is horrific. 
