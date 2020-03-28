# Bellman-Policy-Iteration
also known as Euler Equation Iteratoin (Moll), Time Iteration (Sargent), based on Kaplan's codes, written using Python 3

There are three .py files, which has slightly different models 

EEI_AG_IID looks at the consumption policy for an income fluctuation model (also known as the Aiyagari model) with iid income distribution, allows savings with a constant rate of return, code written based on Kaplan's Matlab codes 

EEI_AG_Markov modifies the above model to a finite state Markov process, which allows for policies conditional on past probabilities 

EEI_GM_Deter looks at the classic deterministic growth model, which is different from the income fluctuation model, you can compare the results using policy iteration in EEI_GM_Deter.png and the result using bisection/shooting algorithm in the "Discrete Ramsey" repository 
