#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johannes Wiesel
"""

import numpy as np
from Plugin_estimator import plugin_gurobi
import matplotlib.pyplot as plt
import scipy.stats as sct
from scipy.optimize import minimize

def g(x):
    return np.maximum(x-2,0)

def percentile(a,p):
    return np.sort(a)[int(p*len(a))-1:]

#Approximation of Wasserstein estimator
def Wasserstein_approx(runs, N, F, g):
    res = np.zeros((N,runs))
    data = F(np.random.rand(N,runs))
    for j in range(0, runs):
        print(j)
        for i in range(1, N):
            cc= 1/np.power(i, 0.25)
            def avar_cpt(x):
                return np.mean(percentile(g(data[:i,j])-x[0]*(data[:i,j]-1), 1-cc))
            res[i,j] = minimize(avar_cpt, x0=[0], method='BFGS').fun+ +2*cc
            res[res<0] = 2
    return(res)
    
if __name__ == '__main__':

    # Runs and plots
    runs = 10**3
    N = 10**5
    
    def g(x):
        return(np.maximum(x-2,0))
    #test log-normal
    def F1(x):
        return np.exp(sct.norm.ppf(x))
    plugin_estimator = plugin_gurobi(runs, N, F1, g)
    wasserstein_estimator = Wasserstein_approx(runs, N, F1, g)
    
    plt.figure(figsize=(16.000, 8.000), dpi=100)
    plt.loglog(range(0,N), plugin_estimator.mean(axis=1), label="Plugin")
    plt.loglog(range(0,N), wasserstein_estimator.mean(axis=1), label="Wasserstein")
    plt.loglog(range(0,N), np.repeat(1,N), label="True Value")
    plt.legend()
    plt.tight_layout()
