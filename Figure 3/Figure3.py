#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johannes Wiesel
"""


import numpy as np
import matplotlib.pyplot as plt
import pylab
from plugin_github import plugin_gurobi


if __name__ == '__main__':
    # No of runs and samples
    runs=10**3
    N=10**4
    
    def g(x):
        return np.array([int(i <= 0.5) for i in x])

    #test exponential
    la=1
    def F(x,la):
        return -np.log(1-x)/la
    def F1(x):
        return F(x,la)
    
    val = plugin_gurobi(runs, N, F1, g)
    
    plt.figure(figsize=(16.000, 8.000), dpi=100)
    plt.loglog(range(0,N),val.mean(axis=1), label="Plugin")
    plt.loglog(range(0,N), val.mean(axis=1)+1/np.log(range(1,N+1)), label="Penalty")
    plt.plot(range(0,N), np.repeat(1,N), label="True Value")
    plt.legend(loc=4, prop={'size': 20})
    import matplotlib.ticker as ticker
    ax = plt.axes()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(plt.MaxNLocator(12))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    
    
    pylab.savefig('hyb_exp.pdf')
    plt.clf()
