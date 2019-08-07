#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Johannes Wiesel 
"""

import numpy as np
from gurobipy import *
import matplotlib.pyplot as plt
import pylab

# Plugin estimator Gurobi version
def plugin_gurobi(runs, N, F,g):
    no = range(0, runs)
    ind = range(0, N, 1)
    res = np.zeros((N, runs))
    for j in no:
        print(j)
        data =  F(np.random.random_sample([N, 1]))
        
        # Set up gurobi model
        model = Model()
        model.setParam( 'OutputFlag', False )
        
        model.addVar(obj = 1)
        for l in range(1, data.shape[1]+1):
            model.addVar(obj = 0)
        model.update()
        vars = model.getVars()
         
        # Initialise first 10 iterations
        for k in range(10):
            expr = LinExpr()
            expr += -vars[0]
            for l in range(1, data.shape[1]+1):
                expr += -(data[k, l-1]-1)*vars[l]
            model.addConstr(lhs=expr, sense=GRB.LESS_EQUAL, rhs=-g(data[k, :]))
        
        model.update()
    
        # Run optimisation
        for i in ind:
            expr = LinExpr()
            expr += -vars[0]
            for l in range(1,data.shape[1]+1):
                expr += -(data[i, l-1]-1)*vars[l]
            model.addConstr(lhs=expr, sense=GRB.LESS_EQUAL, rhs=-g(data[i, :]))
            model.update()
            model.optimize()
            res[i,j] = model.ObjVal
            
    return(res)
    

if __name__ == '__main__':

    # Define functions g
    def g(x):
        return abs(x-1)
    
    # Number of runs and plots
    runs = 10**2
    N = 10**2
    
    #test uniform with thinner tails
    def F(x,n):
        return (np.power(2*x,1/(n+1)))*(x<0.5)+(2-np.power(2*(1-x),1/(n+1)))*(x>=0.5)
    n = 1
    def F1(x):
        return F(x,n)
    
    plugin = plugin_gurobi(runs,N,F1,g)
    
    plt.figure(figsize=(16.000, 8.000), dpi=100)
    plt.plot(range(0,N), plugin.mean(axis=1))
    
    
