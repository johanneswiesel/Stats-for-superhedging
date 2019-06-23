#!/usr/bin/env python3
""
@author: Johannes Wiesel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab
import arch
from scipy.optimize import minimize
from Wasserstein_stats import calculate_wasserstein      

#Calculate AVAR-estimator directly
def percentile(a,p):
    return np.sort(a, axis=0)[int(p*len(a))-1:]

def avar(data, g, alpha):
    def avar_cpt(x):
        return np.mean(percentile(g(data)-x[0]*(data-1), alpha))
    res = minimize(avar_cpt,x0=[0], method='BFGS').fun
    return(res)
        
def simulate_garch(dates, garch_parameters, no_samples, starting_val=1): #simulates GARCH(1,1) with student-t distribution according to given dates, starting value 1
    omega = garch_parameters[0]
    alpha = garch_parameters[1] 
    beta = garch_parameters[2] 
    nu = garch_parameters[3]
    n = dates.shape[0]
    epsilon = np.zeros((n, no_samples))
    epsilon[0, :] = starting_val
    t_rand = np.sqrt((nu-2)/nu)*np.random.standard_t(df= nu, size=(n, no_samples))
    for i in range(0, no_samples):
        sigma2 = np.zeros(n)
        for k in range(1, n):
            sigma2[k] = omega + alpha*(epsilon[k-1,i]**2) + beta*sigma2[k-1]
            epsilon[k, i] = t_rand[k,i] * np.sqrt(sigma2[k])
    return(pd.DataFrame(data=epsilon, index=dates.index))
    
def estimate_garch(data, window): #estimate GARCH(1,1) model with running window, data Nx1 dimensional
    n = data.shape[0]
    garch_parameters = np.zeros((n, 4))
    for i in range(0, n-window+1):
        print(i)
        model = arch.arch_model(data.values[i:i+window],mean='Zero', vol='GARCH', dist='StudentsT')
        model_fit = model.fit(disp='off')
        garch_parameters[window-1+i,0] = model_fit.params[0]
        garch_parameters[window-1+i,1] = model_fit.params[1]
        garch_parameters[window-1+i,2] = model_fit.params[2]
        garch_parameters[window-1+i,3] = model_fit.params[3]
    res = pd.DataFrame(data = garch_parameters, index = data.index)
    return(res)
        
def monte_carlo_garch(data, garch_parameters, no_samples, no_mc, g, alpha):
    epsilon = np.zeros((no_mc, no_samples))
    sigma2 = np.zeros((no_mc, no_samples))
    N = data.shape[0]
    avar_out =  pd.DataFrame(data=np.zeros((N, no_samples)), index=data.index)
    for i in range(0,N):
        print(i)
        epsilon[0,:] = 1
        omega = garch_parameters.iloc[i,0]
        gamma = garch_parameters.iloc[i,1] 
        beta = garch_parameters.iloc[i,2] 
        nu = garch_parameters.iloc[i,3]
        if omega == 0 or gamma == 0 or beta == 0 or nu == 0: continue
        t_rand = np.random.standard_t(df= nu, size=(no_mc, no_samples))
        for k in range(1, no_mc):
            sigma2[k, :] = omega + gamma*(epsilon[k-1, :]**2) + beta*sigma2[k-1, :]
            epsilon[k, :] = t_rand[k, :] * np.sqrt(sigma2[k,:])
        returns = np.exp(epsilon/100)
        for j in range(0, no_samples):
            avar_out.iloc[i, j] = avar(returns[:, j],g,alpha)
    return(avar_out)


if __name__ == '__main__':

    ###########################################################
    # Test with simulated real data - constant GARCH(1,1) model
    garch_params = [0.02, 0.1, 0.83, 5]
    N = 10**6 #Data points to calculate true value
    dates = pd.DataFrame(index=range(0,N))
    garch_simulation = simulate_garch(dates, garch_params, 1)
    returns_garch_simulation = np.exp(garch_simulation/100)
    #plt.plot(returns_garch_simulation)
    
    #check model
    model = arch.arch_model(garch_simulation,mean='Zero', vol='GARCH', dist='StudentsT')
    model_fit = model.fit()
    #print(model_fit)
    ####
    
    #Concrete application 
    def g(x):
        return(np.maximum(x-1,0))
    alpha = 0.95
    interval = 50
    
    #Convention price S0=1
    true_value = avar(returns_garch_simulation,g, alpha)
    
    #Rolling estimation
    n = 52#10**3 # length of simulated data
    dates_est = pd.DataFrame(index=range(0,n))
    garch_samples_est = simulate_garch(dates_est, garch_params, 1)
    returns_garch_est = np.exp(garch_samples_est/100)
    #plt.plot(returns_garch_est)
    
    #Plugin estimator on given data
    def avar_concrete(data):
        return (avar(data,g,alpha))
    plugin_avar = returns_garch_est.rolling(interval).apply(avar_concrete)
    
    #Wasserstein estimator
    def g_wass(r):
            return tf.nn.relu(r-1)
    eps_const = 0.5
    def wasserstein_concrete(data):
        return (calculate_wasserstein(data, eps_const, alpha, g_wass))
    
    wasserstein_avar = returns_garch_est.rolling(interval).apply(wasserstein_concrete)
    
    #Monte Carlo GARCH(1,1)
    no_samples = 10**2
    no_mc = 10**2
    garch_params_est = estimate_garch(garch_samples_est, interval)
    garch_avar = monte_carlo_garch(dates_est, garch_params_est, no_samples, no_mc, g, alpha)
    garch_avar[garch_avar < 0 ] = 0
    garch_avar[ garch_avar > 0.04] = 0.04
    
    #Plot
    plt.figure(figsize=(16.000, 8.000), dpi=100)
    plt.plot(pd.DataFrame(data=np.repeat(true_value, plugin_avar.shape[0]), index=plugin_avar.index), label='True Value', linewidth = 3)
    plt.plot(plugin_avar, label='Plugin historical', linewidth=3)
    plt.plot(wasserstein_avar, label = 'Wasserstein historical', linewidth=3)
    plt.plot((garch_avar.mean(axis=1)).rolling(2).mean(), label = 'Plugin GARCH(1,1)')
    plt.legend()
    pylab.savefig('garch_fixed_par.pdf')
    
    #########################################################
    #Test with simulated data - Changing GARCH(1,1) model
    garch_params = [0.02, 0.1, 0.8, 5]
    garch_params2 = [0.05, 0.14, 0.83, 20]
    N = 10**6 #Data points to calculate true value
    dates = pd.DataFrame(index=range(0,N))
    garch_simulation2 = simulate_garch(dates, garch_params2, 1)
    returns_garch_simulation2 = np.exp(garch_simulation2/100)
    #plt.plot(returns_garch_simulation2)
    
    #check model
    model2 = arch.arch_model(garch_simulation2,mean='Zero', vol='GARCH', dist='StudentsT')
    model_fit2 = model2.fit()
    #print(model_fit)
    ####
    
    #Concrete application 
    def g(x):
        return(np.maximum(x-1,0))
    alpha = 0.95
    interval = 50
    
    #Convention price S0=1
    true_value2 = avar(returns_garch_simulation2,g, alpha)
    
    #Rolling estimation
    n = 10**3 # length of simulated data
    dates_est2 = pd.DataFrame(index=range(0,n))
    garch_samples_est2 = simulate_garch(dates_est, garch_params, 1, starting_val=1)
    garch_samples_est_temp = simulate_garch(dates_est2, garch_params2, 1, starting_val=garch_samples_est2.iloc[n//3-1])
    garch_samples_est2.iloc[n//3:2*n//3] = garch_samples_est_temp.iloc[0:n//3].values
    garch_samples_est_temp = simulate_garch(dates_est2, garch_params2, 1, starting_val=garch_samples_est2.iloc[2*n//3-1])
    garch_samples_est2.iloc[2*n//3:] = garch_samples_est_temp.iloc[0:n//3+1].values
    returns_garch_est2 = np.exp(garch_samples_est2/100)
    #plt.plot(returns_garch_est)
    
    #Plugin estimator on given data
    def avar_concrete(data):
        return (avar(data,g,alpha))
    plugin_avar2 = returns_garch_est2.rolling(interval).apply(avar_concrete)
    
    #Wasserstein estimator
    def g_wass(r):
            return tf.nn.relu(r-1)
    eps_const = 0.5
    def wasserstein_concrete(data):
        return (calculate_wasserstein(data, eps_const, alpha, g_wass))
    
    wasserstein_avar2 = returns_garch_est2.rolling(interval).apply(wasserstein_concrete)
    
    #Monte Carlo GARCH(1,1)
    no_samples = 10**2
    no_mc = 10**2
    garch_params_est2 = estimate_garch(garch_samples_est2, interval)
    garch_avar2 = monte_carlo_garch(dates_est2, garch_params_est2, no_samples, no_mc, g, alpha)
    garch_avar2[garch_avar2 < 0 ] = 0
    garch_avar2[ garch_avar2 > 0.1] = 0.1
    
    #Plot
    plt.figure(figsize=(16.000, 8.000), dpi=100)
    true_value_mat = np.append(np.append(np.repeat(true_value,n//3), np.repeat(true_value2, n//3)),np.repeat(true_value,n//3+1))
    plt.plot(pd.DataFrame(data = true_value_mat, index = plugin_avar.index), label = 'True value', linewidth =3)
    plt.plot(plugin_avar2, label = 'Plugin historical', linewidth = 3)
    plt.plot((garch_avar2.mean(axis=1)).rolling(2).mean(), label = 'Plugin GARCH(1,1)')
    plt.plot(wasserstein_avar2, label = 'Wasserstein historical', linewidth=3)
    plt.legend()
    pylab.savefig('garch_changing_par.pdf')
    
    #storing values
    import pickle
    
    f = open('avar_empiricial_garch.pckl', 'wb')
    pickle.dump([true_value, true_value2, garch_avar, plugin_avar, garch_avar2, plugin_avar2,
                 wasserstein_avar, wasserstein_avar2], f)
    f.close()
    
    f = open('avar_empiricial_garch.pckl', 'rb')
    [true_value, true_value2, garch_avar, plugin_avar, garch_avar2, plugin_avar2,
     wasserstein_avar,wasserstein_avar2] = pickle.load(f)
    f.close()
    
