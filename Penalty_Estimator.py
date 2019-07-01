import numpy as np
from scipy.optimize import * 
from plugin_github import plugin_gurobi
import matplotlib.pyplot as plt


def ind(x,eps):
    return(np.max(x-eps,0)+np.min(x+eps,0))

def hybrid_opt(runs, N, F, g, g_max):
    res = np.zeros((N,runs))
    
    for j in range(runs):
        data =  F(np.random.random_sample([N, 1]))
            
        for i in range(1,N):
            count = 0
            aux_s = False
            
            temp = np.append(data[0:i], np.linspace(0, max(data[0:i]), num=max(10,i)))
            c = g(temp)
            x0 = [0]*len(temp)
        
            if max(temp)>1 and min(data[0:i])<1:
                eps = 0.01/2/len(temp)
                def cpt(x):
                    return -(np.dot(c,x[:len(c)])-g_max*(1+g_max/i)*(max(np.append(x[len(c):], np.zeros(len(c)-i))/np.maximum(x[:len(c)], eps))-1))
                constr_prob = LinearConstraint(np.append(np.ones(len(c)), np.zeros(i)), 1-eps, 1+eps)
                constr_prob_a =  LinearConstraint(np.append(np.zeros(len(temp)), np.ones(i)), 1-eps, 1+eps)
                constr_mart = LinearConstraint(np.append(temp, np.zeros(i)), 1-eps, 1+eps)
                constr_mart_a = LinearConstraint(np.append(np.zeros(len(c)), temp[0:i]), 1-eps, 1+eps)
                constr_prob_b = NonlinearConstraint(min, -eps,1+eps)
            
                if sum(x0) == 0:
                    #Initialise x0
                    x0[i] = 0
                    x0[-1] = 0
                    x0[i] = 1-sum(x0)+(np.dot(x0, temp)-1)/temp[-1]
                    x0[-1] = 1-sum(x0)
                    
                    x1 = [0] *i
                    ind1 = np.argmin(data[0:i])
                    ind2 = np.argmax(data[0:i])
                    x1[ind1] = 0
                    x1[ind2] = 0
                    x1[ind1] = (1-np.dot(x1, temp[0:i])-temp[ind2]*(1-sum(x1)))  /(temp[ind1]-temp[ind2])
                    x1[ind2] = 1-sum(x1)
                    
                    x0 = np.append(x0,x1)
                else:
                    #Use z as seed
                    a = np.random.rand()*eps/len(temp)/max(temp)
                    if i>= 10:
                        x0 = np.append(np.append(np.append(z[:i],a),z[i:len(c)]), np.append(z[len(c):],a))
                    else:
                        x0 = np.append(np.append(np.append(z[:i],a),np.append(z[i:len(c)],a)), np.append(z[len(c):],a))
                    
                while aux_s== False and count<= 10**2:
                    print(j,i,count)
                    z = x0 +(-0.5+np.random.rand())*eps/len(temp)/max(temp)
                    aux = minimize(cpt, x0=z,method='SLSQP', constraints= {constr_prob, constr_prob_a, constr_mart,
                                                                           constr_mart_a, constr_prob_b}, 
                                   tol=1e-02, options={"maxiter": 1000000})
                    aux_s = aux.success
                    count += 1
                if aux_s == True and aux.fun >= -g_max and aux.fun <= 2 *g_max: 
                        res[i,j] = -aux.fun
                else:
                    res[i,j] = np.nan
                print(res[i,j])
    return(res)
    


if __name__ == '__main__':
    N = 2*10 ** 1
    runs = 2*10 ** 1
    
    #test exponential
    la=1
    def F(x,la):
        return -np.log(1-x)/la
    def F1(x):
        return F(x,la)
    
    def g(x):
        return np.array([int(i <= 0.5) for i in x])

    
    plugin = plugin_gurobi(runs, N, F1, g)
    hybrid = hybrid_opt(runs, N, F1, g, 1)
    
    
    #Plot
    plt.loglog(range(0,N), plugin.mean(axis=1), label="Plugin")
    plt.plot(np.nanmean(hybrid,axis=1))
    plt.plot(range(0,N), np.repeat(1,N), label="True Value")
    plt.legend()
