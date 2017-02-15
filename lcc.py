#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:38:55 2017

@author: paopow
"""

# %%
import autograd.numpy as np
from autograd import value_and_grad
from itertools import combinations
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# %%
D = 2
# %%    
def get_Y(clusters, n):
    Y = np.zeros((n,n))
    seen_so_far = set()
    for c in clusters:
        for (i,j) in combinations(c,2):
            Y[i,j] = 1
            Y[j,i] = 1
        for i in c:
            for s in seen_so_far:
                Y[s,i] = -1
                Y[i,s] = -1
        seen_so_far.update(c)        
    return Y
 
Y = get_Y([[0,1],[2,3]], 4)

# %%
def loss_M(Y, X, U, b):
    mask_pos = Y==1
    mask_neg = Y==-1
    tm = np.sum(mask_pos)/np.sum(mask_pos + mask_neg)
    Beta = (mask_pos)*1 + (mask_neg)*tm
    F = np.dot(np.dot(X,U),(X.T)) + b
    diff = (F - Y)
    sum_loss = np.sum(Beta*(diff**2))  
    return 0.5 * sum_loss
    
# %%
def loss_all(Ys, X, Us, b, lambda_x = 1.,lambda_u = 1., sigma_u = 0.5):
    sum_loss_M = 0
    sum_reg_u = 0
    num_user = int(Us.size/D)
    for i in range(num_user):
        Y = Ys[i]
        U = np.make_diagonal(Us[D*i:D*(i+1)], axis1=-1, axis2=-2)
        sum_loss_M += loss_M(Y, X, U, b)
        diff_U = U - sigma_u * np.eye(U.shape[0])
        sum_reg_u += np.linalg.norm(diff_U,ord='fro')**2

    Rx = lambda_x * np.sum((np.linalg.norm(X, axis = 1)**2))
    Ru = lambda_u * sum_reg_u
    return Rx + Ru + sum_loss_M
    
# %%
def paramsX_to_minimize(params, Ys, Us, b):
    X = params.reshape((int(params.size/D),D))
    return loss_all(Ys, X, Us, b)

paramsX_to_minimize_with_grad = value_and_grad(paramsX_to_minimize)  

# %%
def paramsUs_to_minimize(params, Ys, X, b):
    return loss_all(Ys, X, params, b)
    
paramsUs_to_minimize_with_grad = value_and_grad(paramsUs_to_minimize) 

# %%
def prepare_clusters(raw_clusters):
    '''
    input [
           [['item1','item3'],['item2','item4']],
           [[],[],[]]
           ]
    '''
    items = set()
    for u in raw_clusters:
        for c in u:
            for i in c:
                items.add(i)
    N = len(items)
    item_list = list(items)
    item_dict = {}
    for i in range(N):
        item_dict[item_list[i]] = i
    
    users = []
    for u in raw_clusters:
        clusters = []
        for c in u:
            items= []
            for i in c:
                items.append(item_dict[i])
            clusters.append(items)
        users.append(clusters)
    return users, N, item_list, item_dict
# %% test prepare_clusters algorithm
raw_clusters = [
    [['A'],['B']],
    [['B','C'],['A','D']],
    [['A','B'],['C'],['D','E','F']]
    ]    
prepare_clusters(raw_clusters) 

# %%
def calculate_b(Ys,X,Us):
    bs = []
    for i in range(len(Ys)):
        Y = Ys[i]
        mask_pos = Y==1
        mask_neg = Y==-1
        tm = np.sum(mask_pos)/np.sum(mask_pos + mask_neg)
        Beta = (mask_pos)*1 + (mask_neg)*tm
        U = np.make_diagonal(Us[D*i:D*(i+1)], axis1=-1, axis2=-2)
        b = np.sum(Beta*(Y - np.dot(np.dot(X,U),(X.T))))
        bs.append(b)

    return np.mean(bs)

# %%    
def learn_lcc(user_clusters, N, max_itr=1):
    # Initialize values
    num_user = len(user_clusters)
    Ys = []
    for i in range(num_user):
        Y = get_Y(user_clusters[i],N)
        Ys.append(Y)    
    X = np.random.rand(N,D)  
    print(X)
    Us = np.ones(num_user*D,dtype=float)    
    itr = 0
    b = 0.
    last_loss = None
    
    while itr < max_itr: #TODO add not converged condition ==1e-6
        print(itr)
        b = calculate_b(Ys, X, Us) # update b 
        '''
        output_for_X = minimize(paramsX_to_minimize_with_grad, X.ravel(), 
                  args=(Ys,Us,b),method='CG', jac=True)
        if output_for_X.status == 0:
            X = output_for_X.x.reshape(N,D)
        else:
            print('X does not converge properly.')
        print(output_for_X.fun)
        '''
        '''
        output_for_Us = minimize(paramsUs_to_minimize_with_grad, Us, 
                  args=(Ys,X,b),method='CG', jac=True) 
        if output_for_Us.status == 0:
            Us = output_for_Us.x
        else:
            print('Us does not converge properly.')
        '''
        '''
        if last_loss:
            if abs(last_loss - output_for_Us.fun) <= (1e-6):
                break
    
        last_loss = output_for_Us.fun
        '''
        #print(last_loss)
        itr += 1
        
    return X, Ys, Us, b

    
y1 = [list(range(25)), 
     list(range(25,50)),
     list(range(50,75))]
y2 = [list(range(25,50)),
     list(range(50,75)), 
     list(range(75,100))]
ys = [y1,y2,y1,y2,y1,y2]
ys, N, item_list, item_dict = prepare_clusters(ys) 
X, Ys, Us, b = learn_lcc(ys,100)

   
    
# %% test paramsUs_to_minimize
y1 = [list(range(25)), 
     list(range(25,50)),
     list(range(50,75))]
y2 = [list(range(25,50)),
     list(range(50,75)), 
     list(range(75,100))]
Y1 = get_Y(y1,100)
Y2 = get_Y(y2,100)
Ys = [Y1,Y2]
b = 0.
X_correct = np.concatenate((
    np.concatenate((
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1)),
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01, 0 + 0.01,(25,1)),
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1))
        ),axis = 1),
    ),axis = 0) 
X_correct = np.random.randn(100,D) 
output = minimize(paramsUs_to_minimize_with_grad, np.array([1.,2.,1.,1.5]), 
                  args=(Ys,X_correct,b),method='CG', jac=True)

# %% test paramsX_to_minimize
y1 = [list(range(25)), 
     list(range(25,50)),
     list(range(50,75))]
y2 = [list(range(25,50)),
     list(range(50,75)), 
     list(range(75,100))]
Y1 = get_Y(y,100)
Y2 = get_Y(y2,100)
Ys = [Y1,Y2]
Us = [np.eye(2), np.eye(2)]
b = 0.
X_correct = np.concatenate((
    np.concatenate((
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1)),
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01, 0 + 0.01,(25,1)),
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1))
        ),axis = 1),
    ),axis = 0) 
X_shuffle = np.concatenate((
    np.concatenate((
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1)),
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01, 0 + 0.01,(25,1)),
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1))
        ),axis = 1),
    ),axis = 0) 
np.random.shuffle(X_shuffle)
output = minimize(paramsX_to_minimize_with_grad, X_correct.ravel(), 
                  args=(Ys,Us,b),method='CG', jac=True)
X_correct_predict = output.x.reshape(100,D)
plt.scatter(X_correct_predict[:,0],X_correct_predict[:,1])

output_shuffle = minimize(paramsX_to_minimize_with_grad, X_shuffle.ravel(), 
                  args=(Ys,Us,b),method='CG', jac=True)
#print(loss_all(Ys, output.x.reshape(100,D), Us,b))
# Reminder: check the output of minimize. see that it terminates
#output.status == 0
    

# %%  
# test with num user (M) = 1
y = [list(range(25)), 
     list(range(25,50)),
     list(range(50,75)), 
     list(range(75,100))]

Y = get_Y(y,100)
Ys = [Y]
Us = [np.eye(2)]
b = 0
# -- at each corner of X axis
X_correct = np.concatenate((
    np.concatenate((
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1)),
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01, 0 + 0.01,(25,1)),
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1))
        ),axis = 1),
    ),axis = 0)
loss_correct = loss_all(Ys, X_correct, Us, b)

# all pile around (0,1)
X_off = np.concatenate((
    np.random.uniform(0 - 0.01,0 + 0.01,(100,1)),
    np.random.uniform(1 - 0.01,1 + 0.01,(100,1))
    ),axis = 1)
loss_off = loss_all(Ys, X_off, Us, b)

X_shuffle = np.concatenate((
    np.concatenate((
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1)),
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01, 0 + 0.01,(25,1)),
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1))
        ),axis = 1),
    ),axis = 0)
np.random.shuffle(X_shuffle)
loss_shuffle = loss_all(Ys, X_shuffle, Us, b)

print('loss correct: ', loss_correct)
print('loss off: ', loss_off)
print('loss shuffle: ', loss_shuffle)

# %% 
# test loss funciton with num user (M) > 1
y1 = [list(range(25)), 
     list(range(25,50)),
     list(range(50,75))]
y2 = [list(range(25,50)),
     list(range(50,75)), 
     list(range(75,100))]
Y1 = get_Y(y,100)
Y2 = get_Y(y2,100)
Ys = [Y1,Y2]
Us = [np.eye(2), np.eye(2)]

# -- at each corner of X axis
X_correct = np.concatenate((
    np.concatenate((
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1)),
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01, 0 + 0.01,(25,1)),
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1))
        ),axis = 1),
    ),axis = 0)
b = 0
loss_correct = loss_all(Ys, X_correct, Us, b)

# all pile around (0,1)
X_off = np.concatenate((
    np.random.uniform(0 - 0.01,0 + 0.01,(100,1)),
    np.random.uniform(1 - 0.01,1 + 0.01,(100,1))
    ),axis = 1)
loss_off = loss_all(Ys, X_off, Us, b)

X_shuffle = np.concatenate((
    np.concatenate((
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1)),
        np.random.uniform(1 - 0.01,1 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1)),
        np.random.uniform(0 - 0.01,0 + 0.01,(25,1))
        ),axis = 1),
    np.concatenate((
        np.random.uniform(0 - 0.01, 0 + 0.01,(25,1)),
        np.random.uniform(-1 - 0.01,-1 + 0.01,(25,1))
        ),axis = 1),
    ),axis = 0)
np.random.shuffle(X_shuffle)
loss_shuffle = loss_all(Ys, X_shuffle, Us, b)

print('loss correct: ', loss_correct)
print('loss off: ', loss_off)
print('loss shuffle: ', loss_shuffle)


# %%

'''
class CollaborativeClustering:
    
    def __init__(self, lambda_x, lambda_u, sigma_u, D):
        self.lambda_x = lambda_x
        self.lambda_u = lambda_u
        self.sigma_u = sigma_u
        self.D = D
    
        
    def fit(self, user_clusters, N=None):
        pass
        
'''        