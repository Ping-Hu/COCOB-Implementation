#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:14:28 2018

@author: Peggy
"""
import torch
from torch.autograd import Variable
import math
import numpy as np
#import oracles as ora
import time
import coinBetting as cb
import oracles
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

T = 256
TESTREGRET = 0
d = 2

initial_x=np.zeros((2,1))
initial_x[0][0] = -0.5
initial_x[1][0] = -0.5
max_iterations_sgd = T
initial_learning_rate = [0.01,0.1,1]
ada_xs_list = [[],[]]
for ite in range(0,3):
    ada_x, ada_values, ada_xs = cb.adagrad( oracles.real_coin_value3 , initial_x, max_iterations_sgd, initial_learning_rate[ite])
    print('ada: ',ada_x)
    ada_xs = np.array(ada_xs)
    for i in range(T):    
        ada_xs_list = np.append(ada_xs_list,[[ada_xs[i][0][0]],[ada_xs[i][1][0]]],axis=1)

# test COCOB-ONS using 2D function f(x) of cb.func1_grad
initw = torch.Tensor([[1],[1]])
inputxlist,wealthlist,vlist,objlist = cb.cocob_torch(cb.func1_grad, initw,d,T)
#print('cocob: ',vlist[T-1][:][0])
v_list = [[],[]]
for i in range(T):
    #a = np.array(vlist[i][0][0],vlist[i][1][0])
    #print(a)
    v_list = np.append(v_list,[[vlist[i][0][0]],[vlist[i][1][0]]],axis=1)

ax = plt.figure(2)
# visualize v_t in COCOB-ONS
#line_vt0, = plt.semilogx(range(1,T+1,1),v_list[0,:], linewidth=1,  label='x_t')
#line_vt1, = plt.semilogx(range(1,T+1,1),v_list[1,:], linewidth=1,  label='y_t')
# visualize v_t in AdaGrad
line_vt0, = plt.semilogx(range(1,T+1,1),ada_xs_list[0,T:T*2], linewidth=1,  label='x_t')
line_vt1, = plt.semilogx(range(1,T+1,1),ada_xs_list[1,T:T*2], linewidth=1,  label='y_t')
plt.xlabel('iteration')
plt.ylabel('v_t')
ax.legend()
plt.show()

    
fig = 1

cb.draw_contour( cb.func1_grad, T, v_list, ada_xs_list[:,T:2*T], fig)
