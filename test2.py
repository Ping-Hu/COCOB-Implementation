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

T = 128
TESTREGRET = 0
d = 2

initial_x=np.zeros((2,1))
initial_x[0][0] = -0.5
initial_x[1][0] = -0.5
max_iterations_sgd = T
initial_learning_rate = [0.01,0.1,1]
ada_xs_list = [[],[]]
for ite in range(0,3):
    ada_x, ada_values, ada_xs = cb.adagrad( oracles.real_coin_value1 , initial_x, max_iterations_sgd, initial_learning_rate[ite])
    print('ada: ',ada_x)
    ada_xs = np.array(ada_xs)
    for i in range(T):    
        ada_xs_list = np.append(ada_xs_list,[[ada_xs[i][0][0]],[ada_xs[i][1][0]]],axis=1)
#print(ada_x)

initw = torch.Tensor([[1],[1]])
#

inputxlist,wealthlist,vlist,objlist = cb.cocob_torch(cb.func1_grad, initw,d,T)
print('cocob: ',vlist[T-1][:][0])
v_list = [[],[]]
for i in range(T):
    #a = np.array(vlist[i][0][0],vlist[i][1][0])
    #print(a)
    v_list = np.append(v_list,[[vlist[i][0][0]],[vlist[i][1][0]]],axis=1)
   
    
#print(v_list)
#v_list = vlist.numpy()
#print(v_list[20][1])
fig = 1
##x1 = np.arange(-4,4,1.0)
##x2 = np.arange(-4,4,1.0)
##X = np.meshgrid(x1,x2)
###cb.draw_contour(cb.func1_grad, X,X,1)
##Z = np.zeros((len(x1), len(x2)))
##for i in range(len(x1)):
##    for j in range(len(x2)):
##        a = torch.Tensor([[x1[i]],[x2[j]]])
##        #print('a: ',a)
##        Z[i, j] = cb.func1_grad( a , 2, 0 )
##print(Z)
##plt.figure(fig)
#gs = []
##x1 = x1.flatten()
##print(x1)
##x2 = x2.flatten()
#for i in range(T):
#    a = np.array([v_list[i][0],v_list[i][1]])
#    print(a)
#    gs.append(a)
#print(gs[0][0],gs[7][1],gs[1][0])
#cb.draw_contour( cb.func1_grad, T, v_list, ada_xs_list[:,256:384], fig)
#plt.ion()
#plt.show()