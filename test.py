#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Peggy
"""
import torch
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import coinBetting as cb
import oracles

T = 32
TESTREGRET = 0
d = 2
initw = torch.Tensor([[1],[1]])


inputxlist,wealthlist,vlist,objlist = cb.cocob_torch(initw,d,T)
wealth_list = wealthlist[:,0].numpy()
v_list = vlist[:,0].numpy()

#input_xlist,wealth_list,v_list,obj_list = cb.coin_betting(1,T)

initial_x=np.zeros((1,1))
initial_x = -0.5
max_iterations_sgd = T
initial_learning_rate = [0.01,0.1,1,10,100]
ada_xs_list = np.array([])
for ite in range(0,5):
    ada_x, ada_values, ada_xs = cb.adagrad( oracles.coin_func_adagrad1, initial_x, max_iterations_sgd, initial_learning_rate[ite])
    ada_xs = np.array(ada_xs)
    ada_xs_list = np.append(ada_xs_list, ada_xs)
#adadelta_x, adadelta_values, adadelta_xs = cb.adaDelta( oracles.coin_func_adagrad1, initial_x, max_iterations_sgd, 0.9, 2)

if TESTREGRET==1:
    w_bm_list, w_list, regret_list = cb.regret_cb(T, input_x_list, wealth_list)



plt.figure(1)
line_wealth = plt.plot(wealth_list, linewidth=1, color='r', label='1D cb wealth')
plt.xlabel('t')
plt.ylabel('Wealth_t')
plt.title('Wealth_t in COCOB method in the first 128 iterations')
plt.show()
plt.figure(2)
#plt.subplot(212)
line_v = plt.plot(v_list, linewidth=1, color='b', label='1D cb v')
plt.xlabel('t')
plt.ylabel('v_t')
plt.title('v_t in COCOB method in the first 128 iterations')
plt.show()

#plt.figure(2)
#plt.plot(obj_list,linewidth=1, color='b')
#plt.xlabel('t')
#plt.ylabel('obj func')
#plt.show()

points_to_plot=T
ax = plt.figure(3)
#line_cocob = plt.semilogx(range(1,max_iterations_sgd+1,int(max_iterations_sgd/points_to_plot)),v_list, linewidth=1,  label='COCOB')
line_ada0, = plt.semilogx(range(1,max_iterations_sgd+1,int(max_iterations_sgd/points_to_plot)),ada_xs_list[0:128], linewidth=1,  label='lr = 0.01')
line_ada1, = plt.semilogx(range(1,max_iterations_sgd+1,int(max_iterations_sgd/points_to_plot)),ada_xs_list[128:256], linewidth=1,  label='lr = 0.1')
line_ada2, = plt.semilogx(range(1,max_iterations_sgd+1,int(max_iterations_sgd/points_to_plot)),ada_xs_list[256:384], linewidth=1,  label='lr = 1')
#line_ada3, = plt.semilogx(range(1,max_iterations_sgd+1,int(max_iterations_sgd/points_to_plot)),ada_xs_list[384:512], linewidth=2,  label='AdaGrad')
#line_ada4, = plt.semilogx(range(1,max_iterations_sgd+1,int(max_iterations_sgd/points_to_plot)),ada_xs_list[512:640], linewidth=2,  label='AdaGrad')
ax.legend()
plt.xlabel('iterations')
plt.ylabel('v_t')
plt.title('Comparison between AdaGrad method and COCOB-ONS')
plt.show()

#plt.figure(4)
#plt.plot(ada_xs,linewidth=0.5, color='b')
#plt.xlabel('iterations')
#plt.ylabel('AdaGrad x')
#plt.show()
#
#points_to_plot=T
#plt.figure(5)
#line_ada, = plt.semilogx(range(1,max_iterations_sgd+1,int(max_iterations_sgd/points_to_plot)),adadelta_values, linewidth=2, color='b', label='AdaGrad')
#plt.xlabel('iterations')
#plt.ylabel('AdaDelta Obj Func')
#plt.show()
#
#plt.figure(6)
#plt.plot(adadelta_xs,linewidth=0.5, color='b')
#plt.xlabel('iterations')
#plt.ylabel('AdaDelta x')
#plt.show()



if TESTREGRET==1:
    plt.figure(4)
    plt.subplot(211)
    plt.plot(regret_list)
    plt.xlabel('T')
    plt.ylabel('regret')
    plt.subplot(212)
    line_w_bm = plt.plot(w_list, linewidth=0.5, color='r', marker='.', label='wealth t')
    line_w_bm = plt.plot(w_bm_list, linewidth=0.5, color='b', marker='.', label='wealth bm')
    plt.xlabel('T')
    plt.ylabel('wealth')
    plt.show()

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
x1 = np.arange(-1.0, 1.0, delta)
x2 = np.arange(-1.0, 1.0, delta)

X1, X2 = np.meshgrid(x1, x2)
Z = -X1**2 - X2**2
#Z2 = np.exp(-(X1 - 1)**2 - (X2 - 1)**2)
#Z = (Z1 - Z2) * 2

plt.figure()
CS = plt.contour(X1, X2, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Cone function')

