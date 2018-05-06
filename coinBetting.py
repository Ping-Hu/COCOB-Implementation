#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Peggy
"""

import torch
from torch.autograd import Variable
import math
import numpy as np
import oracles as ora
import functions as funcs
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time

"""
 Online Newton Step (ONS)
   input: 
       G, D, alpha from the problem setting
       x(t), 
       fvalue_xt<-ora, 
       fgradient_xt<-ora,
   output:
       x(t+1)
"""
def online_newton_step(G, D, alpha, gradient_sum, xt, gt):
    
    beta = 0.5*min(1/(4*G*D),alpha)
    eps = 1/(beta^2*D^2)
    
    value, gradient = funcs.log_loss_function(xt, gt, 1)
    #xt = np.array(xt)
    (m,n) = xt.shape
    In = np.identity(m)
    A = gradient_sum+gradient*gradient.T+eps*In
    A = np.matrix(A)
    y = xt-1/beta*A.I*gradient
    
    nextx = ora.matrix_projection(y, A)

    return nextx

"""   
Coin betting through ONS
   input:        
       epsilon from user
       G, D, alpha from the problem setting
       
"""     
def coin_betting(init_wealth, T, epsilon = 1e-10):
    
    realvalues = [] # append each real bet value, gt
    z2sum = 1
    objfuncvalues = []
    #D = 1 # vt's set 1/2-(-1/2) = 1
    #G = 2 # see the analysis
    wealths = []
    vs = []
    #wealths = wealth.append(init_wealth)
    wealth = init_wealth
    v = -0.5
    k = 0
    while True:
        vs.append(v)
        w = v*wealth
                 
        #real_coin = ora.real_coin_value(v)
        gt = ora.real_coin_value2(v)
        #obj = ora.coin_func_adagrad(real_coin,v,0)
        obj = ora.coin_func_adagrad2(v,0)
        objfuncvalues.append(obj)
        #print(real_coin)
        realvalues.append(gt)
        
        wealth = wealth - gt*w
        wealths.append(wealth)
        #v = online_newton_step(G, D, alpha, grad_sum, vt, real_bet)
        z = gt/(1-gt*v)
        z2sum = z2sum + z*z
        A = z2sum
        
        v = max(min(v-2/(2-math.log(3))*z/A, 0.5), -0.5)
        
        k = k + 1
        if k>=T:
            print("wealth = : ", wealth)
            print("v: ", v)
            break
        if wealth <= 0:
            print("terminating k = ", k)
            print("v: ", v)
            break
    return (realvalues,wealths,vs,objfuncvalues)

def func1_grad(inx,d,order):
    center = torch.Tensor(d,1)
    center.fill_(0.0)
    center = Variable(center,requires_grad=False)
    x = Variable(inx, requires_grad=True)
    if d==1:
        A = Variable(torch.Tensor([[0.2]]),requires_grad=False)
    elif d==2:
        A = Variable(torch.Tensor([[0.2],[0.5]]),requires_grad=False)# params in y = A1*(x1-center1)^2+A2*(x2-center2)^2
    else:
        a = torch.Tensor(d,1)
        a.fill_(0.2)
        A = Variable(a,requires_grad=False)
      
    res = torch.mul((x-center).pow(2),A)
    y = torch.sqrt(res.sum())
    if order==0:
        return y
    elif order==1:
        l = torch.Tensor(d,1)
        l.fill_(1)
        y.backward(l)
        return (y.data,x.grad.data)
    else:
        raise NotImplementedError

def func2_grad(inx,d,order,epsilon=1e-10):
    v = abs(inx[0,0]-0.2)
    value = torch.Tensor(d,1)
    value.fill_(v)
    if abs(inx[0,0]-0.2)<epsilon:
        gradient = 0
    elif inx[0,0]<0.2:
        gradient = torch.Tensor(d,1)
        gradient.fill_(-1)
    else:
        gradient = torch.Tensor(d,1)
        gradient.fill_(1)
    return (value,gradient)
#    center = torch.Tensor(d,1)
#    center.fill_(0.2)
#    x = Variable(inx-center, requires_grad=True)
#    y = x.pow(2).sum()
#    if order==0:
#        return y
#    elif order==1:
#        l = torch.Tensor(d,1)
#        l.fill_(1)
#        y.backward(l)
#        return (y.data,x.grad.data)
#    else:
#        raise NotImplementedError


def cocob_torch(func, init_wealth, d, T):
    
    sqsum = torch.Tensor(d,1)
    sqsum = sqsum.fill_(1)
    
    # for log, collecting gt,ft,wealtht,vt
    gs = [] 
    objfuncvalues = []
    wealths = []
    vs = []
    
    wealth = init_wealth.clone()
    #v = torch.Tensor(T,1)
    #v = sqsum.fill_(-0.5)
    v = torch.Tensor(d,1)
    v[0,0] = -0.5
    v[1,0] = -0.5
    #v = v/2
    #print(v)
    k = 0
    while True:
        vs.append(v.numpy())
        w = (v*wealth).clone()
                 
        fvalue, gt = func(v,d,1)
        gs.append(gt)
        
        wealth = wealth - gt*w
        wealths.append(wealth.numpy())
#        print(vs)
        #v = online_newton_step(G, D, alpha, grad_sum, vt, real_bet)
        z = gt/(1-gt*v)
        sqsum = sqsum + z.pow(2)
        A = sqsum
        
        div = 2/(2-math.log(3))*z/A
        lb = v.clone()
        ub = v.clone()
        lb.fill_(-1/2)
        ub.fill_(1/2)
        v = torch.min(torch.max(v - div,lb),ub)
        
        
        k = k + 1
        if k>=T:
            print("wealth = : ", wealth)
            print("v: ", v)
            break
#        if wealth <= 0:
#            print("terminating k = ", k)
#            print("v: ", v)
#            break
    return (gs,wealths,vs,objfuncvalues)

def regret_cb(T, realbets, wealths): 
    
    wealth_benchmark_list = []
    wealth_list = []
    regret_list = []
    for k in range(T):
        v_benchmark = np.sum(np.array(realbets[0:k]))/(-2*np.sum(np.array(realbets[0:k])**2)-2*np.abs(np.sum(np.array(realbets[0:k]))))
        tmp = 1-v_benchmark * np.array(realbets[0:k])    
        wealth_bm_of_t = np.array([-math.log(y,math.e) for y in tmp])
        wealth_benchmark = np.sum(wealth_bm_of_t)
        wealth_benchmark_list.append(wealth_benchmark)
        wealth = math.log(wealths[k],math.e)
        wealth_list.append(wealth)
        regret_list.append(wealth_benchmark-wealth)
    
    return (wealth_benchmark_list, wealth_list, regret_list)

    

def adagrad( func, initial_x, maximum_iterations, initial_stepsize=1, initial_sum_of_squares=1e-3):
    """ 
    adagrad
    func:                   the function to optimize. It is called as "value, gradient = func( x, 1 )
    initial_x:              the starting point, should be a float
    maximum_iterations:     the maximum allowed number of iterations
    initial_stepsize:       the initial stepsize, should be a float
    initial_sum_of_squares: initial sum of squares
    """
    
    x = np.matrix(initial_x)
    d = initial_x.size
    # initialization
    values = []
    
    xs = []
    #start_time = time.time()
    iterations = 0
   
    # subgradient updates
    while True:
        gradient = func(x)        
        xs.append( x )
        
        # x = ( TODO: update of adagrad )
        gradient = np.array(gradient)
        initial_sum_of_squares = initial_sum_of_squares+gradient**2
        stepsize = initial_stepsize/np.sqrt(initial_sum_of_squares)
        sg = np.matrix(np.multiply(stepsize,gradient))
        x = np.maximum(np.minimum(x - sg,0.5),-0.5)
        #x = x - stepsize*gradient
        
        iterations += 1
        if iterations >= maximum_iterations:
            break
                
    return (x, values, xs)    
def draw_contour( func, T, gd_xs, newton_xs, fig, levels=np.arange(-0.5, 0.6, 0.01), x=np.arange(-0.6, 0.6, 0.05), y=np.arange(-0.6, 0.6, 0.05)):
    """ 
    Draws a contour plot of given iterations for a function
    func:       the contour levels will be drawn based on the values of func
    gd_xs:      gradient descent iterates
    newton_xs:  Newton iterates
    fig:        figure index
    levels:     levels of the contour plot
    x:          x coordinates to evaluate func and draw the plot
    y:          y coordinates to evaluate func and draw the plot
    """
    Z = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            a = torch.Tensor([[x[i]],[y[j]]])
            Z[i, j] = func( a , 2, 0 )
    
    plt.figure(fig)
    plt.contour( x, y, Z.T, levels, colors='0.75')
    plt.ion()
    plt.show()
    
    line_gd, = plt.plot( gd_xs[0][0], gd_xs[1][0], linewidth=2, color='r', marker='o', label='GD' )
    line_newton, = plt.plot( newton_xs[0][0], newton_xs[1][0], linewidth=2, color='m', marker='o',label='Newton' )
    
    L = plt.legend(handles=[line_gd,line_newton])
    plt.draw()
    time.sleep(1)
    
    for i in range( 1, T ):
        
        line_gd.set_xdata( np.append( line_gd.get_xdata(), gd_xs[0, min(i,T-1) ] ) )
        line_gd.set_ydata( np.append( line_gd.get_ydata(), gd_xs[1, min(i,T-1) ] ) )
        
        line_newton.set_xdata( np.append( line_newton.get_xdata(), newton_xs[0, min(i,T-1) ] ) )
        line_newton.set_ydata( np.append( line_newton.get_ydata(), newton_xs[1, min(i,T-1) ] ) )

        
        L.get_texts()[0].set_text( " GD, %d iterations" % min(i,T-1) )
        L.get_texts()[1].set_text( " Newton, %d iterations" % min(i,T-1) )    
        
        plt.draw()
        input("Press Enter to continue...")

def adaDelta( func, initial_x, maximum_iterations, decay, windowsize, initial_stepsize=1, initial_sum_of_squares=1e-3):
    """ 
    adagrad
    func:                   the function to optimize. It is called as "value, gradient = func( x, 1 )
    initial_x:              the starting point, should be a float
    maximum_iterations:     the maximum allowed number of iterations
    initial_stepsize:       the initial stepsize, should be a float
    initial_sum_of_squares: initial sum of squares
    """
    
    x = initial_x
    
    # initialization
    values = []
    runtimes = []
    xs = []
    #start_time = time.time()
    iterations = 0
    v0=10
    gradients = np.array([])
    dxs = np.array([])
    RMSg = 0
    RMSx = 0
    Eg = 0
    Edx = 0
    # subgradient updates
    while True:
        gt = ora.real_coin_value1(x)
        value, gradient = ora.coin_func_adagrad1(gt,x,1)
        print(gt)
        value = np.double( value )
    
        # updating the logs
        values.append( value )
        #runtimes.append( time.time() - start_time )
        xs.append( x )
        
        Eg = decay*np.sum(gradients)/(max(gradients.size,1)) + (1-decay)*(gradient**2)
        RMSg = np.sqrt(initial_sum_of_squares+Eg)
        
        if gradients.size <= windowsize-1:
            gradients = np.append(gradients,[gradient**2])
            
        else:
            gradients = np.delete(gradients,0)
            gradients = np.append(gradients,[gradient**2])
        
        RMSdx = np.sqrt(Edx+initial_sum_of_squares)
        dx = -1*RMSdx/RMSg*gradient
        Edx = decay*np.sum(dxs)/(max(dxs.size,1)) + (1-decay)*(dx**2)
        
        if dxs.size <= windowsize-1:
            dxs = np.append(dxs, [dx**2])
        else:
            dxs = np.delete(dxs,0)
            dxs = np.append(dxs,[dx**2])
            
        
        x = x + dx
        
        iterations += 1
        if iterations >= maximum_iterations:
            break
                
    return (x, values, xs)    
     
    
    
    