#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Peggy
"""
import torch
import numpy as np
import math
import random

def real_coin_value1(x, epsilon = 1e-10):
    d = x.size
    gradient = np.zeros(d)
    if np.sum(np.array(x)**2) < epsilon:
        gradient = 0
    else:
        A = np.matrix([[0.2,0.0],[0.0,0.5]])
        x = np.matrix(x)
        gradient = A*x/np.sqrt(x.T*A*x)
    return gradient
#    b = (x-0.2)>0
#    c = (x-0.2)<0
#    indexlist1 = np.logical_and(a,b)
#    indexlist2 = np.logical_and(a,c)
    
    
   
    return gradient
    #return 2/(1+np.exp(-(1*x)))-1
    #return -math.log(1-w,math.e)
    #return 0.1
    #return x

def coin_func_adagrad1(gt,v,order):
    # f = -ln(1-gt*v)
    # df = gt/(1-gt*v)
    value = -1*math.log(1-gt*v)
    if order == 0:
        return value
    elif order == 1:
        gradient = gt/(1-gt*v)
        return (value,gradient)
def real_coin_value2(x):
    # y=1/8*(x-10)^2
    return 1/4*(x)

def coin_func_adagrad2(x,order):
     # y=1/8*(x-10)^2
    value = 1/8*(x)*(x)
    if order == 0:
        return value
    elif order == 1:
        gradient = 1/4*(x)
        return (value, gradient)
    
def matrix_projection(y, A):
    
    # arg min (y-x)'A(y-x)
    proj_x = x
    return proj_x
    


    