#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 00:12:06 2018

@author: ykwon
"""

import numpy as np
import matplotlib.pyplot as plt


def model0(sigmaepsilon, sigmaV, theta, gamma, sigmaS, T, k, N, plot = False):
    ts = np.arange(1, T+1)
    alpha = sigmaepsilon**2/sigmaV**2
    pis = ts/(ts + alpha)
    Ettheta = np.concatenate(((1+theta)*pis[0:k], \
                              (1+theta)*pis[k:T] - theta*pis[0:(T-k)]))
    p = Ettheta
    pE = np.concatenate(((1+theta)*pis[1:k]*Ettheta[1:k],
                        ((1+theta)*pis[k:T] - theta*pis[0:(T-k)])*Ettheta[k:T]))
    pE = np.concatenate((np.zeros(1) + np.nan, pE))
    Etrat = pis[0:T]
    prat = Etrat
    pErat = pis[1:T]*Etrat[0:(T-1)]
    pErat = np.concatenate((np.zeros(1) + np.nan, pErat))
    trough = k*((1-np.sqrt(theta/(1+theta)))**(-1.))\
                                       -sigmaepsilon**2/sigmaV**2
    if plot:
        plt.figure(figsize = (15, 15))
        plt.plot(np.arange(1, T+1), p)
        plt.plot(np.arange(1, T+1), pE)
        plt.plot(np.arange(1, T+1), prat)
        plt.plot(np.arange(1, T+1), pErat)
        plt.plot(np.arange(1, T+1), np.ones(T), color = "black")
        plt.legend(['Price', 'Expected Price', \
                    'Rational Price', 'Rational Expected Price'],
        fontsize = 20)
        plt.annotate('Boom',xy=(4.5, 0.05), fontsize = 25)
        plt.annotate('Bust',xy=(k + 10.5, 0.05), fontsize = 25)
        plt.annotate('Recovery',xy=(trough + 20, 0.05),\
                                    fontsize = 25)
        plt.annotate('V', xy = (0, 1.05), fontsize = 30)
s        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)


    return{'p': p, 'pE': pE, 'prat': prat, 'pErat': pErat}
def timelines(y, xstart, xstop, color='b'):
    """Plot timelines at y from xstart to xstop with given color."""   
    plt.hlines(y, xstart, xstop, color, lw=4)
    plt.vlines(xstart, y+0.03, y-0.03, color, lw=2)
    plt.vlines(xstop, y+0.03, y-0.03, color, lw=2)

sigmaV = .4
sigmaS = .3
gamma = .12
T = 100
sigmaepsilon = 1.3
theta = 1.
N = 20000
k = 16


model0(sigmaepsilon, sigmaV, theta, gamma, sigmaS, T, k, N, plot = True)
