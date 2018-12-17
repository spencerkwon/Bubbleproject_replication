#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:55:39 2018

@author: ykwon
"""

import numpy as np

#from mpl_toolkits.mplot3d import Axes3D as axo
#def model1_nolag_auto(sigmaepsilon, sigmaV, sigmaS, theta, gamma, T, N,\
#               plot = False, title = "output.png"):
#    shocks = np.random.normal(scale = sigmaS, size = (N, T))
#    
#    # Calculating coefficients
#    a = np.zeros(T)
#    b = np.zeros(T)
#    c = np.zeros(T)
#    precP = np.zeros(T)
#    precP[0] = 1/sigmaV**2 + ((1+theta)/(gamma*sigmaepsilon**2*sigmaS))**2
#    for t in range(1, T):
#        precP[t] = precP[t-1] + ((1+theta)*(t+1)/(gamma*sigmaepsilon**2 * sigmaS))**2
#    z = 1/(1 + sigmaepsilon**2*precP/np.arange(1, T+1))
#    a = (1+theta)*(1-z)
#    b = (1+theta)*z
#    c = gamma/(precP + np.arange(1, T+1)/sigmaepsilon**2)
#
#    # compute public signals
#    pubsignal = 1-c/b*shocks
#    E = np.cumsum(pubsignal*(b/c)**2/sigmaS**2, axis = 1)/precP
#    Ea = np.cumsum((b/c)**2/sigmaS**2)/precP
#
#    # compute price paths and diagnostic expectations
#    ps = a*E + b - c*shocks
#    pa = a*Ea + b
#    Ebar = z + (1-z)*E
#    Ebara = z + (1-z)*Ea
#    Ebardiag = (1+theta)*Ebar
#    Ebardiaga = (1+theta)*Ebara
#    
#    # calculated forward expectations of prices, calculate cg coefficients
#    ratprec = precP[0:(T-1)]/precP[1:]
#    lookahead1 = a[1:] * ratprec * E[:,:(T-1)] + \
#    (a[1:]*(1-ratprec) + b[1:])*Ebardiag[:, :(T-1)]
#    lookahead1a = a[1:] * ratprec * Ea[:(T-1)] + \
#    (a[1:]*(1-ratprec) + b[1:])*Ebardiaga[:(T-1)]
#    
#    ratprec2 = precP[0:(T-2)]/precP[2:]
#    lookahead2 = a[2:] * ratprec2 * E[:,:(T-2)] + \
#    (a[2:]*(1-ratprec2) + b[2:])*Ebardiag[:, :(T-2)]
#    lookahead2a = a[2:] * ratprec2 * Ea[:(T-2)] + \
#    (a[2:]*(1-ratprec2) + b[2:])*Ebardiaga[:(T-2)]
#
#    
#    fr = lookahead1[:, 1:] - lookahead2
#    fe = ps[:, 2:] - lookahead1[:, 1:]
#    cgs = np.zeros(T - 2)
#    for t in np.arange(0, T-2):
#        covs = np.cov(fr[:, t], fe[:, t])
#        cgs[t] = covs[1, 0]/covs[0, 0]
#    
#    lamb = np.zeros(T - 2)
#    diff2 = ps[:, 2:] - ps[:, 1:(T-1)]
#    diff1 = ps[:, 1:(T-1)] - ps[:, :(T-2)]
#    for t in np.arange(0, T-2):
#        covdiff = np.cov(diff1[:, t], diff2[:,t])
#        lamb[t] = covdiff[1, 0]/covdiff[0,0]
#    
#
#    return {'p': pa, 'Ed': Ebardiaga, 'cg': cgs, \
#            'pE': np.concatenate((np.zeros(1) + np.nan, lookahead1a)),\
#            'pE2': np.concatenate((np.zeros(2) + np.nan, lookahead2a)),\
#            'public': a*Ea, 'private': b, 'lambda': lamb}

def M2_noncontemp(sigmaepsilon, sigmaV, sigmaS, theta, gamma, T, k, N):
    shocks = np.random.normal(scale = sigmaS, size = (N, T))
    # Calculating coefficients
    a1 = np.zeros(T + 1)
    a2 = np.zeros(T + 1)
    b = np.zeros(T + 1)
    c = np.zeros(T + 1)
    z = np.zeros(T + 1)
    precP = np.zeros(T + 1)
    precP[1] = 1/sigmaV**2
    z[1] = 1/sigmaepsilon**2 / (1/sigmaepsilon**2 + precP[1])
    b[1] = (1+theta)*z[1]
    a1[1] = (1+theta)*(1-z[1])
    c[1] = gamma*(1/sigmaepsilon**2 + precP[1])**(-1.)
    # update as before up to k
    for t in range(2, k +1):
        precP[t] = precP[t-1] + (b[t-1]/c[t-1])**2/sigmaS**2
        z[t] = t/sigmaepsilon**2 / (t/sigmaepsilon**2 + precP[t])
        a1[t] = (1+theta)*(1-z[t])
        b[t] = (1+theta)*z[t]
        c[t] = gamma*(t/sigmaepsilon**2 + precP[t])**(-1.)
        
    for t in range(k+1, T+1):
        precP[t] = precP[t-1] + (b[t-1]/c[t-1])**2/sigmaS**2
        z[t] = t/sigmaepsilon**2 / (t/sigmaepsilon**2 + precP[t])
        a1[t] = (1+theta)*(1-z[t])
        b[t] = (1+theta)*z[t] - theta*z[t-k]
        a2[t] = -theta*(1-z[t-k])
        c[t] = gamma*(t/sigmaepsilon**2 + precP[t])**(-1.)

    # compute public signals
    pubsignal = 1-c[1:]/b[1:]*shocks
    E = np.concatenate((np.zeros((N, 1)), \
                        np.cumsum(pubsignal[:,0:(T-1)]*(b[1:T]/c[1:T])**2/sigmaS**2, axis = 1)\
                        ), axis = 1)/precP[1:]
    lagE = np.concatenate((np.zeros((N, k)), E[:, 0:(T-k)]), axis = 1)
    
    Ea = np.concatenate((np.zeros(1), np.cumsum((b[1:T]/c[1:T])**2/sigmaS**2)))/precP[1:]
    lagEa = np.concatenate((np.zeros(k), Ea[0:(T-k)]))

    # compute price paths and diagnostic expectations
    ps = a1[1:]*E +a2[1:]*lagE + b[1:] - c[1:]*shocks
    pa = a1[1:]*Ea + a2[1:]*lagEa + b[1:]
    futprices = getDEprice(theta, k, E, Ea, a1, a2, z, b, precP)

    Ebara = z[1:] + (1-z[1:])*Ea
    lagEbara = np.concatenate((np.zeros(k), Ebara[0:(T-k)]))
    Ebardiaga = (1+theta)*Ebara - theta*lagEbara
    
    disp = (sigmaepsilon**2)/np.arange(1, T+1) * (z[1:])**2
    disp = np.concatenate((np.zeros(1), disp))
    return {'p': pa, 'Ed': Ebardiaga, 'pE': np.concatenate((np.zeros(1) + np.nan,\
                                                            futprices['pEa'])),
    'precisions': precP, 'praw': ps, 'lookaheadraw': futprices['pE'],
    'disp': disp}  
    

def M2_contemp(sigmaepsilon, sigmaV, sigmaS, theta, gamma, T, k, N):
    shocks = np.random.normal(scale = sigmaS, size = (N, T))
    # Calculating coefficients
    a1 = np.zeros(T + 1)
    a2 = np.zeros(T + 1)
    b = np.zeros(T + 1)
    c = np.zeros(T + 1)
    z = np.zeros(T + 1)
    precP = np.zeros(T + 1)
    precP[1] = 1/sigmaV**2 + ((1+theta)/(gamma*sigmaepsilon**2*sigmaS))**2
    z[1] = 1/(1+sigmaepsilon**2 *precP[1])
    a1[1] = (1+theta)*(1-z[1])
    b[1] = (1+theta)*z[1]
    c[1] = gamma/(precP[1] + 1/sigmaepsilon**2)
    noeq = False
    
    # update as before up to k
    for t in range(2, k+1):
        precP[t] = precP[t-1] + (b[t-1]/c[t-1])**2/sigmaS**2
        z[t] = t/sigmaepsilon**2 / (t/sigmaepsilon**2 + precP[t])
        a1[t] = (1+theta)*(1-z[t])
        b[t] = (1+theta)*z[t]
        c[t] = gamma*(t/sigmaepsilon**2 + precP[t])**(-1.)

    for t in range(k+1, T+1):
        if noeq == False:
            A = (theta* z[t-k])**2
            if A > 0:
                B = -2*theta*z[t-k]*(1+theta - theta * z[t-k])*t/sigmaepsilon**2 - \
                              (gamma*sigmaS)**2
                C = (1+theta*(1-z[t-k]))**2 * (t/sigmaepsilon**2)**2 + \
                precP[t-1]*(sigmaS*gamma)**2
                if B**2 - 4*A*C < 0:
                    print("Warning! No Equilibrium!")
                    noeq = True
                    precP[t:T] = np.nan
                    z[t:(T+1)] = np.nan
                    a1[t:(T+1)] = np.nan
                    a2[t:(T+1)] = np.nan
                    b[t:(T+1)] = np.nan
                    c[t:(T+1)] = np.nan
                else:
                    precP[t] = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
            else:
                B = -2*theta*z[t-k]*((1+theta)*t/sigmaepsilon**2 - theta*t/sigmaepsilon**2 * z[t-k]) - (gamma*sigmaS)**2
                C = (1+theta*(1-z[t-k]))**2 * (t/sigmaepsilon**2)**2 + precP[t-1]*(sigmaS*gamma)**2
                precP[t] = C/(-B)
        # the rest follows
            z[t] = 1/(1+sigmaepsilon**2 * precP[t]/t)
            a1[t] = (1+theta)*(1-z[t])
            a2[t] = -theta*(1-z[t-k])
            b[t] = (1+theta)*z[t] - theta*z[t-k]
            c[t] = gamma/(precP[t] + t/sigmaepsilon**2)
            
    # compute public signals
    pubsignal = 1-c[1:]/b[1:]*shocks
    E = np.concatenate((np.zeros((N, 1)), \
                        np.cumsum(pubsignal[:,0:(T-1)]*(b[1:T]/c[1:T])**2/sigmaS**2, axis = 1)\
                        ), axis = 1)/precP[1:]
    lagE = np.concatenate((np.zeros((N, k)), E[:, 0:(T-k)]), axis = 1)
    
    Ea = np.concatenate((np.zeros(1), np.cumsum((b[1:T]/c[1:T])**2/sigmaS**2)))/precP[1:]
    lagEa = np.concatenate((np.zeros(k), Ea[0:(T-k)]))

    # compute price paths and diagnostic expectations
    ps = a1[1:]*E +a2[1:]*lagE + b[1:] - c[1:]*shocks
    pa = a1[1:]*Ea + a2[1:]*lagEa + b[1:]
    futprices = getDEprice(theta, k, E, Ea, a1, a2, z, b, precP)

    Ebara = z[1:] + (1-z[1:])*Ea
    lagEbara = np.concatenate((np.zeros(k), Ebara[0:(T-k)]))
    Ebardiaga = (1+theta)*Ebara - theta*lagEbara
    
    return {'p': pa, 'Ed': Ebardiaga, 'pE': np.concatenate((np.zeros(1) + np.nan,\
                                                            futprices['pEa'])),
    'precisions': precP, 'praw': ps, 'lookaheadraw': futprices['pE']}  




# Helper functions to derive coefficients
    
def xi(t, s, a1, a2, z, b, precP):
    return (a1[s] * (1 - z[t]*(1 - precP[t]/precP[s])) + b[s]*(1-z[t]))
def zeta(t, s, a1, a2, z, b, precP):
    return(z[t] * (a1[s] *(1-precP[t]/precP[s]) + b[s]))

def xiprime(t, s, a1, a2, z, b, precP):
    return (a2[s] * (1 - z[t]*(1 - precP[t]/precP[s])))
def zetaprime(t, s, a1, a2, z, b, precP):
    return(z[t] * a2[s] *(1-precP[t]/precP[s]))

def getDEprice(theta, k, E, Ea, a1, a2, z, b, precP):
    T = np.shape(E)[1]
    N = np.shape(E)[0]
    DEprice = np.zeros((N, T - 1))
    DEpricea = np.zeros(T - 1)
    for s in range(0, T-1):
        t = s+1
        myxi = xi(t, t+1, a1, a2, z, b, precP)
        myzeta = zeta(t, t+1, a1, a2, z, b, precP)
        if t < k +1:
            DEprice[:, s] = (1+theta)*(myxi * E[:, s] + myzeta)
            DEpricea[s] = (1+theta)*(myxi * Ea[s] + myzeta)
        else :
            myxi1 = xi(t, t+1, a1, a2, z, b, precP)
            myzeta1 = zeta(t, t+1, a1, a2, z, b, precP)
            
            myxi2 = xi(t - k, t+1, a1, a2, z, b, precP)
            myzeta2 = zeta(t - k, t+1, a1, a2, z, b, precP)
            myxiprime2 = xiprime(t-k, t+1, a1, a2, z, b, precP)
            myzetaprime2 = zetaprime(t-k, t+1, a1, a2, z, b, precP)

            temp1 = myxi1 * E[:, s] + myzeta1 + a2[t+1]*E[:, s-k]
            temp2 = (myxi2 + myxiprime2) * E[:, s-k] + myzeta2 \
            + myzetaprime2
            temp1a = myxi1 * Ea[s] + myzeta1 + a2[t+1]*Ea[s-k]
            temp2a = (myxi2 +myxiprime2)* Ea[s-k] + myzeta2 + \
            + myzetaprime2
            
            DEprice[:, s] = (1+theta)*temp1 - theta*temp2
            DEpricea[s] = (1+theta)*temp1a - theta*temp2a
    return {'pE': DEprice, 'pEa': DEpricea}



