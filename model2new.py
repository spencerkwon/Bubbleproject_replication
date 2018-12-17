#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 16:24:19 2018

@author: ykwon
"""
import numpy as np

def returnprecisions (PTs, sigmaepsilon, sigmaV, sigmaS, gamma, theta, T):
    precisions = np.zeros((len(PTs), T+1))
    amat = np.zeros((len(PTs), T+1))
    bmat = np.zeros((len(PTs), T+1))
    cmat = np.zeros((len(PTs), T+1))
    zmat = np.zeros((len(PTs), T+1))
    # set final T
    precisions[:, T] = PTs
    zmat[:, T] = (T/sigmaepsilon**2)/(T/sigmaepsilon**2 + PTs)
    amat[:, T] = (1.+theta) * (1.-zmat[:, T])
    bmat[:, T] = (1.+theta) * zmat[:, T]
    cmat[:, T] = gamma/(T/sigmaepsilon**2 + PTs)
    # move down the induction
    for t in np.arange(T - 1, 0, -1):
        precisions[:, t] = precisions[:, t+1] - 1./sigmaS**2 * (bmat[:, t+1]/cmat[:, t+1])**2
        zmat[:, t] = (t/sigmaepsilon**2)/(t/sigmaepsilon**2 + precisions[:, t])
        bmat[:, t] = (1.+theta)*zmat[:, t] * ((1-precisions[:, t]/precisions[:, t+1])*(1.+theta)**(T-t-1)\
            + (precisions[:, t]/precisions[:, t+1]) * bmat[:, t+1])
        amat[:, t] = (1.+theta)**(T-t + 1) - bmat[:, t]

#        temp = (((1.+theta)**(T-t-1) - bmat[:, t+1]) * \
#        (1-precisions[:,t]/precisions[:, t+1]) + bmat[:, t+1])**2/(precisions[:, t] + t/sigmaepsilon**2) + \
#        sigmaS**2 *(cmat[:, t+1] + ((1+theta)**(T-t-1) - bmat[:, t+1])*(1/sigmaS**2) * (bmat[:, t+1]/cmat[:, t+1]) / precisions[:, t+1])**2
        
        temp2 = (amat[:, t+1] * (1-precisions[:,t]/precisions[:, t+1]) + bmat[:, t+1])**2
        temp3 = 1./(precisions[:, t] + t/sigmaepsilon**2) + (sigmaS**2) *(cmat[:, t+1]/bmat[:, t+1])**2
        temp2[temp2 < 0.] = np.nan
        temp3[temp3 < 0.] = np.nan
        cmat[:, t] = np.exp(np.log(gamma) + np.log(temp2) + np.log(temp3))
    
    precisions[:, 0] = precisions[:, 1] - (bmat[:, 1]/cmat[:, 1])**2/sigmaS**2
    
    return{'prec': precisions, 'a': amat, 'b': bmat, 'c': cmat, 'z': zmat}

def returnprecisionsnew (PTs, sigmaepsilon, sigmaV, sigmaS, gamma, theta, T):
    precisions = np.zeros((len(PTs), T+1))
    amat = np.zeros((len(PTs), T+1))
    bmat = np.zeros((len(PTs), T+1))
    logcmat = np.zeros((len(PTs), T+1))
    zmat = np.zeros((len(PTs), T+1))
    # set final T
    precisions[:, T] = PTs
    zmat[:, T] = (T/sigmaepsilon**2)/(T/sigmaepsilon**2 + PTs)
    amat[:, T] = (1.+theta) * (1.-zmat[:, T])
    bmat[:, T] = (1.+theta) * zmat[:, T]
    logcmat[:, T] = np.log(gamma) - np.log(T/sigmaepsilon**2 + PTs)
    
    # move down the induction
    for t in np.arange(T - 1, 0, -1):
        temp_p = precisions[:, t+1] - 1./sigmaS**2 * np.exp(2*(np.log(bmat[:, t+1]) -\
                  logcmat[:, t+1]))
        temp_p[np.isnan(temp_p)] = -1.
        temp_p[temp_p < 0.] = np.nan
        precisions[:, t] = temp_p

        zmat[:, t] = (t/sigmaepsilon**2)/(t/sigmaepsilon**2 + precisions[:, t])
        bmat[:, t] = (1.+theta)*zmat[:, t] * ((1-precisions[:, t]/precisions[:, t+1])*(1.+theta)**(T-t-1)\
            + (precisions[:, t]/precisions[:, t+1]) * bmat[:, t+1])
        amat[:, t] = (1.+theta)**(T-t + 1) - bmat[:, t]

#        temp = (((1.+theta)**(T-t-1) - bmat[:, t+1]) * \
#        (1-precisions[:,t]/precisions[:, t+1]) + bmat[:, t+1])**2/(precisions[:, t] + t/sigmaepsilon**2) + \
#        sigmaS**2 *(cmat[:, t+1] + ((1+theta)**(T-t-1) - bmat[:, t+1])*(1/sigmaS**2) * (bmat[:, t+1]/cmat[:, t+1]) / precisions[:, t+1])**2
        
        temp2 = (amat[:, t+1] * (1-precisions[:,t]/precisions[:, t+1]) + bmat[:, t+1])**2
        temp3 = 1./(precisions[:, t] + t/sigmaepsilon**2) + \
        (sigmaS**2) * np.exp(2*logcmat[:, t+1] - 2*np.log(bmat[:, t+1]))
        temp_c = np.log(gamma) + np.log(temp2) + np.log(temp3)
        # censor very large c values to prevent overflow
        temp_c[np.isnan(temp_c)] = 150.
        temp_c[temp_c > 100.] = np.nan
        logcmat[:, t] = temp_c    
    precisions[:, 0] = precisions[:, 1] - (bmat[:, 1]/np.exp(logcmat[:, 1]))**2\
    /sigmaS**2
    
    return{'prec': precisions, 'a': amat, 'b': bmat, 'logc': logcmat, 'z': zmat}



def model2_auto (sigmaepsilon, sigmaV, sigmaS, gamma, theta, T, N,\
              maxprec = 200000., gridlength = 200000):
    PTs = np.exp(np.linspace(start = np.log(1/(2*sigmaV**2)),\
                             stop = np.log(maxprec), num = gridlength))
    temp = returnprecisionsnew(PTs, sigmaepsilon, sigmaV, sigmaS, gamma, theta, T)
    initprecisions = temp['prec']
    amat = temp['a']
    bmat = temp['b']
    logcmat = temp['logc']
    zmat = temp['z']
    optind = np.nanargmin(abs(initprecisions[:, 0] - 1/sigmaV**2))
    precisions = initprecisions[optind, :]
    a = amat[optind, :]
    b = bmat[optind, :]
    c = np.exp(logcmat[optind, :])
    c[0] = 0.
    z = zmat[optind, :]
    
    # compute public signals and price paths
    shocks = np.random.normal(scale = sigmaS, size = (N, T))
    Ea = np.cumsum((b[1:]/c[1:])**2 / sigmaS**2)/precisions[1:]
    pa = a[1:]*Ea + b[1:]
    pubsignals = 1 - c[1:]/b[1:]*shocks
    E = np.cumsum((b[1:]/c[1:])**2  * pubsignals / sigmaS**2, axis = 1)/precisions[1:]
    p = a[1:]*E + b[1:] - c[1:]*shocks

    # compute price paths and diagnostic expectations
    Ebar = z[1:] + (1-z[1:])*E
    Ebara = z[1:] + (1-z[1:])*Ea
    Ebardiaga = (1+theta)*Ebara
    
    # calculated forward expectations of prices, calculate cg coefficients
    ratprec = precisions[1:T]/precisions[2:]
    lookahead1 = (1+theta)*(a[2:] * ratprec * E[:,:(T-1)] + \
    (a[2:]*(1-ratprec) + b[2:])*Ebar[:, :(T-1)])
    lookahead1a = (1+theta)*(a[2:] * ratprec * Ea[:(T-1)] + \
    (a[2:]*(1-ratprec) + b[2:])*Ebara[:(T-1)])

    return {'p': pa, 'Ed': Ebardiaga, \
            'pE': np.concatenate((np.zeros(1) + np.nan, lookahead1a)),
            'public': a[1:]*Ea,
            'private': b[1:], 'lookaheadraw': lookahead1,
            'praw': p,
            'pEraw': lookahead1}


def model2onlyV(sigmaepsilon, sigmaV, sigmaS, gamma, theta, T):
    
    # compute equilibrium prices under no learning
    temp = np.log(np.arange(T, 0, -1)/(np.arange(T, 0, -1) + sigmaepsilon**2/sigmaV**2))
    hatpi = np.exp(np.cumsum(temp)/np.arange(1, T+1))
    hatpi = hatpi[::-1]
    p = np.exp(np.arange(T, 0, -1)*(np.log(1+theta) + np.log(hatpi)))
    E = np.arange(1, T+1)/(np.arange(1, T+1) + sigmaepsilon**2/sigmaV**2)
    return{'p': p, 'E': (1+theta)*E}

