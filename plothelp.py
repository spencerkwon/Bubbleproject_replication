#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:00:03 2018

@author: ykwon
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.cm as cm
import statsmodels.api as sm

    

# helper function: divides into groups of size C, 
# collects sample regression coefficients
# X, Ys are N by T, tc is the time interval
def timelines(y, xstart, xstop, color='b'):
    """Plot timelines at y from xstart to xstop with given color."""   
    plt.hlines(y, xstart, xstop, color, lw=4)
    plt.vlines(xstart, y+0.03, y-0.03, color, lw=2)
    plt.vlines(xstop, y+0.03, y-0.03, color, lw=2)

def wholereg (Xs, Ys):
    rawbetas = np.mean(Ys*Xs, axis = 1)/np.mean(Xs**2, axis = 1)
    return rawbetas

def windowreg_va (windows, Xs, Ys):
    grid = [0] + windows
    grid = np.append(grid, np.shape(Ys)[1])
    ts = np.zeros(len(grid)-1)
    betas = np.zeros(len(grid)-1)
    betalow = np.zeros(len(grid)-1)
    betahigh = np.zeros(len(grid)-1)

    for t in range(0, len(grid)-1):
        tYs = Ys[:, grid[t]:grid[t+1]]
        tXs = Xs[:, grid[t]:grid[t+1]]
        
        tempX = np.ndarray.flatten(tXs)
        tempY = np.ndarray.flatten(tYs)
        mod = sm.OLS(tempY, tempX)
        res = mod.fit()
        betas[t] = res.params
        confint = res.conf_int(0.05)
        betahigh[t] = confint[0, 1]
        betalow[t] = confint[0, 0]
        ts[t] = (grid[t+1]+grid[t])/2
    return {'t': ts, 'beta': betas, 'betalow': betalow,
            'betahigh': betahigh}

def windowreg_vb (windows, Xs, Ys):
    grid = [0] + windows
    grid = np.append(grid, np.shape(Ys)[1])
    ts = np.zeros(len(grid)-1)
    betas = np.zeros(len(grid)-1)
    betalow = np.zeros(len(grid)-1)
    betahigh = np.zeros(len(grid)-1)
    N = np.shape(Ys)[0]

    for t in range(0, len(grid)-1):
        tYs = Ys[:, grid[t]:grid[t+1]]
        tXs = Xs[:, grid[t]:grid[t+1]]
        
        
        temp_betas = np.sum(tYs*tXs, axis = 1)/\
        np.sum(tXs*tXs, axis = 1)
        
        betas[t] = np.mean(temp_betas)
        width = np.std(temp_betas)/np.sqrt(N)
        betahigh[t] = betas[t] + 2*width
        betalow[t] = betas[t] - 2*width
        ts[t] = (grid[t+1]+grid[t])/2
    return {'t': ts, 'beta': betas, 'betalow': betalow,
            'betahigh': betahigh}

# all increasing buckets, 
# va: no intercept, pooling
# vb: intercept, pooling
# vc: no intercept, no pooling
def expandreg_va (Xs, Ys):
    T = np.shape(Ys)[1]
    ts = np.arange(1, T+1)
    betas = np.zeros(T) + np.nan
    betahigh = np.zeros(T) + np.nan
    betalow = np.zeros(T) + np.nan
    for t in range(1, T):
        tempX = np.ndarray.flatten(Xs[:, 0:(t+1)])
        tempY = np.ndarray.flatten(Ys[:, 0:(t+1)])
        mod = sm.OLS(tempY, tempX)
        res = mod.fit()
        betas[t] = res.params
        confint = res.conf_int(0.05)
        betahigh[t] = confint[0, 1]
        betalow[t] = confint[0, 0]
    return {'t': ts, 'beta': betas, 'betalow': betalow,
            'betahigh': betahigh}
    
#def expandreg_va_intercept (Xs, Ys):
#    T = np.shape(Ys)[1]
#    ts = np.arange(1, T+1)
#    betas = np.zeros(T) + np.nan
#    betahigh = np.zeros(T) + np.nan
#    betalow = np.zeros(T) + np.nan
#    for t in range(1, T):
#        tempX = np.ndarray.flatten(Xs[:, 0:(t+1)])
#        tempY = np.ndarray.flatten(Ys[:, 0:(t+1)])
#        tempX2 = sm.add_constant(tempX)
#        mod = sm.OLS(tempY, tempX2)
#        res = mod.fit()
#        betas[t] = res.params[1]
#        confint = res.conf_int(0.05)
#        betahigh[t] = confint[1, 1]
#        betalow[t] = confint[1, 0]
#    return {'t': ts, 'beta': betas, 'betalow': betalow,
#            'betahigh': betahigh}
    
def expandreg_vb (Xs, Ys):
    T = np.shape(Ys)[1]
    N = np.shape(Ys)[0]
    ts = np.arange(1, T+1)
    betas = np.zeros(T) + np.nan
    betahigh = np.zeros(T) + np.nan
    betalow = np.zeros(T) + np.nan
    for t in range(1, T):
        tempX = Xs[:, 0:(t+1)]
        tempY = Ys[:, 0:(t+1)]
        temp_betas = np.sum(tempX*tempY, axis = 1)/\
        np.sum(tempX*tempX, axis = 1)
        
        betas[t] = np.mean(temp_betas)
        width = np.std(temp_betas)/np.sqrt(N)
        betahigh[t] = betas[t] + 2*width
        betalow[t] = betas[t] - 2*width
    return {'t': ts, 'beta': betas, 'betalow': betalow,
            'betahigh': betahigh}
    

def getExtrapWindows (windows, modelobject):
    praw = modelobject['praw']
    pEraw = modelobject['lookaheadraw']
    pdiff = np.diff(praw, axis = 1)
    T = np.shape(praw)[1]
    polddiff = pdiff[:, :(T-2)]
    pnewdiff = pdiff[:, 1:(T-1)]
    pexpectedchange = pEraw - praw[:, :(T-1)]
    expectedextrap_a = windowreg_va(windows, polddiff, pexpectedchange[:, 1:])
    expectedextrap_b = windowreg_vb(windows, polddiff, pexpectedchange[:, 1:])
    expectedextrap_s = windowreg_va(windows, polddiff, pnewdiff)
    rawexpected = wholereg(polddiff, pexpectedchange[:, 1:])
    rawraw = wholereg(polddiff, pnewdiff)
    return {'t': 1 + expectedextrap_a['t'],
            'expectationa': expectedextrap_a,
            'expectationb': expectedextrap_b,
            'expectations': expectedextrap_s,
            'rawexpected': rawexpected,
            'rawraw': rawraw}

def getExtrapExpand (modelobject):
    praw = modelobject['praw']
    pEraw = modelobject['lookaheadraw']
    pdiff = np.diff(praw, axis = 1)
    T = np.shape(praw)[1]
    polddiff = pdiff[:, :(T-2)]
    pexpectedchange = pEraw - praw[:, :(T-1)]
    expectedextrap_a = expandreg_va(polddiff, pexpectedchange[:, 1:])
    expectedextrap_b = expandreg_vb(polddiff, pexpectedchange[:, 1:])
    
    rawexpected = wholereg(polddiff, pexpectedchange[:, 1:])
    return {'t': 1 + expectedextrap_a['t'],
            'expectationa': expectedextrap_a,
            'expectationb': expectedextrap_b,
            'rawexpected': rawexpected}

def wholehist(extrapdiag, extraprat):
    xdiag = extrapdiag['rawexpected']
    xrat = extraprat['rawexpected']
    kdediag = stats.gaussian_kde(xdiag)
    kderat = stats.gaussian_kde(xrat)
    
    countd, binsd = np.histogram(xdiag, bins = 50, normed = True)
    countr, binsr = np.histogram(xrat, bins = 50, normed = True)
    
    fig, ax1 = plt.subplots(figsize = (15, 8))
    ax1.set_title("Expected price change extrapolation", fontsize = 30)
    ax1.plot(binsr, kderat(binsr), 'b-')
    ax1.set_xlabel('Beta', fontsize = 20)
    plt.xticks(fontsize = 20)
    ax1.set_ylabel('Density', color='b', fontsize = 20)
    ax1.tick_params('y', colors='b')
    ax1.legend(['Rational'], loc = 'upper left', fontsize = 20)
    plt.yticks(fontsize = 20)
    ax2 = ax1.twinx()
    ax2.plot(binsd, kdediag(binsd), 'r-')
    ax2.set_ylabel('Density', color = 'r', fontsize = 20)
    ax2.tick_params('y', colors = 'r')
    plt.yticks(color = "r", fontsize = 20)
    ax2.legend(['Diagnostic'],loc = 'upper right', fontsize = 20)
    plt.show()
    
    xdiag = extrapdiag['rawraw']
    xrat = extraprat['rawraw']
    kdediag = stats.gaussian_kde(xdiag)
    kderat = stats.gaussian_kde(xrat)
    
    countd, binsd = np.histogram(xdiag, bins = 50, normed = True)
    countr, binsr = np.histogram(xrat, bins = 50, normed = True)

    fig, ax1 = plt.subplots(figsize = (15, 8))
    ax1.set_title("Raw price change extrapolation", fontsize = 30)
    ax1.plot(binsr, kderat(binsr), 'b-')
    ax1.set_xlabel('Beta', fontsize = 20)
    plt.xticks(fontsize = 20)
    ax1.set_ylabel('Density', color='b', fontsize = 20)
    ax1.tick_params('y', colors='b')
    ax1.legend(['Rational'], loc = 'upper left', fontsize = 20)
    plt.yticks(fontsize = 20)
    ax2 = ax1.twinx()
    ax2.plot(binsd, kdediag(binsd), 'r-')
    ax2.set_ylabel('Density', color = 'r', fontsize = 20)
    ax2.tick_params('y', colors = 'r')
    plt.yticks(color = "r", fontsize = 20)
    ax2.legend(['Diagnostic'],loc = 'upper right', fontsize = 20)
    plt.show()
    
def localbucket(extrapdiag, extraprat, windows, T, modelobject, modelobjectrat, reduced):
    fig, ax1 = plt.subplots(figsize = (15, 8))
    plt.title("Expected price change extrapolation", fontsize = 30)
    t = np.arange(1, T+1)
    s1 = modelobject['p']
    ax1.plot(t, s1, 'b-', alpha = 0.2)
    ax1.set_xlabel('T')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Average Price', color='b', fontsize = 20)
    ax1.tick_params('y', colors='b')
    colors = cm.rainbow(np.linspace(0, 1, 1 + len(windows)))
    tempoffs = np.concatenate((np.ones(1)*2., np.array(windows) + 1.))
    tempoffs = np.concatenate((tempoffs, np.zeros(1) + T-1))
    # create bucket lines
    for i in np.arange(0, len(colors)):
        timelines(0., tempoffs[i], tempoffs[i+1], color = colors[i])
    plt.xticks(5*np.arange(0, T//5 +5 ), fontsize = 20)
    plt.yticks(fontsize = 20)
    ax2 = ax1.twinx()
    if reduced:
        cols = ['r-']
        colsdot = ['r-.']
        t2 = extrapdiag['t'] 
        s2 = extrapdiag['expectationa']['beta']
        s3 = extrapdiag['expectationa']['betalow'] 
        s4 = extrapdiag['expectationa']['betahigh'] 
        ax2.plot(t2, s2, 'r-')
        ax2.plot(t2, s3, 'r-.')
        ax2.plot(t2, s4, 'r-.')
    else:
         cols = ['r-', 'b-', 'g-']
         colsdot = ['r-.', 'b-.', 'g-.']
         names = ['expectationa', 'expectationb', 'expectations']
         for i in np.arange(0, len(names)):
            t2 = extrapdiag['t'] 
            s2 = extrapdiag[names[i]]['beta']
            s3 = extrapdiag[names[i]]['betalow'] 
            s4 = extrapdiag[names[i]]['betahigh'] 
            ax2.plot(t2, s2, cols[i])
            ax2.plot(t2, s3, colsdot[i])
            ax2.plot(t2, s4, colsdot[i])

    plt.yticks(fontsize = 20)
    ax2.set_ylabel('Local betas', color='r', fontsize = 20)
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.show()


    fig, ax1 = plt.subplots(figsize = (15, 8))
    plt.title("Expected price change extrapolation, rational", fontsize = 30)
    t = np.arange(1, T+1)
    s1 = modelobjectrat['p']
    ax1.plot(t, s1, 'b-', alpha = 0.2)
    ax1.set_xlabel('T')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Average Price', color='b', fontsize = 20)
    ax1.tick_params('y', colors='b')
    colors = cm.rainbow(np.linspace(0, 1, 1 + len(windows)))
    tempoffs = np.concatenate((np.ones(1)*2., np.array(windows) + 1.))
    tempoffs = np.concatenate((tempoffs, np.zeros(1) + T-1))
    # create bucket lines
    for i in np.arange(0, len(colors)):
        timelines(0., tempoffs[i], tempoffs[i+1], color = colors[i])
    plt.xticks(5*np.arange(0, T//5 +5 ), fontsize = 20)
    plt.yticks(fontsize = 20)
    ax2 = ax1.twinx()
    if reduced:
        cols = ['r-']
        colsdot = ['r-.']
        t2 = extraprat['t'] 
        s2 = extraprat['expectationa']['beta']
        s3 = extraprat['expectationa']['betalow'] 
        s4 = extraprat['expectationa']['betahigh'] 
        ax2.plot(t2, s2, 'r-')
        ax2.plot(t2, s3, 'r-.')
        ax2.plot(t2, s4, 'r-.')
    else:
         cols = ['r-', 'b-']
         colsdot = ['r-.', 'b-.']
         names = ['expectationa', 'expectationb']
         for i in np.arange(0, len(names)):
            t2 = extraprat['t'] 
            s2 = extraprat[names[i]]['beta']
            s3 = extraprat[names[i]]['betalow'] 
            s4 = extraprat[names[i]]['betahigh'] 
            ax2.plot(t2, s2, cols[i])
            ax2.plot(t2, s3, colsdot[i])
            ax2.plot(t2, s4, colsdot[i])

    plt.yticks(fontsize = 20)
    ax2.set_ylabel('Local betas', color='r', fontsize = 20)
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.show()

    
def masterdisplay1windows(windows, modelobject, modelobjectrat, reduced = True):
    T = np.shape(modelobject['praw'])[1]
    plt.figure(figsize=(15,15))
    plt.plot(np.arange(1, T+1), modelobject['p'], "r")
    plt.plot(np.arange(1, T+1), modelobject['pE'], "r-.")
    plt.plot(np.arange(1, T+1), modelobjectrat['p'], "b")
    plt.plot(np.arange(1, T+1), modelobjectrat['pE'], "b-.")
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(['Diagnostics, price',
                'Expected price',
                'Rational price',
                'Expected price'], prop={'size': 30})

    plt.figure(figsize=(15,15))
    plt.plot(np.arange(1, T+1), modelobject['p'], "r")
    plt.plot(np.arange(1, T+1), modelobjectrat['p'], "b")
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(['Diagnostics, price',
                'Rational price'], prop={'size': 30})

    extrapdiag = getExtrapWindows(windows, modelobject)
    extraprat = getExtrapWindows(windows, modelobjectrat)
    
    wholehist(extrapdiag, extraprat)
    localbucket(extrapdiag, extraprat, windows, T, modelobject, modelobjectrat, reduced)

def masterdisplay1expand(modelobject, modelobjectrat):
    T = np.shape(modelobject['praw'])[1]
    plt.figure(figsize=(15,15))
    plt.plot(np.arange(1, T+1), modelobject['p'], "r")
    plt.plot(np.arange(1, T+1), modelobject['pE'], "r-.")
    plt.plot(np.arange(1, T+1), modelobjectrat['p'], "b")
    plt.plot(np.arange(1, T+1), modelobjectrat['pE'], "b-.")
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(['Diagnostics, price',
                'Expected price',
                'Rational price',
                'Expected price'], prop={'size': 30})

    plt.figure(figsize=(15,15))
    plt.plot(np.arange(1, T+1), modelobject['p'], "r")
    plt.plot(np.arange(1, T+1), modelobjectrat['p'], "b")
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(['Diagnostics, price',
                'Rational price'], prop={'size': 30})

    extrapdiag = getExtrapExpand(modelobject)
    extraprat = getExtrapExpand(modelobjectrat)

    wholehist(extrapdiag, extraprat)

    fig, ax1 = plt.subplots(figsize = (15, 8))
    plt.title("Expected price change extrapolation", fontsize = 30)
    t = np.arange(1, T+1)
    s1 = modelobject['p']
    ax1.plot(t, s1, 'b-', alpha = 0.2)
    ax1.set_xlabel('T')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Average Price', color='b', fontsize = 20)
    ax1.tick_params('y', colors='b')
    plt.xticks(5*np.arange(0, T//5 +5 ), fontsize = 20)
    plt.yticks(fontsize = 20)
    ax2 = ax1.twinx()
    names = ['expectationa', 'expectationb']
    cols = ['r-', 'b-']
    colsdot = ['r-.', 'b-.']
    for i in np.arange(0, len(names)):
        t2 = extrapdiag['t'] 
        s2 = extrapdiag[names[i]]['beta']
        s3 = extrapdiag[names[i]]['betalow'] 
        s4 = extrapdiag[names[i]]['betahigh'] 
    
        ax2.plot(t2, s2, cols[i])
        ax2.plot(t2, s3, colsdot[i])
        ax2.plot(t2, s4, colsdot[i])
    plt.yticks(fontsize = 20)
    ax2.set_ylabel('Local betas', color='r', fontsize = 20)
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize = (15, 8))
    plt.title("Expected price change extrapolation, rational", fontsize = 30)
    t = np.arange(1, T+1)
    s1 = modelobjectrat['p']
    ax1.plot(t, s1, 'b-', alpha = 0.2)
    ax1.set_xlabel('T')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Average Price', color='b', fontsize = 20)
    ax1.tick_params('y', colors='b')
    plt.xticks(5*np.arange(0, T//5 +5 ), fontsize = 20)
    plt.yticks(fontsize = 20)
    ax2 = ax1.twinx()
    names = ['expectationa', 'expectationb']
    cols = ['r-', 'b-']
    colsdot = ['r-.', 'b-.']
    for i in np.arange(0, len(names)):
        t2 = extraprat['t'] 
        s2 = extraprat[names[i]]['beta']
        s3 = extraprat[names[i]]['betalow'] 
        s4 = extraprat[names[i]]['betahigh'] 
    
        ax2.plot(t2, s2, cols[i])
        ax2.plot(t2, s3, colsdot[i])
        ax2.plot(t2, s4, colsdot[i])
    plt.yticks(fontsize = 20)
    ax2.set_ylabel('Local betas', color='r', fontsize = 20)
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.show()
    

def masterdisplay2windows(windows, modelobject, modelobjectrat, reduced = True):
    T = np.shape(modelobject['praw'])[1]
    plt.figure(figsize=(15,15))
    plt.plot(np.arange(1, T+1), modelobject['p'], 'r')
    plt.plot(np.arange(1, T+1), modelobject['pE'], 'r-.')
    plt.plot(np.arange(1, T+1), modelobject['Ed'], 'r-o')
    plt.plot(np.arange(1, T+1), modelobjectrat['p'], 'b')
    plt.plot(np.arange(1, T+1), modelobjectrat['pE'], 'b-.')
    plt.plot(np.arange(1, T+1), modelobjectrat['Ed'],'b-o')

    plt.xticks(np.arange(1, T//5 + 1)*5, fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(['Speculation, price',
                'Speculation, expected price',
                'Speculation, Ed[V]',
                'Rational price',
                'Rational expected price',
                'Rational Ed[V]'], prop={'size': 30})

    plt.figure(figsize=(15,15))
    plt.plot(np.arange(1, T+1), modelobject['p'], 'r')
    plt.plot(np.arange(1, T+1), modelobject['Ed'], 'r-o')
    plt.plot(np.arange(1, T+1), modelobjectrat['p'], 'b')
    plt.plot(np.arange(1, T+1), modelobjectrat['Ed'],'b-o')

    plt.xticks(np.arange(1, T//5 + 1)*5, fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(['Speculation, price',
                'Speculation, Ed[V]',
                'Rational price',
                'Rational E[V]'], prop={'size': 30})
    
    extrapdiag = getExtrapWindows(windows, modelobject)
    extraprat = getExtrapWindows(windows, modelobjectrat)

    wholehist(extrapdiag, extraprat)
    localbucket(extrapdiag, extraprat, windows, T, modelobject, modelobjectrat, reduced)
    
    
    
def masterdisplay2expand(modelobject, modelobjectrat):
    T = np.shape(modelobject['praw'])[1]
    plt.figure(figsize=(15,15))
    plt.plot(np.arange(1, T+1), modelobject['p'], 'r')
    plt.plot(np.arange(1, T+1), modelobject['pE'], 'r-.')
    plt.plot(np.arange(1, T+1), modelobject['Ed'], 'r-o')
    plt.plot(np.arange(1, T+1), modelobjectrat['p'], 'b')
    plt.plot(np.arange(1, T+1), modelobjectrat['pE'], 'b-.')
    plt.plot(np.arange(1, T+1), modelobjectrat['Ed'],'b-o')

    plt.xticks(np.arange(1, T//5 + 1)*5, fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(['Speculation, price',
                'Speculation, expected price',
                'Speculation, Ed[V]',
                'Rational price',
                'Rational expected price',
                'Rational Ed[V]'], prop={'size': 30})

    plt.figure(figsize=(15,15))
    plt.plot(np.arange(1, T+1), modelobject['p'], 'r')
    plt.plot(np.arange(1, T+1), modelobject['Ed'], 'r-o')
    plt.plot(np.arange(1, T+1), modelobjectrat['p'], 'b')
    plt.plot(np.arange(1, T+1), modelobjectrat['Ed'],'b-o')

    plt.xticks(np.arange(1, T//5 + 1)*5, fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(['Speculation, price',
                'Speculation, Ed[V]',
                'Rational price',
                'Rational E[V]'], prop={'size': 30})
    extrapdiag = getExtrapExpand(modelobject)
    extraprat = getExtrapExpand(modelobjectrat)

    wholehist(extrapdiag, extraprat)

    fig, ax1 = plt.subplots(figsize = (15, 8))
    plt.title("Expected price change extrapolation", fontsize = 30)
    t = np.arange(1, T+1)
    s1 = modelobject['p']
    ax1.plot(t, s1, 'b-', alpha = 0.2)
    ax1.set_xlabel('T')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Average Price', color='b', fontsize = 20)
    ax1.tick_params('y', colors='b')
    plt.xticks(5*np.arange(0, T//5 +5 ), fontsize = 20)
    plt.yticks(fontsize = 20)
    ax2 = ax1.twinx()
    names = ['expectationa', 'expectationb']
    cols = ['r-', 'b-']
    colsdot = ['r-.', 'b-.']
    for i in np.arange(0, len(names)):
        t2 = extrapdiag['t'] 
        s2 = extrapdiag[names[i]]['beta']
        s3 = extrapdiag[names[i]]['betalow'] 
        s4 = extrapdiag[names[i]]['betahigh'] 
    
        ax2.plot(t2, s2, cols[i])
        ax2.plot(t2, s3, colsdot[i])
        ax2.plot(t2, s4, colsdot[i])
    plt.yticks(fontsize = 20)
    ax2.set_ylabel('Local betas', color='r', fontsize = 20)
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.show()

    fig, ax1 = plt.subplots(figsize = (15, 8))
    plt.title("Expected price change extrapolation, rational", fontsize = 30)
    t = np.arange(1, T+1)
    s1 = modelobjectrat['p']
    ax1.plot(t, s1, 'b-', alpha = 0.2)
    ax1.set_xlabel('T')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Average Price', color='b', fontsize = 20)
    ax1.tick_params('y', colors='b')
    plt.xticks(5*np.arange(0, T//5 +5 ), fontsize = 20)
    plt.yticks(fontsize = 20)
    ax2 = ax1.twinx()
    names = ['expectationa', 'expectationb']
    cols = ['r-', 'b-']
    colsdot = ['r-.', 'b-.']
    for i in np.arange(0, len(names)):
        t2 = extraprat['t'] 
        s2 = extraprat[names[i]]['beta']
        s3 = extraprat[names[i]]['betalow'] 
        s4 = extraprat[names[i]]['betahigh'] 
    
        ax2.plot(t2, s2, cols[i])
        ax2.plot(t2, s3, colsdot[i])
        ax2.plot(t2, s4, colsdot[i])
    plt.yticks(fontsize = 20)
    ax2.set_ylabel('Local betas', color='r', fontsize = 20)
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.show()
    