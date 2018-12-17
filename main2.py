#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 10:26:52 2018

@author: ykwon
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 00:25:10 2018

@author: ykwon
"""
import os
# change working directory to where the files are located
os.chdir('/Users/ykwon/Desktop/BubbleProject/Code')

import numpy as np
import matplotlib.pyplot as plt
#import model0
import model1
import model2new
import plothelp
    
#### Joint Calibration
sigmaV = .5
sigmaS = .2
gamma = .12
T = 24
sigmaepsilon = 12.5
theta = .8
N = 5000
k = 12
### Joint Calibration
model2object1 = model2new.model2_auto(sigmaepsilon, sigmaV, sigmaS, gamma, theta, T, N)
model2object1rat = model2new.model2_auto(sigmaepsilon, sigmaV, sigmaS, gamma, 0., T, N)

model1object1 = model1.M2_noncontemp(sigmaepsilon, sigmaV, sigmaS, theta, gamma, T, k, N)
model1object1rat = model1.M2_noncontemp(sigmaepsilon, sigmaV, sigmaS, 0., gamma, T, k, N)

model2simple = model2new.model2onlyV(sigmaepsilon, sigmaV, sigmaS, gamma, theta, T)

plt.figure(figsize=(15,15))
plt.plot(np.arange(1, T+1), model2object1['p'])
plt.plot(np.arange(1, T+1), model2object1['Ed'])
plt.plot(np.arange(1, T+1), model1object1['p'])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(['Speculation, price',
            'Speculation, expected fundamentals',
            'Nonspeculative benchmark'], prop={'size': 30})

fig, ax1 = plt.subplots(figsize = (15, 8))
plt.title("Belief Dispersion", fontsize = 30)
plt.xticks(fontsize = 20)
t = np.arange(1, T+1)
s1 = model1object1['p']
s2 = np.sqrt(model1object1['disp'][1:])
plt.yticks(fontsize = 20)
ax1.set_xlabel('t', fontsize = 20)
ax2 = ax1.twinx()
plt.yticks(fontsize = 20)
ax1.plot(t, s2, 'r-')
ax1.set_ylabel('Belief Dispersion', color='r', fontsize = 20)
ax2.plot(t, s1, 'b-', alpha = 0.2)
ax2.set_ylabel('Price', color='b', fontsize = 20)
#plt.plot(np.std(model2object1['praw'], axis = 0))
#plt.plot(np.std(model1object1['praw'], axis = 0))
#plt.legend(['Vol, speculation', 'Vol, learning from prices'])
#plt.plot(model2object1['p'], alpha = 0.2)
#plt.plot(model1object1['p'], alpha = 0.2)

windows = [4, 8, 12, 16, 20]
plothelp.masterdisplay1windows(windows, model1object1, model1object1rat)
plothelp.masterdisplay2windows(windows, model2object1, model2object1rat)

#for i in np.arange(0, 10):
#    plt.plot(model2object1['praw'][i, :])
#plt.plot(model2object1['p'])


sigmaS = 0.05
model2object2 = model2new.model2_auto(sigmaepsilon, sigmaV, sigmaS, gamma, theta, T, N)
model2object2rat = model2new.model2_auto(sigmaepsilon, sigmaV, sigmaS, gamma, 0., T, N)
plothelp.masterdisplay2windows(windows, model2object2, model2object2rat, reduced = False)


# lower learning from sigmaS.
sigmaS = 1.2
model2object3 = model2new.model2_auto(sigmaepsilon, sigmaV, sigmaS, gamma, theta, T, N)
model2object3rat = model2new.model2_auto(sigmaepsilon, sigmaV, sigmaS, gamma, 0., T, N)
plothelp.masterdisplay2windows(windows, model2object3, model2object3rat)



### Joint Figure
plt.figure(figsize=(15,15))
plt.plot(np.arange(1, T+1), model2object1['p'], 'r')
plt.plot(np.arange(1, T+1), model2object3['p'], 'g')
plt.xticks(np.arange(1, T//5 + 1)*5, fontsize=30)
plt.yticks(fontsize=30)
plt.legend(['Speculation, prices highly informative',
            'Speculation, prices less informative'], prop={'size': 30})


### Joint Figure
plt.figure(figsize=(15,15))
plt.plot(np.arange(1, T+1), model2object1['p'], 'r')
plt.plot(np.arange(1, T+1), model2simple['p'], 'g')
plt.xticks(np.arange(1, T//5 + 1)*5, fontsize=30)
plt.yticks(fontsize=30)
plt.legend(['Speculation, Learning from prices',
            'Speculation, No learning from prices'], prop={'size': 30})

