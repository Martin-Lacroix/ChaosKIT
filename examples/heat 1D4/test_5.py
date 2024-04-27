import numpy as np
import chaoskit as ck
from fun import response
from matplotlib import pyplot as plt

# %% Initialisation

ordPoly = 7
ordQuad = 3*ordPoly
dist = ck.Joint([ck.Normal(-18,2),ck.Gamma(2,0.001),ck.Uniform(1.5,3),ck.Normal(21,1)])

# %% Polynomial Chaos

point,weight = ck.tensquad(ordQuad,dist)
poly = ck.polyrecur(ordPoly,dist)
resp = response(point)

coef = ck.spectral(resp,poly,point,weight)
model = ck.Expansion(coef,poly)

ck.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

varMc = np.load('var.npy')
meanMc = np.load('mean.npy')

plt.figure(1)
plt.plot(mean,label='ChaosKIT')
plt.plot(meanMc,'--',label='Monte Carlo')
plt.ylabel('Mean')
plt.show()

plt.figure(2)
plt.plot(var,label='ChaosKIT')
plt.plot(varMc,'--',label='Monte Carlo')
plt.ylabel('Variance')
plt.show()