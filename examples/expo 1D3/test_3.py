import numpy as np
import chaoskit as ck
from fun import response
from matplotlib import pyplot as plt

# %% Initialisation

ordPoly = 5
ordQuad = 2*ordPoly

dist = []
dist.append(ck.Normal(0.5,0.15))
dist.append(ck.Uniform(0.5,2.5))
dist.append(ck.Uniform(0.03,0.07))

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