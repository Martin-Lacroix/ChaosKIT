import numpy as np
import chaoskit as ck
from fun import response
from matplotlib import pyplot as plt

# %% Initialisation

ordPoly = 20
ordQuad = 10*ordPoly
dist = ck.Normal(1,0.5)

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
for i in range(mean.shape[0]):
    plt.plot(mean[i],'C0')
    plt.plot(meanMc[i],'C1--')

plt.legend(['ChaosKIT','Monte Carlo'])
plt.ylabel('Mean')
plt.show()

plt.figure(2)
for i in range(var.shape[0]):
    plt.plot(var[i],'C0')
    plt.plot(varMc[i],'C1--')

plt.legend(['ChaosKIT','Monte Carlo'])
plt.ylabel('Variance')
plt.show()