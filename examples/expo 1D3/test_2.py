import numpy as np
import chaoskit as ck
from fun import sampler,response
from matplotlib import pyplot as plt

# %% Initialisation

order = 7
nbrPts = int(1e3)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ck.gschmidt(order,point)
resp = response(point)

coef = ck.colloc(resp,poly,point)
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