import numpy as np
import chaoskit as ck
from fun import response

# %% Initialisation

ordPoly = 8
ordQuad = 2*ordPoly
dist = [ck.Uniform(-np.pi,np.pi) for i in range(3)]

# %% Polynomial Chaos

point,weight = ck.tensquad(ordQuad,dist)
poly = ck.polyrecur(ordPoly,dist)

resp = response(point)
coef = ck.spectral(resp,poly,point,weight)
model = ck.Expansion(coef,poly)

ck.save(model,'model')
sobol = ck.anova(coef,poly)
mean,var = [model.mean,model.var]
index,ancova = ck.ancova(model,point,weight)

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')