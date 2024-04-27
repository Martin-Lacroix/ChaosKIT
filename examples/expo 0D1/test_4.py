import numpy as np
import chaoskit as ck
from fun import response

# %% Initialisation

ordPoly = 7
ordQuad = 2*ordPoly
dist = ck.Normal(1,0.5)

# %% Polynomial Chaos

point,weight = ck.tensquad(ordQuad,dist)
resp = response(point)

poly = ck.polyrecur(ordPoly,dist)
coef = ck.spectral(resp,poly,point,weight)
model = ck.Expansion(coef,poly)

ck.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')