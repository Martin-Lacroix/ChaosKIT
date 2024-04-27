import numpy as np
import chaoskit as ck
from fun import sampler
from fun import response

# %% Initialisation

order = 30
nbrPts = int(1e5)
dist = ck.Normal(1,0.5)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ck.gschmidt(order,point)
index,weight = ck.fekquad(point,poly)

poly.trunc(15)
point = point[index]

resp = response(point)
coef = ck.spectral(resp,poly,point,weight)
model = ck.Expansion(coef,poly)

ck.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')