import numpy as np
import chaoskit as ck
from fun import sampler,response

# %% Initialisation

order = 12
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ck.gschmidt(order,point)
index,weight = ck.fekquad(point,poly)

poly.trunc(6)
point = point[index]
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