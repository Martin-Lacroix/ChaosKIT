import numpy as np
import chaoskit as ck
from fun import sampler
from fun import response

# %% Initialisation

order = 7
nbrPts = 1000
pdf = ck.Normal(1,0.5).pdf
dom = [-1,3]

# %% Polynomial Chaos

point = sampler(nbrPts)
resp = response(point)

poly = ck.gschmidt(order,point)
coef = ck.colloc(resp,poly,point)
model = ck.Expansion(coef,poly)

ck.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')