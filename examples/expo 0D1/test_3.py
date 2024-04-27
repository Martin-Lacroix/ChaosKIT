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
poly = ck.gschmidt(order,point)
resp = response(point)

coef,index = ck.lars(resp,poly,point,it=10)
coef = coef[index]
poly.clean(index)

model = ck.Expansion(coef,poly)

ck.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')