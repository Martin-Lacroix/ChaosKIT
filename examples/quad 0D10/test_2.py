import numpy as np
import chaoskit as ck
from fun import sampler,response

# %% Initialisation

order = 3
nbrPts = int(1e4)

# %% Polynomial Chaos

point = sampler(nbrPts)
poly = ck.gschmidt(order,point)
resp = response(point)

coef,index = ck.lars(resp,poly,point,it=80)
coef = coef[index]
poly.clean(index)

coef = ck.colloc(resp,poly,point)
model = ck.Expansion(coef,poly)

ck.save(model,'model')
mean,var = [model.mean,model.var]

# %% Figures

meanMc = np.load('mean.npy')
varMc = np.load('var.npy')