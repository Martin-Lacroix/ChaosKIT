# <img src="Python.svg" width="60"/> Distributions

<br />

ChaosKIT provides basic classes of well-know one-dimensional probability distributions for which the three terms recurrence coefficients can be computed analytically. For instance:

<br />

```python
dist = Beta(a,b)            # Beta distribution
dist = Normal(a,b)          # Normal distribution
dist = Uniform(a,b)         # Uniform distribution
```

| Input             | Type              | Description                                   |
|-------------------|-------------------|-----------------------------------------------|
| *a*               | *float*           | *first parameter of the distribution*         |
| *b*               | *float*           | *second parameter of the distribution*        |

<br />

Each of these classes contains different built-in methods.

<br />

```python
dist.pdf(x)             # Probability density function
dist.cdf(x)             # Cumulative distribution function
dist.invcdf(x)          # Inverse cumulative distribution function
dist.coef(n)            # Computes the n first recurrence coefficients
```

| Input             | Type                      | Description                               |
|-------------------|---------------------------|-------------------------------------------|
| *x*               | *float or (n) array*      | *points to be evaluated*                  |

<br />

Finally, independent one-dimensional distributions can be joint together, for instance:

<br />

```python
dist = Joint([Normal(a,b),Beta(a,b)])       # Joint probability density function
```

| Input             | Type              | Description                               |
|-------------------|-------------------|-----------------------------------------------|
| *a*               | *float*           | *first parameter of the distribution*         |
| *b*               | *float*           | *second parameter of the distribution*        |

<br />

And allow the computation to the joint probability distribution function and the generation of a random sample or a low-discrepancy sequences. The parameter `d` is the dimension of the density.

<br />

```python
dist.pdf(x)                     # Joint probability density function
dist.sampler(n)                 # Generates n samples from the distribution
dist[i] = Uniform(a,b)          # Updates the i-th distribution
```

| Input             | Type                      | Description                               |
|-------------------|---------------------------|-------------------------------------------|
| *x*               | *float or (n,d) array*    | *exponent table*                          |
| *n*               | *int*                     | *number of points or coefficients*        |
| *i*               | *int*                     | *index of the distribution to update*     |

<br />

# <img src="Python.svg" width="60"/> Polynomial Basis

<br />

A polynomial basis can be constructed by with an exponent table and a coefficient matrix. The parameter `d` is the dimension, `m` the number of monomials and `p` is the number of polynomials.

<br />

```python
poly = Polynomial(expo,coef,csr=0)      # Class of olynomial basis
```

| Input             | Type                  | Description                       |
|-------------------|-----------------------|-----------------------------------|
| *expo*            | *(d,m) array*         | *exponent table*                  |
| *coef*            | *(p,m) array*         | *coefficients matrix*             |
| *csr*             | *bool*                | *for sparse csr format*           |

<br />

The class of polynomial basis contains some in-built methods such as the evaluation of its polynomials at an array of `n` point:

<br />

```python
V = poly.eval(point)            # Computes the Vandermonde matrix
poly.clean(index)               # Selects the relevant polynomials
poly.trunc(order)               # Truncates the polynomial basis
```

| Input             | Type                  | Description                                   |
|-------------------|-----------------------|-----------------------------------------------|
| *point*           | *(n,d) array*         | *points to evaluate polynomials*              |
| *index*           | *(-) array*           | *indices of the polynomials to keep*          |
| *order*           | *int*                 | *order of truncation*                         |

<br />

An orthonormal polynomial basis with respect to a sample can be generated by Gram Schmidt process. The parameter `weight` must be provided if the points are quadrature nodes, otherwise a Monte Carlo integration is assumed.

<br />

```python
poly = gschmidt(order,point,weight=0,trunc=1)       # Gram-Schmidt process
```

| Input             | Type                  | Description                                   |
|-------------------|-----------------------|-----------------------------------------------|
| *order*           | *int*                 | *maximum order of the polynomials*            |
| *point*           | *(n,d) array*         | *points for the quadrature*                   |
| *weight*          | *(n) array*           | *weights associated to the points*            |
| *trunc*           | *float*               | *hyperbolic truncation q-norm*                |

<br />

Similarly, orthogonal polynomials can be constructed using a three terms recurrence coefficients relation related to a well-knows distribution or joint distribution.

<br />

```python
poly = polyrecur(order,dist,trunc=1)        # Three terms recurrence relation
```

| Input             | Type                          | Description                                           |
|-------------------|-------------------------------|-------------------------------------------------------|
| *order*           | *int*                         | *maximum order of the polynomials*                    |
| *dist*            | *object or array*             | *joint distribution or list of distributions*         |
| *trunc*           | *float*                       | *hyperbolic truncation norm*                          |

<br />

# <img src="Python.svg" width="60"/> Quadrature Rules

<br />

A tensor product quadrature rule with respect to the probability density function of a well-known distribution or joint distribution can be generated by

<br />

```python
point,weight = tensquad(order,dist)         # Tensor product quadrature
```

| Input             | Type                          | Description                                           |
|-------------------|-------------------------------|-------------------------------------------------------|
| *order*           | *int*                         | *order of the quadrature rule*                        |
| *dist*            | *object or array*             | *joint distribution or list of distributions*         |

<br />

Different sparse quadrature rules for a polynomial basis can be generated from a Monte Carlo integration set. The original number of points must be larger than the number of polynomials in the basis.

<br />

```python
index,weight = simquad(point,poly)              # Revised simplex algorithm
index,weight = fekquad(point,poly)              # Approximate Fekete points
index,weight = nulquad(point,poly,weight)       # Positive quadrature with null space
index,weight = newquad(point,poly,weight)       # Positive quadrature with Newton
```

| Input             | Type                  | Description                           |
|-------------------|-----------------------|---------------------------------------|
| *point*           | *(n,d) array*         | *original set of points*              |
| *poly*            | *object*              | *polynomial basis object*             |
| *weight*          | *(n) array*           | *previous weights of the points*      |

<br />

# <img src="Python.svg" width="60"/> Expansion Coefficients

<br />

The polynomial chaos coefficients can be computed by spectral projection of least squares regression. The parameter `n` is the number of points and `d` the dimension. The parameter `weight` must be provided if the points are quadrature nodes, otherwise a Monte Carlo integration is assumed.

<br />

```python
coef = spectral(resp,poly,point,weight=0)           # Spectral projection
coef = colloc(resp,poly,point,weight=0)             # Point collocation
```

| Input             | Type                  | Description                                   |
|-------------------|-----------------------|-----------------------------------------------|
| *resp*            | *(n,-) array*         | *response of the model at the points*         |
| *poly*            | *object*              | *polynomial basis object*                     |
| *point*           | *(n,d) array*         | *quadrature points*                           |
| *weight*          | *(n) array*           | *weights of the points*                       |

<br />

The least angle regression algorithm selects the relevant polynomials in addition to compute their coefficients. The parameter `index` is the indices of the selected polynomials, if `it` is not provided, all the available polynomials are selected.

<br />

```python
coef,index = lars(resp,poly,point,weight=0,it=np.inf)       # Least angle regression
coef,index = lasso(resp,poly,point,weight=0,it=np.inf)      # Least shrinkage operator
```

| Input             | Type                  | Description                                   |
|-------------------|-----------------------|-----------------------------------------------|
| *resp*            | *(n,-) array*         | *response of thr model at the points*         |
| *poly*            | *object*              | *polynomial basis object*                     |
| *point*           | *(n,d) array*         | *original set of points*                      |
| *weight*          | *(n) array*           | *weights of the points*                       |
| *it*              | *int*                 | *maximum number of iterations*                |

<br />

# <img src="Python.svg" width="60"/> Surrogate Model

<br />

The polynomial chaos model can be constructed by invoking the constructor of the expansion class. The parameter `p` is the number of polynomials in the basis. The class provides different built-in methods:

<br />

```python
model = Expansion(coef,poly)            # Class containing the surrogate model
resp = model.eval(point)                # Evaluates the surrogate model
mean = model.mean                       # Returns the mean of the output
var = model.var                         # Return the variance of the output
```

| Input             | Type                  | Description                                   |
|-------------------|-----------------------|-----------------------------------------------|
| *coef*            | *(p,-) array*         | *polynomial chaos coefficients*               |
| *poly*            | *object*              | *polynomial basis object*                     |
| *point*           | *(n,d) array*         | *points at which evaluate the model*          |

<br />

In addition, a simple polynomial mapping between two one-dimensional random variables can be computed by

<br />

```python
mapping = transfo(invcdf,order,dist)        # Transforms a distribution
y = mapping(x)                              # Mapping from the x space to the y space
```

| Input             | Type                  | Description                                       |
|-------------------|-----------------------|---------------------------------------------------|
| *invcdf*          | *callable*            | *inverse cumulative distribution function*        |
| *order*           | *int*                 | *order of the expansion*                          |
| *dist*            | *object*              | *distribution object*                             |

<br />

# <img src="Python.svg" width="60"/> Other Functions

<br />

A class of principal component analysis whitening can be created for a linearly correlated sample. The parameter `n` is the number of points and `d` is the dimension.

<br />

```python
mapping = Pca(sample)           # Class of PCA whitening
```

| Input             | Type                  | Description                           |
|-------------------|-----------------------|---------------------------------------|
| *sample*          | *(n,d) array*         | *reference sample of points*          |

<br />

The class will act as a mapping function between the whitened and the original random vectors.

<br />

```python
whitened = mapping.white(sample)            # Whitens a sample of points
sample = mapping.corr(whitened)             # Recovers the original sample
```

| Input             | Type                  | Description                                                |
|-------------------|-----------------------|------------------------------------------------------------|
| *sample*          | *(n,d) array*         | *sample from the same distribution as the reference*       |
| *whitened*        | *(n,d) array*         | *white noise sample of points*                             |

<br />

The Sobol sensitivity indices of a model can be directly obtained from the polynomial chaos coefficients by

<br />

```python
sobol = anova(coef,poly)            # Computes the Sobol sensitivity indices
```

| Input             | Type                  | Description                               |
|-------------------|-----------------------|-------------------------------------------|
| *coef*            | *(p,-) array*         | *polynomial chaos coefficients*           |
| *poly*            | *object*              | *polynomial basis object*                 |

<br />

For dependent random variables, the analysis of covariance indices can be obtained from the polynomial chaos model with

<br />

```python
index,ancova = ancova(model,point,weight=0)     # Computes the ancova indices
```

| Input             | Type                  | Description                           |
|-------------------|-----------------------|---------------------------------------|
| *model*           | *object*              | *expansion object*                    |
| *point*           | *(n,d) array*         | *quadrature points*                   |
| *weight*          | *(n) array*           | *weights of the points*               |

<br />

Finally, any objects generated by ChaosKIT can be saved in a pickle file:

<br />

```python
save(item,name)         # Saves an object in a file.pickle
```

| Input             | Type                  | Description                               |
|-------------------|-----------------------|-------------------------------------------|
| *item*            | *object*              | *any object to be saved*                  |
| *name*            | *string*              | *desired name for the object*             |
