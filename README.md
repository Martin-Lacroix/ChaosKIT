## Introduction

Arbitrary polynomial chaos toolkit for high dimensional stochastic problems in Python, with correlated random variables. The code can automatically build orthogonal polynomials with respect to an arbitrary joined probability density function of the input, provided it has finite moments. The examples and doc folders contain some test-cases as well as a documentation.

## Installation

First, make sure to work with Python 3 and install the last version of Scipy. Some functionalities may not be available while using older packages. Then, add the main repository folder to your Python path environment variables. Another possibility is to add the path to ChaosKIT in your Python script
```sh
export PYTHONPATH=path-to-chaoskit
```
```python
from sys import path
path.append('path-to-chaoskit')
import chaoskit
```
