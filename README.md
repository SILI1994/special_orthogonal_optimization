# special_orthogonal_optimization

This is a python implementation of this [paper](https://link.springer.com/article/10.1007/s10107-012-0584-1) for optimization over SO(n) groups.


The main philosophy lies in relaxing the geodesic to a retraction via Cayley transform instead of a single tangent vector as done in projected gradient descent.


## Details of implementation

1. Armijo condition for monotone line search. 
2. Samples from Haar distribution for multiple start.


## Dependency
pip install numpy scipy joblib
