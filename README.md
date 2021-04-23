# special_orthogonal_optimization

This is a python implementation of this [paper](https://link.springer.com/article/10.1007/s10107-012-0584-1) for the following optimization problem:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5CLarge%0A%5Cmin%20%5C%20f%5Cleft(%5Cmathbf%7BR%7D%5Cright)%20%5Cquad%20s.t.%20%5C%20%5C%20%5Cmathbf%7BR%7D%20%5Cin%20%5Cmathrm%7BSO%7D%5Cleft(n%5Cright)">
</p>

The main philosophy lies in relaxing the geodesic to a retraction via Cayley transform instead of a single tangent vector as done in projected gradient descent.


## Details of implementation

1. Armijo condition for monotone line search. 
2. Samples from Haar distribution for multiple start.


## Dependency
`pip install numpy scipy joblib`
