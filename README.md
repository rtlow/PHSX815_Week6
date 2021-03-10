# PHSX815_Week5

# Numerical integration

We are integrating `exp(x)` using the trapezoidal method, the Gauss-Legendre method, and the Monte Carlo method.

## Example Usage
`generate_weights.nb` is a Mathematica notebook that can generate the Gauss-Legendre weights and abscissae. It is modified from code from https://pomax.github.io/bezierinfo/legendre-gauss.html.

`NumericalIntegration.py` can be run from the command line with the `-h` flag to display runtime options. `-NmaxT` sets the maximum number of subdivisions for the trapezoidal method. `-NmaxM` sets the maximum number of samples for the Monte Carlo method. `-transform` applies a coordinate transformation to the function before doing Monte Carlo integration.
