#! /usr/bin/env python

# imports of external packages to use in our code
import sys
import numpy as np
import matplotlib.pyplot as plt

# just an exponential function
def exp(x):
    return np.exp(x)

# trapezoidal rule
def trap_method(f, a, b, n):
    
    # creates n evenly-spaced values from a to b
    x = np.linspace(a, b, n-1)

    fx = f(x)

    h = (b - a)/(n-1)

    # trapezoidal sum using array slicing

    trap_est = (h/2) * ( fx[1:] + fx[:-1] ).sum()

    return trap_est

# Gauss-Legendre method
def gauss_legendre(f, xis, weights):

    return np.sum( weights * f(xis) )

# main function
if __name__ == "__main__":
    # if the user includes the flag -h or --help print the options
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s -Nmax [maximum number of trapezoidal subdivisions]" % sys.argv[0])
        print
        sys.exit(1)
    
    Nmax = 50

    GLMax = 50

    if '-Nmax' in sys.argv:
        p = sys.argv.index('-Nmax')
        Ne = int(sys.argv[p + 1])
        if Ne > 0:
            Nmax = Ne

    # interval of [-1, 1] since the Gauss-Legendre method works over that interval

    a = -1.
    b = 1.

    # integral of exp(x) is trivial
    ana_result = exp(b) - exp(a)
    
    trap_ests = []
    
    n = 2
    # generating trapezoidal estimates
    while n <= Nmax:
        trap_ests.append(trap_method(exp, a, b, n))
        n += 1
    
    trap_ests = np.array(trap_ests)

    GL_ests = []
    
    n = 2
    # generating Gauss-Legendre estimates
    while n <= GLMax:
        fname = 'data/weights' + str(n) + '.dat'
        data = np.loadtxt(fname)
        xis = data[:, 0]
        weights = data[:, 1]

        GL_ests.append(gauss_legendre(exp, xis, weights))
        
        n += 1
    
    GL_ests = np.array(GL_ests)
    
    # subtract the analytical result off
    trap_err = trap_ests - ana_result
    
    GL_err = GL_ests - ana_result

    # plotting
    ntrap = np.arange(2, Nmax+1)
    nGL = np.arange(2, GLMax+1)
    plt.figure(figsize=[12, 7])

    plt.plot(ntrap, trap_err, 'ro', label='Trapezoidal estimate')
    plt.plot(nGL, GL_err, 'bo', label='Gauss-Legendre estimate')

    # we can do this since n is the same for both
    plt.plot(ntrap, trap_ests - GL_ests, 'kd', label='Trapezoidal - Gauss-Legendre', alpha=0.75, markersize=5)

    plt.xlabel('n')
    plt.ylabel('True value - Integral estimate')
    plt.legend()

    plt.title('Difference between Analytical value and Estimate up to n = 50')
    
    plt.savefig('Analytical_difference.pdf')

    plt.show()

