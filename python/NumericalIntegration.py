#! /usr/bin/env python

# imports of external packages to use in our code
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('.')
from python.Random import Random

# just an exponential function
def exp(x):
    return np.exp(x)

# sample from a uniform distribution at y = 3
def sampleFlat(a, b, random):

    return a + (b - a) * random.rand()

# a unit funciton
def Unit(x):
    return 1

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

# Monte Carlo method
def monte_carlo(f, a, b, n, random):
    
    # volume element
    V = b - a
    
    samples = []

    i = 1
    
    # take n samples
    while i <= n:
        
        xi = sampleFlat(a, b, random)

        samples.append(f(xi))

        i += 1
    
    # here's the estimate
    integral = (V/n) *  np.sum(samples)

    return integral

# main function
if __name__ == "__main__":
    # if the user includes the flag -h or --help print the options
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s -seed [Random Seed] -NmaxT [maximum number of trapezoidal subdivisions] -NmaxM [maximum number of Monte Carlo samples]" % sys.argv[0])
        print
        sys.exit(1)
    
    NmaxT = 50

    NmaxM = 50

    GLMax = 50

    seed = 5555

    transform = False

    if '-NmaxT' in sys.argv:
        p = sys.argv.index('-NmaxT')
        Ne = int(sys.argv[p + 1])
        if Ne > 0:
            NmaxT = Ne

    if '-NmaxM' in sys.argv:
        p = sys.argv.index('-NmaxM')
        Ne = int(sys.argv[p + 1])
        if Ne > 0:
            NmaxM = Ne

    if '-seed' in sys.argv:
        p = sys.argv.index('-seed')
        Ne = int(sys.argv[p + 1])
        if Ne > 0:
            seed = Ne
    
    if '-transform' in sys.argv:
        transform = True
    random = Random(seed)


    # interval of [-1, 1] since the Gauss-Legendre method works over that interval

    a = -1.
    b = 1.

    # integral of exp(x) is trivial
    ana_result = exp(b) - exp(a)
    
    trap_ests = []
    
    n = 2
    # generating trapezoidal estimates
    while n <= NmaxT:
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
    
    MC_ests = []
    
    n = 2

    # generating Monte Carlo estimates
    while n <= NmaxM:
        MC_ests.append(monte_carlo(exp, a, b, n, random))

        n += 1

    MC_ests = np.array(MC_ests)
    

    # subtract the analytical result off
    trap_err = trap_ests - ana_result
    
    GL_err = GL_ests - ana_result

    MC_err = MC_ests - ana_result

    # plotting
    ntrap = np.arange(2, NmaxT+1)
    nGL = np.arange(2, GLMax+1)
    nMC = np.arange(2, NmaxM+1)
    plt.figure(figsize=[12, 7])

    plt.plot(ntrap, trap_err, 'ro', label='Trapezoidal error')
    plt.plot(nGL, GL_err, 'bo', label='Gauss-Legendre error')
    plt.plot(nMC, MC_err, 'go', alpha=0.25, label='Monte Carlo error')
    
    if transform == True:

        trans_ests = []

        n = 2
        
        # transform by the logarithm
        # doing this gives a flat region

        while n <= NmaxM:

            ap = exp(a)
            bp = exp(b)
            
            trans_ests.append(monte_carlo(Unit, ap, bp, n, random))

            n += 1
        
        trans_ests = np.array(trans_ests)

        trans_err = trans_ests - ana_result

        plt.plot(nMC, trans_err, 'yd', alpha=0.5, label='MC with Transformation')


    plt.xlabel('n')
    plt.ylabel('True value - Integral estimate')
    plt.legend()

    plt.title('Difference between Analytical value and Estimate up to n = 50')
    
    plt.savefig('Analytical_difference.pdf')

    plt.show()

