import numpy as np
import scipy as sp
import scipy.special
import numba
from function_generator import FunctionGenerator

@numba.njit
def __Iv(x, v, tol=1e-15, maxiter=1000):
    """
    For large real x, we can write:
    Iv(x) = e^x/sqrt(2*pi*x)*__Iv(x)
    Where __Iv(x) is a power series in 1/x 
    I've tetsed v=0, 1, which converge for |x|<20.

    These are very useful for computing ratios of Bessel functions
    for x with |x| large
    (i.e. I1(x)/I0(x), which --> 1.0 as x --> infty)
    """
    z = 1.0
    it = 1
    add = 1.0
    alpha = 4.0*v**2
    while np.abs(add) > tol:
        add *= -1.0*(alpha-(2*it-1)**2)/(8.0*x*it)
        z += add
        it += 1
        if it > maxiter:
            raise Exception('__Iv iteration failed to converge')
    return z

def zeta(x):
    """
    Returns I1(x)/I0(x), using standarad funcitons for small |x|
    And asymptotic expansions (see doc for __Iv) for |x|
    """
    return __Iv(x, 1)/__Iv(x, 0) if np.abs(x) > 20 \
        else sp.special.iv(1, x)/sp.special.iv(0, x)

# construct vectorized version of zeta
zetas = np.vectorize(zeta)
# construct an approximation of zeta(x)/x using FunctionGenerator, good on [-1,1]
# note the strange bounds: this just avoids the removable singularity at 0
_zeta_over_x = FunctionGenerator(lambda x: zetas(x)/x, -1.001, 1.0, tol=1e-14)
# construct a function for zeta(x)/x, good on [-Inf, Inf]
def zeta_over_x(x):
    if np.abs(x) > 1:
        return zeta(x)/x
    else:
        return _zeta_over_x(x)
# returns the Bingham Function, Eq ...
def bingham_function(x):
    return 0.5 * (1.0 + zeta(x))
# returns the Bingham Jacobian, Eq ...
def bingham_jacobian(x):
    zl = zeta(x)
    return 0.5 * (1.0 - zeta_over_x(x) - zl*zl)
# computes the integral of the fourth moment, Eq ...
def integral(x):
    return 0.5 - 0.25*zeta_over_x(x) + 0.5*zeta(x)
# computes the solution to the nonlinear equation given by Eq ...
def mu_to_lambda(mu):
    l = 0.5
    err = bingham_function(l) - mu
    itcount = 0
    while np.abs(err) > 1e-14:
        l -= err/bingham_jacobian(l)
        err = bingham_function(l) - mu
        itcount += 1
        if itcount > 1000:
            raise Exception('mu_to_lambda newton iteration failed to converge.')
    return l
# computes S0000(mu), checking the range first
def _func(x):
    if x < 0.5 or x > 1.0:
        raise Exception('x must be in [0.5, 1.0]')
    elif x > 1-1e-8:
        return 1.0
    else:
        return integral(mu_to_lambda(x))
# vectorized version of S0000(mu)
func = np.vectorize(_func)
# build interpolation routine for S0000(mu)
interper = FunctionGenerator(func, 0.5, 1.0, tol=1e-14)

def twod_bingham_closure(D, E):
    """
    Direct Estimation of Bingham Closure (through rotation)
    """
    # some basic checks for D and E
    assert D.shape == E.shape, "Shape of D and E must match"
    assert len(D.shape) > 2, "D must have at least 3 dimensions"
    assert D.shape[0] == 2 and D.shape[1] == 2, "D must be 2x2 in leading dimensions"
    # reshape D and E for use by this function
    in_sh = D.shape
    sh = list(D.shape[:2]) + [np.prod(D.shape[2:]),]
    D = D.reshape(sh)
    E = E.reshape(sh)
    # transpose D for call to Eig routine
    Dd = np.transpose(D, (2,0,1))
    # compute eigenvalues and eigenvectors
    # PARALLELIZE THIS OPERATION FOR PERFORMANCE
    EV = np.linalg.eigh(Dd)
    Eval = EV[0][:,::-1]
    Evec = EV[1][:,:,::-1]
    mu = Eval[:,0]
    # enforce eigenvalue constraint if it isn't quite met
    mu[mu<0.5] = 0.5
    mu[mu>1.0] = 1.0
    # compute S0000 in the rotated frame
    tS0000 = interper(mu)
    # compute S0011 and S1111 in rotated frame according to known identities
    tS0011 = Eval[:,0] - tS0000
    tS1111 = Eval[:,1] - tS0011
    # transform to real coordinates (for S0000 and S0001)
    l00, l01, l10, l11 = Evec[:,0,0], Evec[:,0,1], Evec[:,1,0], Evec[:,1,1]
    S0000 = l00**4*tS0000 + 6*l01**2*l00**2*tS0011 + l01**4*tS1111
    S0001 = l00**3*l10*tS0000 + (3*l00*l01**2*l10+3*l00**2*l01*l11)*tS0011 + l01**3*l11*tS1111
    # get the other necessary pieces of the closure via identites
    S0011 = D[0,0] - S0000
    S1111 = D[1,1] - S0011
    S0111 = D[0,1] - S0001
    # perform contractions
    SD = np.zeros_like(D)
    SD[0,0,:] = S0000*D[0,0] + 2*S0001*D[0,1] + S0011*D[1,1]
    SD[0,1,:] = S0001*D[0,0] + 2*S0011*D[0,1] + S0111*D[1,1]
    SD[1,1,:] = S0011*D[0,0] + 2*S0111*D[0,1] + S1111*D[1,1]
    SD[1,0,:] = SD[0,1]
    SE = np.zeros_like(E)
    SE[0,0,:] = S0000*E[0,0] + 2*S0001*E[0,1] + S0011*E[1,1]
    SE[0,1,:] = S0001*E[0,0] + 2*S0011*E[0,1] + S0111*E[1,1]
    SE[1,1,:] = S0011*E[0,0] + 2*S0111*E[0,1] + S1111*E[1,1]
    SE[1,0,:] = SE[0,1]
    # reshape the output back to the original shape
    SD = SD.reshape(in_sh)
    SE = SE.reshape(in_sh)
    return SD, SE

def twod_bingham_closure_short(D):
    """
    Direct Estimation of Bingham Closure (through rotation)
    Returns S0000 and S0001
    (all other components can be computed from these)
    """
    # some basic checks for D
    assert len(D.shape) > 2, "D must have at least 3 dimensions"
    assert D.shape[0] == 2 and D.shape[1] == 2, "D must be 2x2 in leading dimensions"
    # reshape D for use by this function
    in_sh = D.shape
    sh = list(D.shape[:2]) + [np.prod(D.shape[2:]),]
    D = D.reshape(sh)
    # transpose D for call to Eig routine
    Dd = np.transpose(D, (2,0,1))
    # compute eigenvalues and eigenvectors
    # PARALLELIZE THIS OPERATION FOR PERFORMANCE
    EV = np.linalg.eigh(Dd)
    Eval = EV[0][:,::-1]
    Evec = EV[1][:,:,::-1]
    mu = Eval[:,0]
    # enforce eigenvalue constraint if it isn't quite met
    mu[mu<0.5] = 0.5
    mu[mu>1.0] = 1.0
    # compute S0000 in the rotated frame
    tS0000 = interper(mu)
    # compute S0011 and S1111 in rotated frame according to known identities
    tS0011 = Eval[:,0] - tS0000
    tS1111 = Eval[:,1] - tS0011
    # transform to real coordinates (for S0000 and S0001)
    l00, l01, l10, l11 = Evec[:,0,0], Evec[:,0,1], Evec[:,1,0], Evec[:,1,1]
    S0000 = l00**4*tS0000 + 6*l01**2*l00**2*tS0011 + l01**4*tS1111
    S0001 = l00**3*l10*tS0000 + (3*l00*l01**2*l10+3*l00**2*l01*l11)*tS0011 + l01**3*l11*tS1111
    # reshape these for return
    out_sh = in_sh[2:]
    S0000 = S0000.reshape(out_sh)
    S0001 = S0001.reshape(out_sh)
    return S0000, S0001
