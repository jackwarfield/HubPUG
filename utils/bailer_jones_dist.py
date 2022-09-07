'''
(C) Joel C. Zinn 2018
zinn.44@osu.edu

Given an input L, parallax, and parallax error, will return the mode of the posterior for r.
Based on the method from Bailer-Jones 2015.
'''

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import minimize_scalar, bracket, minimize
import functools
from scipy.stats import norm as normal

def main(L, plx, plx_err):
    '''
    Inputs
    L : float | ndarray
     exponential cutoff length of the exponentially decreasing Volume prior.
    plx : float | ndarray
     measured parallax.
    plx_err : float | ndarray
     error on the measured parallax.
    Outputs
    r : float | ndarray
     estimate for the distance, which would, if plx_err = 0., be 1/plx.
    Notes:
     All the inputs must be in the same units (i.e., if L is in kpc, plx and plx_err need to be in mas).
    '''
    if type(plx) != type(plx_err):
        print('plx is not the same type as plx_err. Both must either be float or ndarray.')

    if type(plx) == type(0.0) or type(plx) == np.float64:
        # the mode is given by the solution to a cubic, whose coefficients starting with the r^3 term coefficient and decrease in the index is:
        coeffs = [1./L, -2., plx/plx_err**2, -1./plx_err**2]
        roots = np.roots(coeffs)
        real_roots = np.array([r for x in roots for r, i in zip([x.real], [x.imag])  if np.abs(i) < 1e-15])
        if len(real_roots) == 0:
            print('warning! bailer_jones_dist.py found no valid distance based on the measured parallax of {}, scale length, L = {}, and parallax error of {}. giving NaN instead.'.format(plx, L, plx_err))
            return np.nan
        real_pos_roots = real_roots[real_roots > 0.]
        if len(real_pos_roots) == 0:
            print('warning! bailer_jones_dist.py found no valid distance based on the measured parallax of {}, scale length, L = {}, and parallax error of {}. giving NaN instead.'.format(plx, L, plx_err))
            return np.nan
        r = np.min(real_pos_roots)
    
        return r
    elif type(plx) == type(np.array([])):
        rs = []
        for _plx, _plx_err in zip(plx, plx_err):
            # the mode is given by the solution to a cubic, whose coefficients starting with the r^3 term coefficient and decrease in the index is:
            coeffs = [1./L, -2., _plx/_plx_err**2, -1./_plx_err**2]
            try:
                roots = np.roots(coeffs)
            except:
                print('warning! bailer_jones_dist.py found no valid distance based on the measured parallax of {}, scale length, L = {}, and parallax error of {}. giving NaN instead.'.format(_plx, L, _plx_err))
                rs.append(np.nan)
                continue
            real_roots = np.array([r for x in roots for r, i in zip([x.real], [x.imag])  if np.abs(i) < 1e-15])
            if len(real_roots) == 0:
                print('warning! bailer_jones_dist.py found no valid distance based on the measured parallax of {}, scale length, L = {}, and parallax error of {}. giving NaN instead.'.format(_plx, L, _plx_err))
                rs.append(np.nan)
                continue
            real_pos_roots = real_roots[real_roots > 0.]
            if len(real_pos_roots) == 0:
                print('warning! bailer_jones_dist.py found no valid distance based on the measured parallax of {}, scale length, L = {}, and parallax error of {}. giving NaN instead.'.format(_plx, L, _plx_err))
                rs.append(np.nan)
                continue
            rs.append(np.min(real_pos_roots))
        rs = np.array(rs)
        return rs
    else:
        print('type of plx and plx_err were not recognized. exiting and returning None. They must be type float or ndarray.')
        return
    


def p(r, L, plx, plx_err):
    '''
    returns unnormalized posterior for r, given scale length L, plx, and plx_err to be used when integrating over dlnr, not r!!! so this is actually p*r.
    Inputs
    L : float | ndarray
     exponential cutoff length of the exponentially decreasing Volume prior.
    plx : float | ndarray
     measured parallax.
    plx_err : float | ndarray
     error on the measured parallax.
    r_lim : float
     maximum allowed r (minimum is 0.0).
    Outputs
    p : float | ndarray
     unnormalized posterior
    '''

    arg = -0.5/plx_err**2*(plx - 1./np.exp(r))**2
    # print arg, plx, 1./np.exp(r)
    # return r**2/plx_err*np.exp(arg)
    return np.exp(r)**3/plx_err*np.exp(arg)*np.exp(-np.exp(r)/L)
    # return 1./plx_err*np.exp(arg)

def jac(r,  L=100.0, plx=1.0, plx_err=0.01,u=0.5,norm=2.70249,r_lim=[0.1,10.0], integrator=integrate.quad):
    '''
    returns dfunc/dlnr
    DOESN'T WORK
    '''
    
    f = (-np.exp(r)**3/L + 2.0*np.exp(r)**2 - plx/plx_err**2*np.exp(r) + 1./plx_err**2)
    abs_fac = p(r, L, plx, plx_err) - u
    # print np.sign(func(r,u=u,L=L,plx=plx,plx_err=plx_err,norm=norm,r_lim=r_lim,_abs=False, integrator=integrator))
    # print p(r, L, plx, plx_err)/np.exp(r)
    return p(r, L, plx, plx_err)/np.exp(r)*np.sign(func(r,u=u,L=L,plx=plx,plx_err=plx_err,norm=norm,r_lim=r_lim,_abs=False, integrator=integrator))#/(np.exp(r)**3)*f*abs_fac/np.abs(abs_fac)
        


def integrate_w_grid(_func, lim0, lim1, args=[], points=None, limit=None):
    '''
    this is meant to have the same call signature as integrate.quad. will integrate by rectangular method by ensuring that
    there are 100 resolution elements in the +/- 2 plx_err region of the region around 1/plx. this is necessary for parallaxes that are more precise than about 20%
    because the quad integrator fails for those.
    args : ndarray
     must be in the order of [L, plx, plx_err]
    points : None
     not used.
    limit : None
     not used.
    lim0 : float
     the lower limit of integration, in ln(r)
    lim1 : float
     the upper limit of integration, in ln(r)
    _func : function
     the function to be integrated.
    '''
    L = args[0]
    plx = args[1]
    plx_err = args[2]
    # desired number of resultion elements in the +/- 2 plx_err region
    N_res = 1000.0
    lower = np.log10(1./(plx+2.0*plx_err))

    func = _func
    if plx > 0.0 and 2.0*plx_err < plx:
    
            # upper = lower + 0.5
            # N_res = 0.5/(100.0*np.log(1./plx))
    
        upper = np.log10(1./(plx-2.0*plx_err))
            
        res = (-lower + upper)/N_res
        # print 'resolution of {}'.format(res)

        # import pylab as plt
        # plt.clf()
        # plt.plot((grid), p(*call_args))
        # plt.savefig('tmp.pdf', format='pdf')

    else: # for plx < 0.0, relevant length scale will be L --- no matter what the error is. because in the case where the error is small, the peak will be at r=infinity, which is ruled out by the exponentially decreasing prior. in teh case where the error is large, there is no information in the data. for intermediate values, IDK !!!
        # for 2.0*plx_err >= plx, there is no info in the parallax so the exponential scale should be correct.
        res = np.log(L)/N_res

        
    grid = (np.arange(lim0, lim1, res))
    integral = [np.sum(func(grid, L, plx, plx_err))*res]        
    

    # print integral
    return integral
    
    
def func(r, u=0.5, L=100.0, plx=1.0, plx_err=0.01, norm=2.70249, r_lim=[0.1,10.0],_abs=True, integrator=integrate.quad):
    '''
    returns the absolute value of the difference between the cumulative probability distribution, F(r), and u: |F(r) - u|. must supply the normalization to the probability function such that \int_{r_lim[0]}^{r_lim[1]} p *norm = 1.0
    Inputs
    _abs : bool
     if False, will return F(r) - u instead of |F(r) - u|. Default True.
    u : float
     probability, u, that enters into |F(r) - u|
    r : float
     distance, r, that enters into |F(r) - u|
    '''
    if integrator == integrate.quad:
        points = np.log([1./plx])
        if not (r_lim[0] < points[0] and r > points[0]):
            points = None
    else:
        points = None
    if _abs:
        return np.abs(integrator(p, r_lim[0], r, args=(L, plx, plx_err), points=points, limit=1000)[0]/norm - u)
    else:
        return (integrator(p, r_lim[0], r, args=(L, plx, plx_err), points=points, limit=1000)[0]/norm - u)

def draw(L, _plx, _plx_err, _r_lim, debug=False, u=None):
    '''
    will draw from the posterior of the distance, by integrating \int_r_lim[0]_x p(r)dr to get the cumulative distribution, F(x)
    and then drawing u from a random distribution from [0,1] and finding where F(x) for x \epsilon [r_lim[0],r_lim[1]] = u. You should use a very small, but nonzero, r_lim[0].
    This method will not return reliable distances when plx_err/plx <= 0.1, so for those cases, it is simply drawn from the distribution of N(1/plx, plx_err/plx^2). will return scalar if _plx and _plx_err are scalars, otherwise will output ndarray. For 
    Inputs
    [ u : float | None ]
     Scalar float to request what percentile you would like to be drawn from the posterior. If this is set to non-None, draw will not actually give random draws, but only draws that correspond to draws at the uth percentile of the distribution (and will be the exact same number each time...). this is good for specifying errors (see get_err()). Default None.
    L : float
     exponential cutoff length of the exponentially decreasing Volume prior.
    _plx : float | ndarray
     measured parallax.
    _plx_err : float | ndarray
     error on the measured parallax.
    _r_lim : float | ndarray
     allowed range for l.
    [ debug : bool ]
     if True, will make some plots
    Outputs
    r : float | ndarray
     r drawn from the posterior, with a maximum value of r_lim
    '''
    plx_ = np.atleast_1d(_plx)
    plx_err_ = np.atleast_1d(_plx_err)
    if len(plx_) != len(plx_err_):
        print('length of _plx and _plx_err are not the same, at: {} and {}'.format(len(plx_), len(plx_err_)))
        print('returning scalar NaN')
        return np.nan

    result = []

    for plx, plx_err in zip(plx_, plx_err_):
        if u is None: 
            _u = np.random.uniform(0.0,1.0)
        else:
            _u = u
        if plx_err/plx > 0.1 or plx <= 0.0: # use the posterior method because the errors are not going to be very symmetric in distance space
            
            # u = 0.9
            integrator = integrate_w_grid
            # the quad integrator runs at 37ms, whereas my simple grid integrator runs at 15ms. the difference for the case of draw(1.35, 0.05, 0.002, [0.01,100.0]) is: 19.663070582777461 (quad) 19.671532126893212 (grid), fixing u manually in the code to be u = 0.5. this is an error of 0.04%, and this code should not be used to compute distances when the fractional parallax error is less than 10% anyway. so this is an acceptable tolerance. plus, for larger absolute parallax, quad seems to fail. so sticking with integrate_w_grid for timing and accuracy! increasing N from 1000 to 10000 gives: 19.671875708087143 --- i.e., away from the quad solution, not closer to it. so i think quad is wrong in this case. all the better!
            # however, the grid method seems to fail above plx \gtrsim 1.6 with 10% errors... so rounding down to 1.5 and above that, use quad. which does OK for those large parallaxes.
            if plx > 1.5:
                integrator = integrate.quad

            r_lim = np.log(_r_lim)

            # JCZ 130418
            # warning: this fails when plx = plx_err
            # points = np.log([1./(plx-plx_err), 1./plx, 1./(plx+plx_err)])
            # points = (np.linspace(np.log(1./(plx+plx_err*2.0)), np.log(1./(plx-plx_err*2.0)), num=4))[::-1]

            if integrator == integrate.quad:
                points = np.log([1./plx])
            else:
                points = None

            norm = integrator(p, r_lim[0], r_lim[1], args=(L, plx, plx_err), points=points)[0]
            if norm == 0.0:
                print('could return a distance for plx {:4.3e} {:4.3e} because of underflow in the integral'.format(plx, plx_err))
                print('these cases have found to happen for negative plx and very small plx_err, in which case the exp(plx - 1/r) makes non-infinite r so unlikely as to underflow the calculation of the integral over the posterior probability. returning NaN.')
                result.append(np.nan)
                continue
            function = functools.partial(func, u=_u, L=L, plx=plx, plx_err=plx_err, norm=norm, r_lim=r_lim, integrator=integrator)
            jac_function = functools.partial(jac, u=_u, L=L, plx=plx, plx_err=plx_err, norm=norm, r_lim=r_lim, integrator=integrator)            

            # JCZ 130418
            # using method='Golden' does not work, but 'Bounded' does, which uses Brent method.
            # r_gs = minimize_scalar(function, bracket=(r_lim[0],  r_lim[1]), method='Golden').x
            r_b = minimize_scalar(function, bounds=(r_lim[0], r_lim[1]),  method='Bounded').x
            # JCZ 130418
            # this really does not work... even with the help of the jacobian.
            # print 'Brent search distance:'
            # print np.exp(r_b)
            # r_bfgs = minimize(function, np.log([1./plx]), bounds=[(r_lim[0], r_lim[1])], jac=jac_function, method='L-BFGS-B').x[0]
            # print 'L-BFGS-B distance:'
            # print np.exp(r_bfgs)



            # the below only plots the region +/- 4plx_err around 1/plx, so it will not show all of the posterior probability! in particular,
            # the percentiles can look wrong because there is more probability not being plotted !!!
            lower = np.log10(1./(plx+4.0*plx_err))
            if 4.0*plx_err >= plx:
                upper = lower + 0.5
            else:
                upper = np.log10(1./(plx-4.0*plx_err))
            x = np.logspace(lower, upper, num=500)
            # normalize the mode of the probability density to be 1.0
            logr_mode = np.log(main(L,plx, plx_err))
            norm = p(logr_mode, L, plx, plx_err)/np.exp(logr_mode)
            if debug:
                import pylab as plt
                plt.clf()
                for _x in np.log(x):
                    # this is the difference between the desired probability and the cumulative probability distribution at _x (log(r))
                    plt.scatter(_x, function(_x))
                    # this is the actual probability density, divided by r (np.exp(_x)) because p() actually returns p*r (to do the cumulative probability calculation in steps of dlnr)
                    plt.scatter(_x, p(_x, L, plx, plx_err)/np.exp(_x)/norm)
                    # this is the derivative of function(_x) (i.e., it is just the actual probability density, with a sign switch when function(_x) = u
                    plt.scatter(_x, jac_function(_x))
                # this is the r for which F(r) = u using L-BFGS-B method
                # plt.axvline(r_bfgs, label='L-BFGS-B', color='red')
                # this is the r for which F(r) = u using Brent search method
                plt.axvline(r_b, label='Brent', color='gold')
                # this is the mode from Bailer-Jones+ 2016
                plt.axvline(logr_mode)
                plt.legend()
                plt.savefig('tmp.pdf', format='pdf')
            result.append(np.exp(r_b))
        else:
            # JCZ 151018
            # replacing this with the scipy function, so that can use this to get errors. otherwise, this function would draw something randomly, and not according to the requested <u>.
            result.append(normal.ppf(_u, loc=1./plx, scale=plx_err/plx**2))
            # result.append(np.random.normal(loc=1./plx, scale=plx_err/plx**2))

    if len(plx_) == 0: # return a scalar
        result = result[0]
    else:
        result = np.array(result)
    return result
def get_err(L, _plx, _plx_err, _r_lim, debug=False):
    '''
    Returns (lower, upper, mean), where upper and lower are the 1-sigma errors from the posterior and mean is the sum of those two divided by two.
    args
    see draw()
    '''
    from my_routines import sigma_to_percent
    u_upper = 1. - (1.0 - sigma_to_percent(1.0))/2.0
    u_lower = (1.0 - sigma_to_percent(1.0))/2.0
    upper = draw(L, _plx, _plx_err, _r_lim, debug=debug, u=u_upper)
    lower = draw(L, _plx, _plx_err, _r_lim, debug=debug, u=u_lower)
    dist = main(L, _plx, _plx_err)
    return (dist-lower, upper-dist, (upper - lower)/2.0)
