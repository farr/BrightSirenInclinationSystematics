import numpy as np
import pymc as pm
import pytensor
import pytensor.printing
import pytensor.tensor as pt
import pytensor.tensor.extra_ops as pte
from tqdm import tqdm
import xarray as xr

def draw_gws(F_em, N, fr=1.0, fl=1.0, h=0.7, dmax=0.5, rho_thresh=10, F_thresh=25):
    """Returns `(em_detected, x_true, d_true, z_true, Al_true, Ar_true, Al, Ar,
    Ndraw)` for a population draw from our model.

    `F_em` returns the expected EM counts in a Poisson detection model with
    threshod `F_thresh` given a (cosine) inclination and distance:
    `counts_expected = F_em(x,d)`.  If the observed counts are above `F_thresh`
    then the source is considered EM detected.

    The GW observations are assumed independent measurements of the left and
    right circular polarizations, each with unit scale, additive Gaussian noise.
    `fr` and `fl` give the couplings of the true L and R flux (equal to ``(1
    \\pm x)^2/d``) into the observed flux:

    ..math::

        h_{obs,R/L} = f_{R/L} \\frac{\\left( 1 \\pm x \\right)^2}{d} + N(0,1)

    If the S/N, defined by 

    ..math::

        \\rho^2 \\equiv h_{obs,R}^2 + h_{obs,L}^2

    is larger than `rho_thresh` the source is considered to be GW detected.

    The cosmology assumed is Euclidean, so $d = h z$.

    The function returns a set of GW-detected events with arrays
    `(em_detected_flag, x_true, d_true, z_true, h_L, h_R, h_L_obs, h_R_obs,
    Ndraw)`; `Ndraw` is the number of events simulated to produce the GW
    detections returned.
    """
    em_detected = []
    x_true = []
    d_true = []
    z_true = []
    Al_true = []
    Ar_true = []
    Al = []
    Ar = []

    Ndraw = 0
    with tqdm(total=N) as bar:
        while len(x_true) < N:
            Ndraw = Ndraw + 1
            d = dmax*np.cbrt(np.random.uniform(low=0, high=1))
            x = np.random.uniform(low=-1, high=1)
            R = np.square(1+x)/d
            L = np.square(1-x)/d

            Robs = np.random.normal(loc=fr*R, scale=1)
            Lobs = np.random.normal(loc=fl*L, scale=1)

            rho2 = (Robs*Robs + Lobs*Lobs)/2
            if rho2 > rho_thresh*rho_thresh:
                # GW detection!
                x_true.append(x)
                d_true.append(d)
                z_true.append(d*h)
                Al_true.append(L)
                Ar_true.append(R)
                Al.append(Lobs)
                Ar.append(Robs)

                F_ex = F_em(x, d)
                F_obs = np.random.poisson(F_ex)

                if F_obs > F_thresh:
                    em_detected.append(True)
                else:
                    em_detected.append(False)

                bar.update(1)

    em_detected, x_true, d_true, z_true, Al_true, Ar_true, Al, Ar = \
        map(np.array, [em_detected, x_true, d_true, z_true, Al_true, Ar_true, Al, Ar])
    return (em_detected, x_true, d_true, z_true, Al_true, Ar_true, Al, Ar, Ndraw)

def log_half_erfc(x):
    sqrt_pi = np.sqrt(np.pi)
    return pt.where(x < -5, pt.log1p(pt.exp(-x*x)*(1/x*(1/(2*sqrt_pi) - 1/(4*sqrt_pi*x*x)))),
                    pt.where(x > 5, -x*x + pt.log(1/x*(1/(2*sqrt_pi) - 1/(4*sqrt_pi*x*x))),
                             pt.log(0.5*pt.erfc(x))))

def log_diff_exp(x, y):
    return x + pt.log1p(-pt.exp(y-x))

def Pl(x, l):
    if l == 0:
        return 1
    elif l == 1:
        return x
    else:
        Plm1 = Pl(x, l-1)
        Plm2 = Pl(x, l-2)
        return ((2*l-1)*x*Plm1 - (l-1)*Plm2)/l
    
def make_basic_pe_model(Al_obs, Ar_obs, dmax=1, Fl=1, Fr=1):
    with pm.Model() as model:
        x = pm.Uniform('x', -1, 1)
        V = pm.Uniform('V', 0, dmax*dmax*dmax)
        d = pm.Deterministic('d', V**(1/3))

        Al = pm.Deterministic('Al', pt.square(1-x)/d)
        Ar = pm.Deterministic('Ar', pt.square(1+x)/d)

        _ = pm.Normal('Ar_obs', mu=Fr*Ar, sigma=1, observed=Ar_obs)
        _ = pm.Normal('Al_obs', mu=Fl*Al, sigma=1, observed=Al_obs)
    return model
    
def make_model(Al_obs, Ar_obs, z, x_det, d_det, Ndraw, Fl=1, Fr=1, correct_em=False, log_L0_guess=None, hmin=0.35, hmax=1.4, lmax=4, xinterp=np.linspace(-1, 1, 128)):
    if correct_em and log_L0_guess is None:
        raise ValueError('must specify log_L0_guess when correct_em==True')

    nobs = len(Al_obs)

    coords = {
        'nobs': np.arange(nobs),
        'l': np.arange(lmax)+1,
        'xinterp': xinterp
    }

    def log_inc_dependent_flux(x, fl):
        inc_dependent_flux = 0
        for l in coords['l']:
            inc_dependent_flux += Pl(x, l)*fl[l-1]
        return inc_dependent_flux        

    def log_Pdet(x, d, log_L0, fl, sigma):
        inc_dependent_flux = log_inc_dependent_flux(x, fl)
        erfc_arg = (log_L0 - inc_dependent_flux + 2*pt.log(d))/(np.sqrt(2)*sigma)
        return log_half_erfc(erfc_arg)

    with pm.Model(coords=coords) as model:
        h = pm.Uniform('h', hmin, hmax)

        x = pm.Uniform('x', -1, 1, dims='nobs')

        # \int \dd d \, \delta(z - h d) p(d) = z^2 / h^3
        d = pm.Deterministic('d', z/h, dims='nobs')
        pm.Potential('d_integral', -3*nobs*pt.log(h)) # z is a constant, since it's data.

        if correct_em:
            fl = pm.Normal('fl', 0, 1, dims='l')
            log_L0 = pm.Normal('log_L0', log_L0_guess, 1)
            sigma = pm.Truncated('sigma', pm.Normal.dist(0, 1), lower=0.1)

            log_Pd = log_Pdet(x, d, log_L0, fl, sigma)
            pm.Potential('Pdet_x_d', pt.sum(log_Pd))

            pm.Deterministic('inc_dependent_flux_factor', pt.exp(log_inc_dependent_flux(xinterp, fl)), dims='xinterp')

        Al = pm.Deterministic('Al', Fl*pt.square(1-x)/d, dims='nobs')
        Ar = pm.Deterministic('Ar', Fr*pt.square(1+x)/d, dims='nobs')

        _ = pm.Normal('Al_likelihood', mu=Al, sigma=1, observed=Al_obs)
        _ = pm.Normal('Ar_likelihood', mu=Ar, sigma=1, observed=Ar_obs)

        if correct_em:
            log_pdet = log_Pdet(x_det, d_det, log_L0, fl, sigma)
            log_pdet2 = 2*log_pdet

            log_mu = pt.logsumexp(log_pdet) - pt.log(Ndraw)
            log_s2 = log_diff_exp(pt.logsumexp(log_pdet2) - 2*pt.log(Ndraw), 2*log_mu - pt.log(Ndraw))

            Neff = pm.Deterministic('Neff', pt.exp(2*log_mu - log_s2))
            _ = pm.Potential('selection', -nobs*log_mu)

    return model