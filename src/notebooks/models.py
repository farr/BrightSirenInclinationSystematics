import aesara.tensor as at
import numpy as np
import pymc as pm
from tqdm import tqdm
import xarray as xr

def log_P_det(x, A, N=None):
    """Return the log of the detection probability as a function of `x = cos(iota)`.
    
    Our model is that the array `A` gives the coefficients of a Legendre polynomial expansion of the detection efficiency.

    Expansions up to `P_6` are supported.
    """
    if N is None:
        N = len(A)

    Pnm1 = 1
    Pn = x
    output = A[0]*Pn

    for i in range(1, N):
        Pnp1 = (x*(2*i+1)*Pn - i*Pnm1)/(i+1)

        output += A[i]*Pnp1
        Pnm1 = Pn
        Pn = Pnp1
    
    return output

def P_det(x, A):
    """The detection probability at `x = cos(iota)` with Legendre polynomial coefficients `A`.
    """
    return np.exp(log_P_det(x,A))

def draw_gws(P_det, N, fr=1.0, fl=1.0, h=0.7, dmax=0.5, rho_thresh=10):
    """Returns `(em_detected, x_true, d_true, z_true, Al_true, Ar_true, Al, Ar,
    Ndraw)` for a population draw from our model.
    
    """
    # We need to know the maximum value so we can rejection sample later
    x = np.linspace(-1, 1, 1024)
    Pdet_max = np.max(P_det(x))

    em_detected = []
    x_true = []
    d_true = []
    z_true = []
    Al_true = []
    Ar_true = []
    Al = []
    Ar = []

    N = 2048
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

                if np.random.uniform(low=0, high=Pdet_max) < P_det(x):
                    em_detected.append(True)
                else:
                    em_detected.append(False)

                bar.update(1)

    em_detected, x_true, d_true, z_true, Al_true, Ar_true, Al, Ar = \
        map(np.array, [em_detected, x_true, d_true, z_true, Al_true, Ar_true, Al, Ar])
    return (em_detected, x_true, d_true, z_true, Al_true, Ar_true, Al, Ar, Ndraw)

def logsubexp(x, y):
    """`log(exp(x)-exp(y))` but computed stably."""
    return x + at.log1p(at.exp(y-x))

def make_model(z_obs, Al_obs, Ar_obs, x_draw, Ndraw, fl=1.0, fr=1.0, fix_A = None, N_A = 2):
    N = len(z_obs)
    event_index = np.arange(N)
    LP_index = np.arange(N_A) + 1
    with pm.Model(coords={'event_index': event_index, 'LP_index': LP_index}) as model:
        h = pm.Uniform('h', lower=0.35, upper=1.4)

        if fix_A is None:
            A = pm.Normal('A', mu=0, sigma=1, dims='LP_index')
        else:
            A = pm.Deterministic('A', at.as_tensor(np.array(fix_A)), dims='LP_index')

        x = pm.Uniform('x', lower=-1, upper=1, dims='event_index')
        _ = pm.Potential('x_prior', at.sum(log_P_det(x, A, N_A)))

        d = pm.Deterministic('d', z_obs / h, dims='event_index')
        _ = pm.Potential('d_likelihood', at.sum(2*at.log(z_obs) - 3*at.log(h)))

        log_sel_wt = log_P_det(x_draw, A, N_A)
        log_sel_wt2 = 2.0*log_sel_wt

        log_mu = at.logsumexp(log_sel_wt) - at.log(Ndraw)
        log_s2 = logsubexp(at.logsumexp(log_sel_wt2) - 2*at.log(Ndraw), 2.0*log_mu - at.log(Ndraw))

        Neff = pm.Deterministic('Neff', at.exp(2.0*log_mu - log_s2))
        _ = pm.Potential('selection_effect', -N*log_mu)

        _ = pm.Normal('Ar_likelihood', mu=fr*at.square(1+x)/d, sigma=1, observed=Ar_obs, dims='event_index')
        _ = pm.Normal('Al_likelihood', mu=fl*at.square(1-x)/d, sigma=1, observed=Al_obs, dims='event_index')
    return model

def calculate_P_det(trace, Nx=256):
    A = trace.posterior.A

    c = A.coords['chain']
    d = A.coords['draw']

    x = np.linspace(-1, 1, Nx)

    pds = xr.DataArray(np.zeros((len(c), len(d), Nx)), dims=['chain', 'draw', 'x'],
                       coords={'chain': c, 'draw': d, 'x': x})
    x = pds.coords['x']
    for cc in c:
        for dd in d:
            AA = A.isel(chain=cc, draw=dd)
            p = P_det(x, AA)
            p /= np.trapz(p, x) # Normalize
            pds.loc[dict(chain=cc, draw=dd)] = p

    trace.posterior.coords['x'] = x
    trace.posterior['P_det'] = pds