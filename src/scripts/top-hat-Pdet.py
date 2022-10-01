import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import paths
import seaborn as sns

sns.set_theme('paper', 'ticks')

# Ugh---pasted from notebook.
opening_angle = 15
xmin = np.cos(opening_angle*np.pi/180)
top_hat_fraction = 0.9
def Pdet_top_hat(x):
    x = np.atleast_1d(x)
    p_broad = 3*(x*x + 0.5)/5
    p_narrow = np.where(np.abs(x)>xmin, 0.5/(1-xmin), 0)

    return (1-top_hat_fraction)*p_broad+top_hat_fraction*p_narrow

dim = ['chain', 'draw']

trace = az.from_netcdf(op.join(paths.data, 'top_hat_chains.nc'))
pd = trace.posterior.P_det
x = pd.coords['x']

l, = plt.plot(x, pd.median(dim=dim), label='Fit')
plt.fill_between(x, pd.quantile(0.84, dim=dim), pd.quantile(0.16, dim=dim), alpha=0.25, color=l.get_color())
plt.fill_between(x, pd.quantile(0.975, dim=dim), pd.quantile(0.025, dim=dim), alpha=0.25, color=l.get_color())

plt.plot(x, Pdet_top_hat(x), color='k', label='Truth')

plt.xlabel(r'$x$')
plt.ylabel(r'$P_{\mathrm{det},\mathrm{EM}}(x)$')
plt.legend()

plt.ylim(bottom=0)

plt.tight_layout()
plt.savefig(op.join(paths.figures, 'top-hat-Pdet.pdf'))