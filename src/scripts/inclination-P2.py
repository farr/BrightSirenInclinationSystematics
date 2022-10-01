import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import seaborn as sns
import sys
import paths

sns.set_theme(context='paper', style='ticks')

trace = az.from_netcdf(op.join(paths.data, 'Pl_2_chains.nc'))

d = ['chain', 'draw']

pd = trace.posterior.P_det
x = pd.coords['x']
l, = plt.plot(x, pd.median(dim=d), label='Fit')
plt.fill_between(x, pd.quantile(0.84, dim=d), pd.quantile(0.16, dim=d), color=l.get_color(), alpha=0.25)
plt.fill_between(x, pd.quantile(0.975, dim=d), pd.quantile(0.025, dim=d), color=l.get_color(), alpha=0.25)

pd_true = np.exp(0.5*(3*x*x-1)) # This is a hack---should really include models.py
pd_true = pd_true / np.trapz(pd_true, x)
plt.plot(x, pd_true, color='k', label='Truth')

plt.xlabel(r'$x$')
plt.ylabel(r'$P_{\mathrm{det},\mathrm{EM}}\left( x \right)$')
plt.legend()
plt.tight_layout()

plt.savefig(op.join(paths.figures, 'inclination-P2.pdf'))