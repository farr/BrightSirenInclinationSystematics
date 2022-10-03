import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import paths
import seaborn as sns

sns.set_theme('paper', 'ticks')

trace = az.from_netcdf(op.join(paths.data, 'top_hat_flat_comparison.nc'))

dim = ['chain', 'draw']

pd = trace.posterior.P_det
x = pd.coords['x']

l, = plt.plot(x, pd.median(dim=dim), label='Fit')
plt.fill_between(x, pd.quantile(0.84, dim=dim), pd.quantile(0.16, dim=dim), color=l.get_color(), alpha=0.25)
plt.fill_between(x, pd.quantile(0.975, dim=dim), pd.quantile(0.025, dim=dim), color=l.get_color(), alpha=0.25)

plt.plot(x, 0.5*np.ones_like(x), label='Truth', color='k')

plt.xlabel(r'$x$')
plt.ylabel(r'$P_{\mathrm{det},\mathrm{EM}}(x)$')

plt.legend()

plt.tight_layout()
plt.savefig(op.join(paths.figures, 'flat-Pdet.pdf'))