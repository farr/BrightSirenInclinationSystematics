import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import paths
import seaborn as sns

sns.set_theme('paper', 'ticks')

trace_hat = az.from_netcdf(op.join(paths.data, 'top_hat_chains.nc'))
trace_nofit = az.from_netcdf(op.join(paths.data, 'top_hat_flat_inclination_chains.nc'))

dim = ['chain', 'draw']

sns.kdeplot(trace_hat.posterior.h.stack(dim=dim), label=r'Fit $P_{\mathrm{det},\mathrm{EM}}$')
sns.kdeplot(trace_nofit.posterior.h.stack(dim=dim), label='Ignore $P_{\mathrm{det},\mathrm{EM}}$')

plt.axvline(0.7, color='k')

plt.legend()

plt.xlabel(r'$H_0$')
plt.ylabel(r'$p\left( H_0 \right)$')

plt.tight_layout()
plt.savefig(op.join(paths.figures, 'H0-top-hat.pdf'))