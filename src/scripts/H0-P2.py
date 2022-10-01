import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import paths
import seaborn as sns

sns.set_theme('paper', 'ticks')

trace = az.from_netcdf(op.join(paths.data, "Pl_2_chains.nc"))
trace_fixed = az.from_netcdf(op.join(paths.data, "Pl_2_fixed_chains.nc"))
trace_nofit = az.from_netcdf(op.join(paths.data, 'Pl_2_no_fit_chains.nc'))

dim = ['chain', 'draw']

sns.kdeplot(trace.posterior.h.stack(dim=dim), label=r'Fit $P_{\mathrm{det},\mathrm{EM}}$')
sns.kdeplot(trace_nofit.posterior.h.stack(dim=dim), label=r'Ignore $P_{\mathrm{det},\mathrm{EM}}$')

plt.axvline(0.7, color='k')

plt.xlabel(r'$H_0$')
plt.ylabel(r'$p\left( H_0 \right)$')
plt.legend()
plt.tight_layout()

plt.savefig(op.join(paths.figures, 'H0-P2.pdf'))