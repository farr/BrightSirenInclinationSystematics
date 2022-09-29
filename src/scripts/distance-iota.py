import arviz as az
import matplotlib.pyplot as plt
import os.path as op
import paths
import seaborn as sns

sns.set_theme(context='paper', style='ticks')

trace = az.from_netcdf(op.join(paths.data, 'GW170817-like-chain.nc'))
post = az.extract_dataset(trace.posterior)

sns.kdeplot(data=post, x='d', y='x')
plt.xlabel(r'$d$')
plt.ylabel(r'$x = \cos \iota$')
plt.tight_layout()
plt.ylim(bottom=-1, top=-0.6)

plt.axhline(-0.99, color='k')
plt.axvline(0.13200333333333333, color='k')

plt.savefig(op.join(paths.figures, 'distance-iota.pdf'))
