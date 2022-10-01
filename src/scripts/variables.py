import arviz as az
import os.path as op
import paths

dim = ['chain', 'draw']

def H0_range(file):
    trace = az.from_netcdf(op.join(paths.data, file))
    h = trace.posterior.h.stack(dim=dim)
    return h.median(), h.quantile(0.84), h.quantile(0.16)
def H0_format(m, h, l):
    return '{:.4f}^{{+{:.4f}}}_{{-{:.4f}}}'.format(m, h-m, m-l)
    

# H0 from P2
with open(op.join(paths.output, 'H0-P2.txt'), 'w') as out:
    out.write(H0_format(*H0_range('Pl_2_chains.nc')))

with open(op.join(paths.output, 'H0-P2-nofit.txt'), 'w') as out:
    out.write(H0_format(*H0_range('Pl_2_no_fit_chains.nc')))

# H0 from top hat
with open(op.join(paths.output, 'H0-top-hat.txt'), 'w') as out:
    out.write(H0_format(*H0_range('top_hat_chains.nc')))
with open(op.join(paths.output, 'H0-top-hat-flat.txt'), 'w') as out:
    out.write(H0_format(*H0_range('top_hat_flat_inclination_chains.nc')))