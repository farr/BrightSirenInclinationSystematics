rule generate_variables:
    input:
        'src/data/Pl_2_chains.nc',
        'src/data/Pl_2_no_fit_chains.nc',
        'src/data/top_hat_chains.nc',
        'src/data/top_hat_flat_inclination_chains.nc',
        'src/data/top_hat_flat_comparison.nc'
    output:
        'src/tex/output/H0-P2.txt',
        'src/tex/output/H0-P2-nofit.txt',
        'src/tex/output/H0-top-hat.txt',
        'src/tex/output/H0-top-hat-flat.txt',
        'src/tex/output/H0-flat-comparison.txt'
    script:
        'src/scripts/variables.py'