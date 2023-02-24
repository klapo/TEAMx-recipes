import numpy as np
import scipy


def ols_linear_detrend(ds, var, lnz=True, normalize=True):
    ds['{}_ols_m'.format(var)] = (('time'), np.zeros(len(ds.time.values)))
    ds['{}_ols_r'.format(var)] = (('time'), np.zeros(len(ds.time.values)))
    ds['{}_ols_p'.format(var)] = (('time'), np.zeros(len(ds.time.values)))
    ds['{}_ols_b'.format(var)] = (('time'), np.zeros(len(ds.time.values)))

    if lnz:
        z = ds.lnz
    else:
        z = ds.z

    for t in ds.time:
        ds_time = ds.sel(time=t)[var]

        result_dts = scipy.stats.linregress(z, ds_time.values)
        ds['{}_ols_m'.format(var)].loc[{'time': t}] = result_dts.slope
        ds['{}_ols_r'.format(var)].loc[{'time': t}] = result_dts.rvalue
        ds['{}_ols_p'.format(var)].loc[{'time': t}] = result_dts.pvalue
        ds['{}_ols_b'.format(var)].loc[{'time': t}] = result_dts.intercept

    ds['{}_ols_reconstructed'.format(var)] = (
                ds['{}_ols_m'.format(var)] * z + ds['{}_ols_b'.format(var)])
    ds['{}_detrend'.format(var)] = (ds[var] - ds['{}_ols_reconstructed'.format(var)])

    if normalize:
        ds['{}_norm'.format(var)] = (ds[var] - ds[var].mean(dim='z'))

    return ds
