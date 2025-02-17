# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account=dasrepo

import sys
sys.path.append('../util')
import contribution_util as ctr 
import numpy as np
import xarray as xr 
import os

mjo_ind = os.environ['mjo_ind']
lead = os.environ['lead']

lat_avg = 20
relu = False
rm_wn0 = True

channel_sel = np.arange(0,192)
freq, power, power_norm = ctr.get_fft_power(mjo_ind=mjo_ind, lead=lead, order=channel_sel, lat_avg=lat_avg, relu=relu, rm_wn0=rm_wn0)
freq0, power0, power_norm0 = ctr.get_fft_power_input(mjo_ind=mjo_ind, lead=lead, lat_avg=lat_avg, rm_wn0=rm_wn0)

data_vars = {
    'freq':(['freq'],freq),
    'channel':(['channel'],channel_sel),
    'power':(['channel','freq'],power),
    'power_norm':(['channel','freq'],power_norm),
    'power0':(['freq'],np.squeeze(power0)),
    'power_norm0':(['freq'],np.squeeze(power_norm0))
}

ds = xr.Dataset(data_vars)
ds.to_netcdf('./scale_change/'+mjo_ind+'_fft_hid26_L2L6_lead'+str(lead)+'_beforerelu_rmwn0.nc')

