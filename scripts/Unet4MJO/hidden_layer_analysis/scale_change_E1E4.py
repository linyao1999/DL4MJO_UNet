# salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu --account=dasrepo

import sys
sys.path.append('../util')
import contribution_util as ctr 
import numpy as np
import xarray as xr 
import os
import pickle

mjo_ind = 'RMM'
lead = 15
m=1
mflg='off'
wnx=1
wnxflg='off'

# mjo_ind = 'RMM'
# lead = 10
# m=10
# mflg='resi'
# wnx=9
# wnxflg='resi'

relu = False
rm_wn0 = False

channel_sel = np.arange(0,192)
freq, power_norm = ctr.get_fft_power_E1E4(vn='olr',lat_lim=20,mjo_ind=mjo_ind,lead=lead,m=m,mflg=mflg,wnx=wnx,wnxflg=wnxflg,relu=relu, rm_wn0=rm_wn0)
_, _, power_norm0 = ctr.get_fft_power_input(mjo_ind=mjo_ind,lead=lead,m=m,mflg=mflg,wnx=wnx,wnxflg=wnxflg,rm_wn0=rm_wn0)

power_norm['input'] = power_norm0

# store the power_norm
if relu:
    if rm_wn0:
        with open(f'/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/power_norm_relu_m{m}{mflg}wnx{wnx}{wnxflg}_{mjo_ind}_{lead}.pkl', 'wb') as f:
            pickle.dump(power_norm, f)
    else: 
        with open(f'/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/power_norm_relu_m{m}{mflg}wnx{wnx}{wnxflg}_{mjo_ind}_{lead}_keepwnx0.pkl', 'wb') as f:
            pickle.dump(power_norm, f)
else:
    if rm_wn0:
        with open(f'/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/power_norm_m{m}{mflg}wnx{wnx}{wnxflg}_{mjo_ind}_{lead}.pkl', 'wb') as f:
            pickle.dump(power_norm, f)
    else:
        with open(f'/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/power_norm_m{m}{mflg}wnx{wnx}{wnxflg}_{mjo_ind}_{lead}_keepwnx0.pkl', 'wb') as f:
            pickle.dump(power_norm, f)
