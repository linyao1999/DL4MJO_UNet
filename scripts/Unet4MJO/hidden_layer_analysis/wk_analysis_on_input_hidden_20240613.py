import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
import pickle
import sys 

sys.path.append('../util/')

import WheelerKaladis_util as wk

# Load the dictionary from the file
vn = 'olr'
lat_lim = 20
mjo_ind = 'RMM' 
leadmjo = 15
m=1
mflg = 'off' 
wnx = 1
wnxflg = 'off'
zmode = 1
nmem = 1
dataflg = '' 
with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/outfields'+vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+dataflg+'.pickle', 'rb') as file1:
    outfields = pickle.load(file1)

fntime = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/output/predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(leadmjo)+'_dailyinput_1979to2015_c51_mem'+str(nmem)+'d.nc' 
time = xr.open_dataset(fntime)['time']

feature_map = np.asarray(outfields['hidden5'])
old_shape = feature_map.shape
new_shape = [old_shape[0]*old_shape[1], old_shape[2], old_shape[3], old_shape[4]]

feature_map = feature_map.reshape(new_shape)

hid5_sm_sym = []
hid5_sm_asym = []
hid5_background = []
hid5_sym_norm = []
hid5_asym_norm = []
hid5_sym_norm_sig = []
hid5_asym_norm_sig = []

for i in range(0, old_shape[2]):
    hid = xr.DataArray(
        feature_map[:,i,:,:].squeeze(),
        dims = ['time', 'lat', 'lon'],
        coords = {'time': time,
                  'lat': np.arange(20, -22, -2),
                  'lon': np.arange(0, 360, 2)}
    )

    sm_sym, sm_asym, background, sym_norm, asym_norm, sym_norm_sig, asym_norm_sig = wk.wk_analysis(hid, sigtest=True)

    # concatenate the dataarrays
    hid5_sm_sym.append(sm_sym)
    hid5_sm_asym.append(sm_asym)
    hid5_background.append(background)
    hid5_sym_norm.append(sym_norm)
    hid5_asym_norm.append(asym_norm)
    hid5_sym_norm_sig.append(sym_norm_sig)
    hid5_asym_norm_sig.append(asym_norm_sig)

    print(i)

# convert the list of dataarrays to a dataset
hid5_sm_sym = xr.concat(hid5_sm_sym, dim='channel')
hid5_sm_asym = xr.concat(hid5_sm_asym, dim='channel')
hid5_background = xr.concat(hid5_background, dim='channel')
hid5_sym_norm = xr.concat(hid5_sym_norm, dim='channel')
hid5_asym_norm = xr.concat(hid5_asym_norm, dim='channel')
hid5_sym_norm_sig = xr.concat(hid5_sym_norm_sig, dim='channel')
hid5_asym_norm_sig = xr.concat(hid5_asym_norm_sig, dim='channel')

# save the dataset, store all the dataarrays
flag = vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+dataflg

hid5_all = xr.Dataset(
    {
        'sm_sym': hid5_sm_sym,
        'sm_asym': hid5_sm_asym,
        'background': hid5_background,
        'sym_norm': hid5_sym_norm,
        'asym_norm': hid5_asym_norm,
        'sym_norm_sig': hid5_sym_norm_sig,
        'asym_norm_sig': hid5_asym_norm_sig
    }
)

hid5_all.to_netcdf('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/hid5_'+flag+'.nc')


import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt
import pickle
import sys 

sys.path.append('../util/')

import WheelerKaladis_util as wk

# Load the dictionary from the file
vn = 'olr'
lat_lim = 20
mjo_ind = 'RMM' 
leadmjo = 15
m=1
mflg = 'off' 
wnx = 1
wnxflg = 'off'
zmode = 1
nmem = 1
dataflg = '' 
with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/outfields'+vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+dataflg+'.pickle', 'rb') as file1:
    outfields = pickle.load(file1)

fntime = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/output/predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(leadmjo)+'_dailyinput_1979to2015_c51_mem'+str(nmem)+'d.nc' 
time = xr.open_dataset(fntime)['time']

feature_map = np.asarray(outfields['input_map'])
old_shape = feature_map.shape
new_shape = [old_shape[0]*old_shape[1], old_shape[2], old_shape[3], old_shape[4]]

feature_map = feature_map.reshape(new_shape)

hid5_sm_sym = []
hid5_sm_asym = []
hid5_background = []
hid5_sym_norm = []
hid5_asym_norm = []
hid5_sym_norm_sig = []
hid5_asym_norm_sig = []

hid = xr.DataArray(
    feature_map.squeeze(),
    dims = ['time', 'lat', 'lon'],
    coords = {'time': time,
                'lat': np.arange(20, -22, -2),
                'lon': np.arange(0, 360, 2)}
)

sm_sym, sm_asym, background, sym_norm, asym_norm, sym_norm_sig, asym_norm_sig = wk.wk_analysis(hid, sigtest=True)

# concatenate the dataarrays
hid5_sm_sym.append(sm_sym)
hid5_sm_asym.append(sm_asym)
hid5_background.append(background)
hid5_sym_norm.append(sym_norm)
hid5_asym_norm.append(asym_norm)
hid5_sym_norm_sig.append(sym_norm_sig)
hid5_asym_norm_sig.append(asym_norm_sig)

# convert the list of dataarrays to a dataset
hid5_sm_sym = xr.concat(hid5_sm_sym, dim='channel')
hid5_sm_asym = xr.concat(hid5_sm_asym, dim='channel')
hid5_background = xr.concat(hid5_background, dim='channel')
hid5_sym_norm = xr.concat(hid5_sym_norm, dim='channel')
hid5_asym_norm = xr.concat(hid5_asym_norm, dim='channel')
hid5_sym_norm_sig = xr.concat(hid5_sym_norm_sig, dim='channel')
hid5_asym_norm_sig = xr.concat(hid5_asym_norm_sig, dim='channel')

# save the dataset, store all the dataarrays
flag = vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+dataflg

hid5_all = xr.Dataset(
    {
        'sm_sym': hid5_sm_sym,
        'sm_asym': hid5_sm_asym,
        'background': hid5_background,
        'sym_norm': hid5_sym_norm,
        'asym_norm': hid5_asym_norm,
        'sym_norm_sig': hid5_sym_norm_sig,
        'asym_norm_sig': hid5_asym_norm_sig
    }
)

hid5_all.to_netcdf('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/input_'+flag+'.nc')

