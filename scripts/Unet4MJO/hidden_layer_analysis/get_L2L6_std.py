import sys
sys.path.append('../util/')
import numpy as np
import xarray as xr 
import matplotlib.pyplot as plt 
from scipy import special
import math 
import matplotlib.colors as colors
import hidden_util as hid 
import torch
import pickle
import os 


"""
conda activate eofenv
# module load pytorch/1.11.0
salloc --nodes 1 --qos interactive --time 03:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

"""

vn = 'olr'
lat_lim = 20
mjo_ind = os.environ["mjo_ind"]  # RMM or ROMI
leadmjo = int(os.environ["leadmjo"]) # lead for output (the MJO index)
m = 1
mflg = 'off' 
wnx = 1
wnxflg = 'off'
zmode = 1 
nmem = 1
if mjo_ind == 'RMM':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
else:
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'
dataflg = 'new'
# calculate the m-k field for a given model

# define parameters for the function
params = {
    'vn': vn,
    'lat_lim': lat_lim,
    'mjo_ind': mjo_ind,
    'leadmjo': leadmjo,
    'm': m,
    'mflg': mflg,
    'wnx': wnx,
    'wnxflg': wnxflg,
    'zmode': zmode,
    'nmem': nmem,
    'dataflg': dataflg,
    'Fnmjo': Fnmjo,
    'outputflg': ''
}

# hid.get_mkfield(**params)

if dataflg=='new':
    dataflg = ''

# load the data
with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/outfields'+vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+dataflg+str(20)+'.pickle', 'rb') as file2:
    outfields = pickle.load(file2)
    
l2 = np.concatenate(outfields['hidden1'], axis=0)  # [time, channel, lat, lon]
l6 = np.concatenate(outfields['hidden5'], axis=0)  # [time, channel, lat, lon]

l2 = np.maximum(l2, 0)  # relu
l6 = np.maximum(l6, 0)  # relu

# calculate the std
l2_std = np.std(l2, axis=0)
l6_std = np.std(l6, axis=0)

# close the file
file2.close()

l6l2_std = np.concatenate([l6_std, l2_std], axis=0)  # [channel, lat, lon]

# save the data
flag = vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)

with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/std_hid26relu'+flag+'.pickle', 'wb') as file2:
    pickle.dump(l6l2_std, file2)

file2.close()

print('done')