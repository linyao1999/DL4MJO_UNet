# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account=dasrepo

import sys
sys.path.append('../util')
import contribution_util as ctr 
import numpy as np
import xarray as xr 

lat_avg = 20
relu = False
rm_wn0 = True

# ROMI
order_zero = np.asarray( 
    [178, 163, 141, 134, 171, 187, 162, 169, 170, 168, 172, 145, 161,
       164, 156, 150, 137, 185, 180, 139, 131, 179, 151, 138, 159, 190,
       191, 189, 157, 129, 152, 177, 174, 142, 184, 144, 182, 158, 173,
       132, 153, 135, 133, 181, 175, 167, 160, 176, 155, 183, 140, 154,
       186, 148, 136, 149, 130, 147, 165, 166,  45, 122, 143,  93, 146,
         6, 113,  33,  70,  35, 111, 128,  91,  84,  79, 100,  39, 188,
        80,  42, 106,  53,  25,  99,  90,  15,  78,  57,  59,   2,  16,
        12, 101, 120,  83,  28, 114,  48, 125,  26,  81,  75,  58,  51,
       123, 127,  24,  97, 126,  55,  65, 102,  61,  32, 116,  56,  29,
         7,   3,  62,  73, 105,  10,  46,  71,   5, 115,  64,  66, 103,
        17,  68,  37,  67,  13,  27,  76,  41, 121,   0,  92,  30,  47,
       112, 124, 118,  98,  31,  49,  54,   8, 109,  21,  72, 107,  77,
        19,  22,  63,  43,  50,  38, 110,  23,  40,   4, 117,  85,  20,
         1,  95,  69, 104,  60,  86,  14,  52,  36,  34,  82, 119,  96,
        87,  89, 108,  74,  88,  94,  11,   9,  18,  44]
)

channel_sel = order_zero
freq, power, power_norm = ctr.get_fft_power(mjo_ind='ROMI', order=channel_sel, lat_avg=lat_avg, relu=relu, lead=25, rm_wn0=rm_wn0)
freq0, power0, power_norm0 = ctr.get_fft_power_input(mjo_ind='ROMI', lat_avg=lat_avg, lead=25, rm_wn0=rm_wn0)

data_vars = {
    'freq':(['freq'],freq),
    'channel':(['channel'],channel_sel),
    'power':(['channel','freq'],power),
    'power_norm':(['channel','freq'],power_norm),
    'power0':(['freq'],np.squeeze(power0)),
    'power_norm0':(['freq'],np.squeeze(power_norm0))
}

ds = xr.Dataset(data_vars)
if relu:
    if rm_wn0:
        ds.to_netcdf('./scale_change/ROMI_fft_hid26_rmsd_'+str(lat_avg)+'deg_zero_channel_rmwn0.nc')
    else:
        ds.to_netcdf('./scale_change/ROMI_fft_hid26_rmsd_'+str(lat_avg)+'deg_zero_channel.nc')
else: 
    if rm_wn0:
        ds.to_netcdf('./scale_change/ROMI_fft_hid26_rmsd_'+str(lat_avg)+'deg_zero_channel_beforerelu_rmwn0.nc')
    else:
        ds.to_netcdf('./scale_change/ROMI_fft_hid26_rmsd_'+str(lat_avg)+'deg_zero_channel_beforerelu.nc')


order_ptb = np.asarray(
    [143, 175, 166, 171, 155, 140, 134, 167, 182, 151, 163, 186, 147,
       168, 180, 189, 191, 177, 183, 135, 181, 142, 164, 185, 130, 160,
       190, 129, 170, 157, 131, 152, 169, 178, 158, 128, 149, 138, 145,
       153, 172, 161, 188, 165, 174, 179, 133, 144, 162, 132, 136, 154,
       176, 148, 139, 187, 137, 156, 146, 173, 150, 184, 159, 141,  49,
       111, 105,  42,  43,  91,  47,  27,  96, 114,  36, 108,  61,  28,
        81, 120,  69,  93,  55,  75,  66,  38,   0,  62,  95,  63,  78,
        79, 123,   4,  80,  39,  86,  52, 107,  88,  34,  37,  82,  99,
        16,  35, 104,  57,   9, 106,  14,  18,  60, 122,  25,  13,  74,
        10,  20,  11,   1, 125,  19,  21,  90,  89, 112,  65,  70,  30,
         8,  77,  44,  26,  50,  98,  41, 113,  84,  23,  22,  51,   2,
        24,  15,  87,   3, 121, 109,  54,  33, 115,  68,   5,  29,  58,
        40,  67,  92,   6,  46, 101,  64,  59, 118,  71, 116, 117,  12,
        76,  94,  48,  73, 110,   7,  31,  97, 126,  83,  45,  72, 103,
       124,  85, 119, 127,  53,  32, 100, 102,  56,  17]
)

channel_sel = order_ptb
freq, power, power_norm = ctr.get_fft_power(order=channel_sel, lat_avg=lat_avg, relu=relu, rm_wn0=rm_wn0, mjo_ind='ROMI', lead=25)
# freq0, power0, power_norm0 = ctr.get_fft_power_input()

data_vars = {
    'freq':(['freq'],freq),
    'channel':(['channel'],channel_sel),
    'power':(['channel','freq'],power),
    'power_norm':(['channel','freq'],power_norm),
    'power0':(['freq'],np.squeeze(power0)),
    'power_norm0':(['freq'],np.squeeze(power_norm0))
}

ds = xr.Dataset(data_vars)
if relu:
    if rm_wn0:
        ds.to_netcdf('./scale_change/ROMI_fft_hid26_rmsd_'+str(lat_avg)+'deg_ptb_channel_rmwn0.nc')
    else:
        ds.to_netcdf('./scale_change/ROMI_fft_hid26_rmsd_'+str(lat_avg)+'deg_ptb_channel.nc')
else:
    if rm_wn0:
        ds.to_netcdf('./scale_change/ROMI_fft_hid26_rmsd_'+str(lat_avg)+'deg_ptb_channel_beforerelu_rmwn0.nc')
    else:
        ds.to_netcdf('./scale_change/ROMI_fft_hid26_rmsd_'+str(lat_avg)+'deg_ptb_channel_beforerelu.nc')

# # Load the dictionary from the file
# vn = 'olr'
# lat_lim = 20
# mjo_ind = 'RMM' 
# leadmjo = 15

# # mjo_ind = 'ROMI' 
# # leadmjo = 25

# m=1
# mflg = 'off' 
# wnx = 1
# wnxflg = 'off'
# zmode = 1
# nmem = 1
# dataflg = '' 
# flag = vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+dataflg

# with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/outfields'+vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+'.pickle', 'rb') as file2:
#     outfields = pickle.load(file2)

# # For each layer in the outfiels
# # the shape of the input_map is (time, batch, channel, lat, lon)
# # reshape the fields to (time*batch, channel, lat, lon)
# # for each time step, perform forier transform for each time step across longitudes
# # average over all time steps and latitudes

# outfields_fft = {}

# for layer_name in outfields.keys():
#     time, batch, channel, lat, lon = np.shape(outfields[layer_name])
#     feature_maps = np.reshape(outfields[layer_name], (time*batch, channel, lat, lon))

#     # apply ReLu
#     feature_maps = np.maximum(feature_maps, 0)

#     feature_maps_fft = np.fft.fft(feature_maps, axis=-1)
#     feature_maps_fft_power = np.mean(np.abs(feature_maps_fft)**2, axis=(0,2))  # average over time and latitudes (channel, lon)
#     feature_maps_fft_power_norm = feature_maps_fft_power / np.sum(feature_maps_fft_power, axis=-1, keepdims=True)

#     outfields_fft[layer_name] = feature_maps_fft_power_norm

#     print(layer_name)

# # store the results
# with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/outfields_fft_only_'+vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+'.pickle', 'wb') as file3:
#     pickle.dump(outfields_fft, file3)
