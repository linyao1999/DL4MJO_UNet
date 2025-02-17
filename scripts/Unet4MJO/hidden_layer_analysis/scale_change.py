# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint cpu --account=dasrepo

import sys
sys.path.append('../util')
import contribution_util as ctr 
import numpy as np
import xarray as xr 

lat_avg = 20
relu = True
rm_wn0 = True

order_zero = np.asarray(
    [175, 143, 189, 166, 167, 140, 182, 191, 180, 168, 183, 171, 163,
       147, 164, 185, 155, 129, 151, 190, 181, 177, 133, 134, 131, 178,
       135, 145, 153, 165, 160, 128, 158, 130, 169, 188, 132, 161, 172,
       152, 186, 187, 170, 142, 154, 174, 149, 144, 137, 162, 148, 157,
       136, 138, 156, 179, 184, 173, 176,  85,  36, 139, 146,  17,  21,
       150,  53,  32,  42, 107,  49, 159, 123,  44, 141, 121,  80,  15,
        99, 105, 102,  19, 113, 115, 125,  31,  58,  57, 124, 114,  26,
        54,  63,  56,  48,  98,  61,  13,  46,  92,  81, 101,  41,  67,
        71,  23,  29, 120,  91,  51,  33,  90, 117,  34, 100,  12,  22,
        18,  76, 127,   4,  40,  95, 126, 106,  74,  59,  60, 108, 109,
        27,  65,  83,   6,  45,  96,  37,  79, 111,  43,  73,   7,  24,
       104,  52,  77,  10,  87,  68,  89,  64,  69,   5,  88,   8,  35,
        14,   1,  94, 119, 110,   9,  66,  62,   2,  55, 112,  70, 116,
        86, 118,  84,  20,  11, 122,   3,  97,  16,  72,  50,  25,  47,
        93,   0,  28,  82,  38,  39, 103,  78,  30,  75]
)

channel_sel = order_zero
freq, power, power_norm = ctr.get_fft_power(order=channel_sel, lat_avg=lat_avg, relu=relu, rm_wn0=rm_wn0)
freq0, power0, power_norm0 = ctr.get_fft_power_input(lat_avg=lat_avg, rm_wn0=rm_wn0)

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
        ds.to_netcdf('./scale_change/fft_hid26_rmsd_'+str(lat_avg)+'deg_zero_channel_rmwn0.nc')
    else:
        ds.to_netcdf('./scale_change/fft_hid26_rmsd_'+str(lat_avg)+'deg_zero_channel.nc')
else: 
    if rm_wn0:
        ds.to_netcdf('./scale_change/fft_hid26_rmsd_'+str(lat_avg)+'deg_zero_channel_beforerelu_rmwn0.nc')
    else:
        ds.to_netcdf('./scale_change/fft_hid26_rmsd_'+str(lat_avg)+'deg_zero_channel_beforerelu.nc')


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
freq, power, power_norm = ctr.get_fft_power(order=channel_sel, lat_avg=lat_avg, relu=relu, rm_wn0=rm_wn0)
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
        ds.to_netcdf('./scale_change/fft_hid26_rmsd_'+str(lat_avg)+'deg_ptb_channel_rmwn0.nc')
    else:
        ds.to_netcdf('./scale_change/fft_hid26_rmsd_'+str(lat_avg)+'deg_ptb_channel.nc')
else:
    if rm_wn0:
        ds.to_netcdf('./scale_change/fft_hid26_rmsd_'+str(lat_avg)+'deg_ptb_channel_beforerelu_rmwn0.nc')
    else:
        ds.to_netcdf('./scale_change/fft_hid26_rmsd_'+str(lat_avg)+'deg_ptb_channel_beforerelu.nc')

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
