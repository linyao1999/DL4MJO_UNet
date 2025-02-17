# This file stores all utility used by Unet


import sys
import numpy as np
import pandas as pd  
import xarray as xr 
import dask
from scipy import special
import math 
import matplotlib.pyplot as plt 
import os 

def Matsuno_kelvin(vn, time_range=['1979-01-01','2019-12-31'], lat_range=[90, -90], pic_save='./',dataflg='new'):
    # time_range: the time range of the data used in training and validating the model [inclusive]
    # lat_range: the latitude range (y) of the data used in projection

    # read data; any file with olr[time, lat, lon]
    if dataflg=='raw':
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/daily/'+vn+'.day.1978to2023.nc'
    elif dataflg=='new':
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/fltano120/'+vn+'.fltano120.1978to2023based1979to2012.nc'
    else:
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5.'+vn+'GfltG.day.1901to2020.nc'
    
    ds = xr.open_dataset(fn)

    ds1 = ds.sel(time=slice(time_range[0], time_range[1]), lat=slice(lat_range[0], lat_range[1])).fillna(0)

    olr = ds1[vn].values  # (time, lat, lon)
    lat = ds1['lat']
    lon = ds1['lon'].values
    time = ds1['time'].values

    R = 6371 * 1000
    alpha = lat.values * 2 * np.pi / 360
    # define y = lat * R
    y_unit =  alpha * R # unit: meter
    # dimensionless
    beta = 2.28e-11
    c = 50
    y = y_unit * np.sqrt(beta / c)

    # define the structure for Kelvin wave in Matsuno 1966
    phi0 = np.exp(- y**2 / 2) 
    phi0 = np.reshape(phi0, (1, len(y), 1))

    dy = y[0] - y[1]

    olrm = np.sum(olr * phi0 * dy, axis=1, keepdims=True) / np.sqrt(np.pi)  # (time, 1, lon)

    # reconstruction 
    olr_re = olrm * phi0

    olr_re_array = xr.DataArray(
        data=olr_re,
        dims=ds1[vn].dims,
        coords=ds1[vn].coords,
        attrs=ds1[vn].attrs
    )

    return olrm.squeeze(), olr_re_array # (time, lat, lon)



def Matsuno_Rossby(vn, time_range=['1979-01-01','2019-12-31'], lat_range=[90, -90], pic_save='./',dataflg='new'):
    # time_range: the time range of the data used in training and validating the model [inclusive]
    # lat_range: the latitude range (y) of the data used in projection

    # read data; any file with olr[time, lat, lon]
    if dataflg=='raw':
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/daily/'+vn+'.day.1978to2023.nc'
    elif dataflg=='new':
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/fltano120/'+vn+'.fltano120.1978to2023based1979to2012.nc'
    else:
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5.'+vn+'GfltG.day.1901to2020.nc'
    
    ds = xr.open_dataset(fn)

    ds1 = ds.sel(time=slice(time_range[0], time_range[1]), lat=slice(lat_range[0], lat_range[1])).fillna(0)

    olr = ds1[vn].values  # (time, lat, lon)
    lat = ds1['lat']
    lon = ds1['lon'].values
    time = ds1['time'].values

    R = 6371 * 1000
    alpha = lat.values * 2 * np.pi / 360
    # define y = lat * R
    y_unit =  alpha * R # unit: meter
    # dimensionless
    beta = 2.28e-11
    c = 50
    y = y_unit * np.sqrt(beta / c)

    # define the structure for Kelvin wave in Matsuno 1966
    phi0 = np.exp(- y**2 / 2) 
    phi0 = np.reshape(phi0, (1, len(y), 1))

    dy = y[0] - y[1]

    olrm = np.sum(olr * phi0 * dy, axis=1, keepdims=True) / np.sqrt(np.pi)  # (time, 1, lon)

    # reconstruction 
    olr_re = olrm * phi0

    olr_re_array = xr.DataArray(
        data=olr_re,
        dims=ds1[vn].dims,
        coords=ds1[vn].coords,
        attrs=ds1[vn].attrs
    )

    return olrm.squeeze(), olr_re_array # (time, lat, lon)
