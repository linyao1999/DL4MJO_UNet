import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os 
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

# write a function to get the file name of the 19 variables
def get_19varfiles(mjo_ind, lead_list, exp_num='', m=1, mflg='off', wnx=1, wnxflg='off',vn='19maps'):

    lat_range = 20
    nmem = 1

    output_path = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/'+vn+'_MCDO_ERA5_yproj_xfft_4gpus_new/output'+exp_num+'/'

    fn_list = []
    for lead in lead_list:
        fn = output_path+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_1979to2015_c51_mem'+str(nmem)+'d.nc'
        fn_list.append(fn)

    return fn_list

def get_19varfiles_one(mjo_ind, lead=0, exp_num='', m=1, mflg='off', wnx=1, wnxflg='off',vn='19maps', dataflg=''):

    lat_range = 20
    nmem = 1

    output_path = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/19maps_MCDO_ERA5_yproj_xfft_4gpus_new/output'+exp_num+'/'

    fn = output_path+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_1979to2015_c51_mem'+str(nmem)+'d'+dataflg+'.nc'

    return fn

def get_1varfiles_one(mjo_ind, lead=0, exp_num='', m=1, mflg='off', wnx=1, wnxflg='off',vn='olr', dataflg='', winter=False, outputflg='', lat_range=20):

    # lat_range = 20
    nmem = 1

    if outputflg == '':
        output_path = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/output'+str(exp_num)+'/'
    else:
        output_path = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_globalmap/output'+str(exp_num)+'/'
        dataflg = 'glb'

    if winter:  
        fn = output_path+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_1979to2015_c51_mem'+str(nmem)+'dwinter.nc'
    else:
        fn = output_path+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_1979to2015_c51_mem'+str(nmem)+'d'+dataflg+'.nc'

    return fn

def get_1varfiles(vn, mjo_ind, lead_list, exp_num='', m=1, mflg='off', wnx=1, wnxflg='off', dataflg=''):
    lat_range = 20
    nmem = 1

    output_path = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/output'+exp_num+'/'

    fn_list = []
    for lead in lead_list:
        fn = output_path+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_1979to2015_c51_mem'+str(nmem)+'d'+dataflg+'.nc'
        fn_list.append(fn)

    return fn_list

# get the phase a given RMM1 and RMM2
def get_phase(RMM1, RMM2):
    if RMM1>=0 and RMM2>=0 and RMM1>=RMM2:
        return 5
    elif RMM1>=0 and RMM2>=0 and RMM1<=RMM2:
        return 6
    elif RMM1<=0 and RMM2>=0 and -RMM1<=RMM2:
        return 7
    elif RMM1<=0 and RMM2>=0 and -RMM1>=RMM2:
        return 8 
    elif RMM1<=0 and RMM2<=0 and RMM1<=RMM2:
        return 1
    elif RMM1<=0 and RMM2<=0 and RMM1>=RMM2:
        return 2 
    elif RMM1>=0 and RMM2<=0 and RMM1<=-RMM2:
        return 3
    elif RMM1>=0 and RMM2<=0 and RMM1>=-RMM2:
        return 4

def vectorized_get_phase(RMM1, RMM2):
    # RMM1 and RMM2 are 1D arrays
    phase = np.zeros_like(RMM1)  # Initialize the phase array with zeros

    phase = np.where((RMM1 >= 0) & (RMM2 >= 0) & (RMM1 >= RMM2), 5, phase)
    phase = np.where((RMM1 >= 0) & (RMM2 >= 0) & (RMM1 <= RMM2), 6, phase)
    phase = np.where((RMM1 <= 0) & (RMM2 >= 0) & (-RMM1 <= RMM2), 7, phase)
    phase = np.where((RMM1 <= 0) & (RMM2 >= 0) & (-RMM1 >= RMM2), 8, phase)
    phase = np.where((RMM1 <= 0) & (RMM2 <= 0) & (RMM1 <= RMM2), 1, phase)
    phase = np.where((RMM1 <= 0) & (RMM2 <= 0) & (RMM1 >= RMM2), 2, phase)
    phase = np.where((RMM1 >= 0) & (RMM2 <= 0) & (RMM1 <= -RMM2), 3, phase)
    phase = np.where((RMM1 >= 0) & (RMM2 <= 0) & (RMM1 >= -RMM2), 4, phase)

    return phase

# calculate the correlation coefficient
def bulk_bcc(F, O):
    # F: forecast [time, index]
    # O: observation [time, index]

    # calculate the correlation coefficient
    corr_nom = sum(F[:,0]*O[:,0] + F[:,1]*O[:,1])
    corr_denom = np.sqrt(sum(F[:,0]**2 + F[:,1]**2)*sum(O[:,0]**2 + O[:,1]**2))

    return corr_nom/corr_denom

def bulk_bcc_mcdp(F, O):
    # F: forecast [time, Nsamp, index]
    # O: observation [time, index]

    # calculate the correlation coefficient
    corr_nom = np.sum(F[:,:,0]*O[:,0] + F[:,:,1]*O[:,1], axis=0)
    corr_denom = np.sqrt(np.sum(F[:,:,0]**2 + F[:,:,1]**2, axis=0)*np.sum(O[:,0]**2 + O[:,1]**2))

    return corr_nom/corr_denom

# calculate the root mean square error
def bulk_rmse(F, O):
    # F: forecast [time, index]
    # O: observation [time, index]

    # calculate the correlation coefficient
    rmse = np.sqrt(np.mean( (F[:,0]-O[:,0])**2 + (F[:,1]-O[:,1])**2 ))

    return rmse

def bulk_rmse_mcdp(F, O):
    # F: forecast [time, Nsamp, index]
    # O: observation [time, index]

    # calculate the correlation coefficient
    rmse = np.sqrt(np.mean( (F[:,:,0]-O[:,0])**2 + (F[:,:,1]-O[:,1])**2, axis=0 ))

    return rmse

# calculate the amplitude error
def amp_error(F, O):
    # F: forecast [time, index]
    # O: observation [time, index]

    AF = np.sqrt(F[:,0]**2 + F[:,1]**2)
    AO = np.sqrt(O[:,0]**2 + O[:,1]**2)

    amp_err = np.mean(AF-AO)

    return amp_err

# # calculate the phase error
# def phase_error(F, O):
#     # F: forecast [time, index]
#     # O: observation [time, index]

#     tanF = (O[:,0] * F[:,1] - O[:,1] * F[:,0]) / ()

# write a function to find the phase and amplitude of a given date
def get_phase_amp_rmm_one(date):
    # date: a string in the format of 'YYYY-MM-DD'
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
    ds = xr.open_dataset(Fnmjo).sel(time=date)
    phase = get_phase(ds['RMM'][0].values, ds['RMM'][1].values)
    amp = np.sqrt(ds['RMM'][0].values**2+ds['RMM'][1].values**2)
    
    return phase, amp

# write a function to give each time step its initial phase and amplitude
def get_phase_amp_rmm(fn):
    ds = xr.open_dataset(fn)
    phase = np.empty(ds.time.shape)
    amp = np.empty(ds.time.shape)
    for i, time in enumerate(ds.time):
        # get the initial phase and amplitude
        phase[i], amp[i] = get_phase_amp_rmm_one(time)
    
    ds['iphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'initial phase of MJO'})
    ds['iamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'initial amplitude of MJO'})
    return ds

def get_phase_amp_rmm_parallel(fn):
    ds = xr.open_dataset(fn)
    times = ds.time.values

    with ProcessPoolExecutor() as executor:
        results = executor.map(get_phase_amp_rmm_one, times)

    # unpack the results    
    phase, amp = zip(*results)
    
    ds['iphase'] = xr.DataArray(np.asarray(phase).flatten(), dims=['time'], attrs={'long_name': 'initial phase of MJO'})
    ds['iamp'] = xr.DataArray(np.asarray(amp).flatten(), dims=['time'], attrs={'long_name': 'initial amplitude of MJO'})
    
    return ds

# write a function to find the phase and amplitude of a given date
def get_phase_amp_romi_one(date):
    # date: a string in the format of 'YYYY-MM-DD'
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'
    ds = xr.open_dataset(Fnmjo).sel(time=date)
    phase = get_phase(ds['ROMI'][1].values, -ds['ROMI'][0].values)
    amp = np.sqrt(ds['ROMI'][0].values**2+ds['ROMI'][1].values**2)
    
    return phase, amp

# write a function to give each time step its initial phase and amplitude
def get_phase_amp_romi(fn):
    ds = xr.open_dataset(fn)
    phase = np.empty(ds.time.shape)
    amp = np.empty(ds.time.shape)
    for i, time in enumerate(ds.time):
        # get the initial phase and amplitude
        phase[i], amp[i] = get_phase_amp_romi_one(time)
    
    ds['iphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'initial phase of MJO'})
    ds['iamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'initial amplitude of MJO'})
    return ds

def get_phase_amp_romi_parallel(fn):
    ds = xr.open_dataset(fn)
    times = ds.time.values

    with ProcessPoolExecutor() as executor:
        results = executor.map(get_phase_amp_romi_one, times)

    # unpack the results    
    phase, amp = zip(*results)
    
    ds['iphase'] = xr.DataArray(np.asarray(phase).flatten(), dims=['time'], attrs={'long_name': 'initial phase of MJO'})
    ds['iamp'] = xr.DataArray(np.asarray(amp).flatten(), dims=['time'], attrs={'long_name': 'initial amplitude of MJO'})
    
    return ds

def get_phase_amp(mjo_ind, datasta, dataend, winter=False): # get initial phase and amplitude
    if mjo_ind == 'RMM':
        Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
    elif mjo_ind == 'ROMI':
        Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'
    
    ds0 = xr.open_dataset(Fnmjo).sel(time=slice(datasta, dataend))

    if winter:
        ds = ds0.where(ds0.time.dt.month.isin([12,1,2,3]), drop=True)

        phase = vectorized_get_phase(ds[:,0].values, ds[:,1].values)
        amp = np.sqrt(ds[:,0].values**2+ds[:,1].values**2)
        
    else:
        ds = ds0
        # phase = vectorized_get_phase(ds[:,0].values, ds[:,1].values)
        # amp = np.sqrt(ds[:,0].values**2+ds[:,1].values**2)
        if mjo_ind == 'RMM':
            phase = vectorized_get_phase(ds['RMM'][:,0].values, ds['RMM'][:,1].values)
        elif mjo_ind == 'ROMI':
            phase = vectorized_get_phase(ds.ROMI[:,1].values, -ds['ROMI'][:,0].values)

        amp = np.sqrt(ds[mjo_ind][:,0].values**2+ds[mjo_ind][:,1].values**2)

    return phase, amp

def get_19var_skill_one(mjo_ind, lead=0, exp_num='', rule='Iamp>1.0', vn='19maps', dataflg='', return_ds=False, month_list=None):
    # mjo_ind: the index of MJO
    # lead: the lead time
    fn = get_19varfiles_one(mjo_ind=mjo_ind, lead=lead, exp_num=exp_num, vn=vn, dataflg=dataflg)

    ds = xr.open_dataset(fn)
    datesta = ds.time[0].values
    dateend = ds.time[-1].values
    phase, amp = get_phase_amp(mjo_ind, datesta, dateend)
    ds['iphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'initial phase of MJO'})
    ds['iamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'initial amplitude of MJO'})
    # target phase and amplitude
    phase = vectorized_get_phase(ds[mjo_ind+'t'][:,0].values, ds[mjo_ind+'t'][:,1].values)
    amp = np.sqrt(ds[mjo_ind+'t'][:,0].values**2+ds[mjo_ind+'t'][:,1].values**2)
    ds['tphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'target phase of MJO'})
    ds['tamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'target amplitude of MJO'})
    
    if rule == 'Iamp>1.0':
        ds_sel = ds.where(ds.iamp>1.0, drop=True)
    elif rule == 'Tamp>1.0':
        ds_sel = ds.where(ds.tamp>1.0, drop=True)
    elif rule == 'DJFM':
        ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
    elif rule == 'DJFM+Iamp>1.0':
        ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
    elif rule == 'NDJF+Iamp>1.0':
        ds_sel = ds.where(ds.time.dt.month.isin([11,12,1,2]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
    elif rule == '1-1.5':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<=1.5, drop=True)
    elif rule == '1.5-2':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.5, drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<=2.0, drop=True)
    elif rule == '2-4':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>2.0, drop=True)
    elif rule == '0-1':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<1.0, drop=True)
    elif rule == 'None':
        ds_sel = ds
    elif rule == 'month+Iamp>1.0':
        ds_sel = ds.where(ds.time.dt.month.isin(month_list), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)

    bcc = bulk_bcc(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
    rmse = bulk_rmse(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])

    if return_ds:
        return bcc, rmse, ds_sel
    else:
        return bcc, rmse

def compute_get_19var_skill_one(mjo_ind, lead=0, exp_num='', rule='Iamp>1.0', vn='19maps', dataflg='', return_ds=False, month_list=None): 
    if return_ds:
        bcc, rmse, ds_sel = get_19var_skill_one(mjo_ind, lead=lead, exp_num=exp_num, rule=rule, vn=vn, dataflg=dataflg, return_ds=return_ds, month_list=month_list)
        return (lead, exp_num), {'bcc':bcc, 'rmse':rmse, 'ds_sel':ds_sel}
    else:
        bcc, rmse = get_19var_skill_one(mjo_ind, lead=lead, exp_num=exp_num, rule=rule, vn=vn, dataflg=dataflg, return_ds=return_ds, month_list=month_list)
        return (lead, exp_num), {'bcc':bcc, 'rmse':rmse}

def get_19var_skill_parallel(mjo_ind, lead_list, exp_num_list=[''], rule='Iamp>1.0', vn='19maps', dataflg='', return_ds=False, month_list=None):
    # mjo_ind: the index of MJO
    # lead_list: the list of lead time
    bcc_list = {}
    rmse_list = {}
    if return_ds:
        ds_list = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_get_19var_skill_one, mjo_ind, lead=lead, exp_num=exp_num, rule=rule, vn=vn, dataflg=dataflg, return_ds=return_ds, month_list=month_list) 
                   for lead in lead_list for exp_num in exp_num_list]
        
        for future in concurrent.futures.as_completed(futures):
            (lead, exp_num), result = future.result()
            bcc_list[(lead, exp_num)] = result['bcc']
            rmse_list[(lead, exp_num)] = result['rmse']
            if return_ds:
                ds_list[(lead, exp_num)] = result['ds_sel']

    if return_ds:
        return bcc_list, rmse_list, ds_list
    else:
        return bcc_list, rmse_list

def select_19var_dataset_one(mjo_ind, lead=0, exp_num='', rule='Iamp>1.0', vn='19maps', dataflg=''):
    # mjo_ind: the index of MJO
    # lead: the lead time
    fn = get_19varfiles_one(mjo_ind=mjo_ind, lead=lead, exp_num=exp_num, vn=vn, dataflg=dataflg)

    ds = xr.open_dataset(fn)
    datesta = ds.time[0].values
    dateend = ds.time[-1].values
    phase, amp = get_phase_amp(mjo_ind, datesta, dateend)
    ds['iphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'initial phase of MJO'})
    ds['iamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'initial amplitude of MJO'})
    # target phase and amplitude
    phase = vectorized_get_phase(ds[mjo_ind+'t'][:,0].values, ds[mjo_ind+'t'][:,1].values)
    amp = np.sqrt(ds[mjo_ind+'t'][:,0].values**2+ds[mjo_ind+'t'][:,1].values**2)
    ds['tphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'target phase of MJO'})
    ds['tamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'target amplitude of MJO'})

    if rule == 'Iamp>1.0':
        ds_sel = ds.where(ds.iamp>1.0, drop=True)
    elif rule == 'Tamp>1.0':
        ds_sel = ds.where(ds.tamp>1.0, drop=True)
    elif rule == 'DJFM':
        ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
    elif rule == 'DJFM+Iamp>1.0':
        ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
    elif rule == '1-1.5':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<=1.5, drop=True)
    elif rule == '1.5-2':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.5, drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<=2.0, drop=True)
    elif rule == '2-4':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>2.0, drop=True)
    elif rule == '0-1':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<1.0, drop=True)
    elif rule == 'None':
        ds_sel = ds

    return ds_sel


def compute_select_19var_dataset_one(mjo_ind, lead=0, exp_num='', rule='Iamp>1.0', vn='19maps', dataflg=''): 
    ds_sel = select_19var_dataset_one(mjo_ind, lead=lead, exp_num=exp_num, rule=rule, vn=vn, dataflg=dataflg)
    return (lead, exp_num), {'ds_sel':ds_sel}

def select_19var_dataset(mjo_ind, lead_list, exp_num_list=[''], rule='Iamp>1.0', vn='19maps', dataflg=''):
    # mjo_ind: the index of MJO
    # lead_list: the list of lead time
    ds_list = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_select_19var_dataset_one, mjo_ind, lead=lead, exp_num=exp_num, rule=rule, vn=vn, dataflg=dataflg) 
                   for lead in lead_list for exp_num in exp_num_list]
        
        for future in concurrent.futures.as_completed(futures):
            (lead, exp_num), result = future.result()
            ds_list[(lead, exp_num)] = result['ds_sel']

    return ds_list


def get_19var_skill(mjo_ind, lead_list, exp_num='', rule='Iamp>1.0', vn='19maps'):
    # mjo_ind: the index of MJO
    # lead_list: the list of lead time
    fn_list = get_19varfiles(mjo_ind, lead_list, exp_num, vn=vn)

    # initialize the lists
    bcc_list = []
    rmse_list = []
    bcc_dis = []
    rmse_dis = []
    
    for fn in fn_list:
        # get the initial phase and amplitude
        if mjo_ind == 'RMM':
            ds = get_phase_amp_rmm_parallel(fn)
        elif mjo_ind == 'ROMI':
            ds = get_phase_amp_romi_parallel(fn)
        else:
            print('Wrong MJO index!')
            return
        
        if rule == 'Iamp>1.0':
            ds_sel = ds.where(ds.iamp>1.0, drop=True)
        elif rule == 'DJFM':
            ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
        elif rule == 'DJFM+Iamp>1.0':
            ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
            ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
        elif rule == '1-1.5':
            ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
            ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
            ds_sel = ds_sel.where(ds_sel.iamp<=1.5, drop=True)
        elif rule == '1.5-2':
            ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
            ds_sel = ds_sel.where(ds_sel.iamp>1.5, drop=True)
            ds_sel = ds_sel.where(ds_sel.iamp<=2.0, drop=True)
        elif rule == '2-4':
            ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
            ds_sel = ds_sel.where(ds_sel.iamp>2.0, drop=True)
        elif rule == '0-1':
            ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
            ds_sel = ds_sel.where(ds_sel.iamp<1.0, drop=True)
        elif rule == 'None':
            ds_sel = ds

        bcc = bulk_bcc(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
        bcc_list.append(bcc.values)

        rmse = bulk_rmse(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
        rmse_list.append(rmse.values)

        del bcc 
        del rmse

        bcc = bulk_bcc_mcdp(ds_sel[mjo_ind+'p_dis'], ds_sel[mjo_ind+'t'])
        bcc_dis.append(bcc.values)
        rmse = bulk_rmse_mcdp(ds_sel[mjo_ind+'p_dis'], ds_sel[mjo_ind+'t'])
        rmse_dis.append(rmse.values)

        # if rule == 'Iamp>1.0':
        #     ds_sel = ds.where(ds.iamp>1.0, drop=True)
        #     bcc = bulk_bcc(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
        #     bcc_list.append(bcc.values)

        #     rmse = bulk_rmse(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
        #     rmse_list.append(rmse.values)

        #     del bcc 
        #     del rmse

        #     bcc = bulk_bcc_mcdp(ds_sel[mjo_ind+'p_dis'], ds_sel[mjo_ind+'t'])
        #     bcc_dis.append(bcc.values)
        #     rmse = bulk_rmse_mcdp(ds_sel[mjo_ind+'p_dis'], ds_sel[mjo_ind+'t'])
        #     rmse_dis.append(rmse.values)
        
    return bcc_list, rmse_list, bcc_dis, rmse_dis

def get_nvar_skill(mjo_ind, lead_list,exp_num_list=[''], rule='Iamp>1.0', vn='19maps'):
    # mjo_ind: the index of MJO
    # lead_list: the list of lead time
    # return the averaged skill of the ensemble members and the uncertainty
    bcc_all = np.empty((len(exp_num_list), len(lead_list)))
    rmse_all = np.empty((len(exp_num_list), len(lead_list)))

    fn_ind = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/'+mjo_ind+'_ERA5_daily_2015to2023.nc'
    ds_ind = xr.open_dataset(fn_ind)  # this is the file to get the initial phase and amplitude

    for i, exp_num in enumerate(exp_num_list):  # for each ensemble member
        fn_list = get_19varfiles(mjo_ind, lead_list, exp_num=exp_num, vn=vn)

        # print(vn + exp_num)
        # initialize the lists
        bcc_list = []
        rmse_list = []
    
        for fn in fn_list:  # for each lead time
            # get the initial phase and amplitude
            ds = xr.open_dataset(fn)
            ds_ind_sel = ds_ind.sel(time=slice(ds.time[0], ds.time[-1]))
            ds['iphase'] = ds_ind_sel['iphase']
            ds['iamp'] = ds_ind_sel['iamp']

            if rule == 'Iamp>1.0':
                ds_sel = ds.where(ds.iamp>1.0, drop=True)
            elif rule == 'DJFM':
                ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
            elif rule == 'DJFM+Iamp>1.0':
                ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
                ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
            elif rule == 'None':
                ds_sel = ds
            
            bcc = bulk_bcc(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
            bcc_list.append(bcc.values)

            rmse = bulk_rmse(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
            rmse_list.append(rmse.values)

            del bcc 
            del rmse

        bcc_all[i,:] = np.asarray(bcc_list)
        rmse_all[i,:] = np.asarray(rmse_list)
    
    key_name = vn

    return {key_name:bcc_all}, {key_name:rmse_all}

def get_19var_skill_phase(mjo_ind, lead_list, rule='Iamp>1.0'):
    # mjo_ind: the index of MJO
    # lead_list: the list of lead time
    fn_list = get_19varfiles(mjo_ind, lead_list)

    # initialize the lists
    bcc_list = np.empty((8, len(lead_list)))
    rmse_list = np.empty((8, len(lead_list)))
    
    for i, fn in enumerate(fn_list):
        # get the initial phase and amplitude
        if mjo_ind == 'RMM':
            ds = get_phase_amp_rmm_parallel(fn)
        elif mjo_ind == 'ROMI':
            ds = get_phase_amp_romi_parallel(fn)
        else:
            print('Wrong MJO index!')
            return
        
        if rule == 'Iamp>1.0':
            ds_sel = ds.where(ds.iamp>1.0, drop=True)
        elif rule == 'DJFM':
            ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
        elif rule == 'DJFM+Iamp>1.0':
            ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
            ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
        elif rule == 'None':
            ds_sel = ds

        grouped = ds_sel.groupby('iphase')

        for phase, group in grouped:
            phase = int(phase)
            # print(phase)
            bcc = bulk_bcc(group[mjo_ind+'p'].values, group[mjo_ind+'t'].values)

            bcc_list[phase-1, i] = bcc

            rmse = bulk_rmse(group[mjo_ind+'p'].values, group[mjo_ind+'t'].values)
            rmse_list[phase-1, i] = rmse

    return bcc_list, rmse_list

def get_1var_skill(vn, mjo_ind, lead_list,exp_num='', rule='Iamp>1.0', m=1, mflg='off', wnx=1, wnxflg='off'):
    # mjo_ind: the index of MJO
    # lead_list: the list of lead time
    fn_list = get_1varfiles(vn, mjo_ind, lead_list, exp_num=exp_num, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg)

    # initialize the lists
    bcc_list = []
    rmse_list = []
    bcc_dis = []
    rmse_dis = []
    
    for fn in fn_list:
        # get the initial phase and amplitude
        if mjo_ind == 'RMM':
            ds = get_phase_amp_rmm_parallel(fn)
        elif mjo_ind == 'ROMI':
            ds = get_phase_amp_romi_parallel(fn)
        else:
            print('Wrong MJO index!')
            return

        if rule == 'Iamp>1.0':
            ds_sel = ds.where(ds.iamp>1.0, drop=True)
        elif rule == 'DJFM':
            ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
        elif rule == 'None':
            ds_sel = ds
        
        bcc = bulk_bcc(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
        bcc_list.append(bcc.values)

        rmse = bulk_rmse(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
        rmse_list.append(rmse.values)

        del bcc 
        del rmse

        bcc = bulk_bcc_mcdp(ds_sel[mjo_ind+'p_dis'], ds_sel[mjo_ind+'t'])
        bcc_dis.append(bcc.values)
        rmse = bulk_rmse_mcdp(ds_sel[mjo_ind+'p_dis'], ds_sel[mjo_ind+'t'])
        rmse_dis.append(rmse.values)
        
    return bcc_list, rmse_list, bcc_dis, rmse_dis

def get_1var_skill_ens_one(vn, mjo_ind, lead_list,exp_num_list=[''], rule='Iamp>1.0', m=1, mflg='off', wnx=1, wnxflg='off', dataflg=''):
    # mjo_ind: the index of MJO
    # lead_list: the list of lead time
    # return the averaged skill of the ensemble members and the uncertainty
    bcc_all = np.empty((len(exp_num_list), len(lead_list)))
    rmse_all = np.empty((len(exp_num_list), len(lead_list)))

    fn_ind = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/'+mjo_ind+'_ERA5_daily_2015to2023.nc'
    ds_ind = xr.open_dataset(fn_ind)  # this is the file to get the initial phase and amplitude

    for i, exp_num in enumerate(exp_num_list):  # for each ensemble member
        fn_list = get_1varfiles(vn, mjo_ind, lead_list, exp_num=exp_num, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, dataflg=dataflg)

        # print(vn + exp_num)
        # initialize the lists
        bcc_list = []
        rmse_list = []
    
        for fn in fn_list:  # for each lead time
            # get the initial phase and amplitude
            ds = xr.open_dataset(fn)
            ds_ind_sel = ds_ind.sel(time=slice(ds.time[0], ds.time[-1]))
            ds['iphase'] = ds_ind_sel['iphase']
            ds['iamp'] = ds_ind_sel['iamp']

            if rule == 'Iamp>1.0':
                ds_sel = ds.where(ds.iamp>1.0, drop=True)
            elif rule == 'DJFM':
                ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
            elif rule == 'None':
                ds_sel = ds
            
            bcc = bulk_bcc(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
            bcc_list.append(bcc.values)

            rmse = bulk_rmse(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
            rmse_list.append(rmse.values)

            del bcc 
            del rmse

        bcc_all[i,:] = np.asarray(bcc_list)
        rmse_all[i,:] = np.asarray(rmse_list)
    
    key_name = vn+str(m)+mflg+str(wnx)+wnxflg

    return {key_name:bcc_all}, {key_name:rmse_all}

def get_1var_skill_one(mjo_ind, lead=0, exp_num='', m=1, mflg='off', wnx=1, wnxflg='off', rule='Iamp>1.0', 
                       vn='olr', dataflg='', winter=False, outputflg='', lat_range=20, fn=None, zero_channel=True, 
                       ptb_channel=False, zero_channel_flg='_zero_all10pls', month_list=[12,1,2,3]):
    # mjo_ind: the index of MJO
    # lead: the lead time
    # fn: the number of channel to be zeroed. 

    if fn is None:
        if zero_channel:
            fn = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/zero_channel_lastconv/'+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_c51_mem1'+zero_channel_flg+'.nc'
        else:
            fn = get_1varfiles_one(mjo_ind=mjo_ind, lead=lead, exp_num=exp_num, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, vn=vn, dataflg=dataflg, winter=winter, outputflg=outputflg, lat_range=lat_range)
    else:
        if zero_channel:
            # new prediction with one channel zeroed out
            fn = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/zero_channel_lastconv/'+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_c51_mem1_zero_'+str(fn)+'.nc'
            # reference prediction
        elif ptb_channel:
            dataflg = '_ptb_' + str(fn) + '_exp' + str(exp_num)
            fn = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/ptb_channel_lastconv/'+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_c51_mem1'+dataflg+'.nc'
            
    ds = xr.open_dataset(fn)
    datesta = ds.time[0].values
    dateend = ds.time[-1].values
    phase, amp = get_phase_amp(mjo_ind, datesta, dateend, winter=winter)
    ds['iphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'initial phase of MJO'})
    ds['iamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'initial amplitude of MJO'})
    # target phase and amplitude
    phase = vectorized_get_phase(ds[mjo_ind+'t'][:,0].values, ds[mjo_ind+'t'][:,1].values)
    amp = np.sqrt(ds[mjo_ind+'t'][:,0].values**2+ds[mjo_ind+'t'][:,1].values**2)
    ds['tphase'] = xr.DataArray(phase, dims=['time'], attrs={'long_name': 'target phase of MJO'})
    ds['tamp'] = xr.DataArray(amp, dims=['time'], attrs={'long_name': 'target amplitude of MJO'})
    
    if rule == 'Iamp>1.0':
        ds_sel = ds.where(ds.iamp>1.0, drop=True)
    elif rule == 'Tamp>1.0':
        ds_sel = ds.where(ds.tamp>1.0, drop=True)
    elif rule == 'DJFM':
        ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
    elif rule == 'DJFM+Iamp>1.0':
        ds_sel = ds.where(ds.time.dt.month.isin([12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
    elif rule == 'month+Iamp>1.0':
        ds_sel = ds.where(ds.time.dt.month.isin(month_list), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
    elif rule == '1-1.5':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.0, drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<=1.5, drop=True)
    elif rule == '1.5-2':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>1.5, drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<=2.0, drop=True)
    elif rule == '2-4':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp>2.0, drop=True)
    elif rule == '0-1':
        ds_sel = ds.where(ds.time.dt.month.isin([10,11,12,1,2,3]), drop=True)
        ds_sel = ds_sel.where(ds_sel.iamp<1.0, drop=True)
    elif rule == 'None':
        ds_sel = ds

    bcc = bulk_bcc(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])
    rmse = bulk_rmse(ds_sel[mjo_ind+'p'], ds_sel[mjo_ind+'t'])

    return bcc, rmse

def compute_get_1var_skill_one(mjo_ind, lead=0, exp_num='', m=1, mflg='off', wnx=1, wnxflg='off', rule='Iamp>1.0', 
                               vn='olr', dataflg='', winter=False, outputflg='', lat_range=20, fn=None, zero_channel=True, 
                               ptb_channel=False, zero_channel_flg='_zero_all10pls', month_list=[12,1,2,3]):    
    bcc, rmse = get_1var_skill_one(mjo_ind, lead=lead, exp_num=exp_num, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, rule=rule, 
                                   vn=vn, dataflg=dataflg, winter=winter, outputflg=outputflg, lat_range=lat_range, fn=fn, 
                                   zero_channel=zero_channel, ptb_channel=ptb_channel, zero_channel_flg=zero_channel_flg, month_list=month_list)
    if fn is None:
        return (lead, exp_num), {'bcc':bcc, 'rmse':rmse}
    else:
        # fn = int(fn)
        return (lead, exp_num, fn), {'bcc':bcc, 'rmse':rmse}

def get_1var_skill_parallel(mjo_ind, lead_list, exp_num_list=[''], m=1, mflg='off', wnx=1, wnxflg='off', rule='Iamp>1.0', 
                            vn='olr', dataflg='', winter=False, outputflg='', lat_range=20, fn_list=None, zero_channel=True, 
                            ptb_channel=False, zero_channel_flg='_zero_all10pls', month_list=[12,1,2,3]):
    # mjo_ind: the index of MJO
    # lead_list: the list of lead time
    # fn_list: the numbers of channels to be zeroed

    bcc_list = {}
    rmse_list = {}

    with ProcessPoolExecutor() as executor:
        if fn_list is None:
            futures = [executor.submit(compute_get_1var_skill_one, mjo_ind, lead=lead, exp_num=exp_num, m=m, mflg=mflg, wnx=wnx, 
                                       wnxflg=wnxflg, rule=rule, vn=vn, dataflg=dataflg, winter=winter, outputflg=outputflg, 
                                       lat_range=lat_range, zero_channel=zero_channel, ptb_channel=ptb_channel, 
                                       zero_channel_flg=zero_channel_flg, month_list=month_list) 
                    for lead in lead_list for exp_num in exp_num_list]
            
            for future in concurrent.futures.as_completed(futures):
                (lead, exp_num), result = future.result()
                bcc_list[(lead, exp_num)] = result['bcc']
                rmse_list[(lead, exp_num)] = result['rmse']
                
        else:
            futures = [executor.submit(compute_get_1var_skill_one, mjo_ind, lead=lead, exp_num=exp_num, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, 
                                       rule=rule, vn=vn, dataflg=dataflg, winter=winter, outputflg=outputflg, lat_range=lat_range, fn=fn, 
                                       zero_channel=zero_channel, ptb_channel=ptb_channel, month_list=month_list) 
                    for lead in lead_list for exp_num in exp_num_list for fn in fn_list]
            
            for future in concurrent.futures.as_completed(futures):
                (lead, exp_num, fn), result = future.result()
                bcc_list[(lead, exp_num, fn)] = result['bcc']
                rmse_list[(lead, exp_num, fn)] = result['rmse']
            
    return bcc_list, rmse_list

def plot_uncertainty_pct(ax, x, y, ydis, thred, xlab, ylab, label=None, title=None, line_c='tab:blue'):
    p1 = 25
    p2 = 75
    plt.rcParams.update({'font.size': 22})
    ax.plot(x, y,'-', linewidth=2.5, label=label, color=line_c)
    # ax.fill_between(x, np.percentile(ydis, p1, axis=1), np.percentile(ydis, p2, axis=1), alpha=0.5, color=line_c)
    # ax.plot(x, np.ones(len(x))*thred, 'k--', linewidth=2)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_ylim([0.4,1.0])
    ax.set_xlim([1,40])
    # ax.set_xtick(np.arange(0, 31, 5))
    if title is not None:
        ax.set_title(title, pad=20)
    ax.grid(visible=True)

# def plot_uncertainty_all(ax, x, y, ydis, thred, xlab, ylab, label=None, title=None, line_c='tab:blue'):
#     plt.rcParams.update({'font.size': 22})
#     ax.plot(x, y,'-', linewidth=2.5, label=label, color=line_c)
#     ax.fill_between(x, np.min(ydis, axis=1), np.max(ydis, axis=1), alpha=0.5, color=line_c)
#     # ax.plot(x, np.ones(len(x))*thred, 'k--', linewidth=2)
#     ax.set_xlabel(xlab)
#     ax.set_ylabel(ylab)
#     ax.set_ylim([0.5,1.0])
#     ax.set_xlim([0,40])
#     # ax.set_xtick(np.arange(0, 31, 5))
#     if title is not None:
#         ax.set_title(title, pad=20)
#     ax.grid(visible=True)


def plot_uncertainty_all(ax, x, y, ydis, thred, xlab=None, ylab=None, label=None, title=None, line_c='tab:blue',alpha=1.0, xlim=[0,30], ylim=[0.3,1.0], ftsize = 26, gap=0.1, style='-', alpha_fill=0.5):
    plt.rcParams.update({'font.size': ftsize})
    ax.plot(x, y, linestyle=style, linewidth=2.5, label=label, color=line_c, alpha=alpha)
    ax.fill_between(x, np.min(ydis, axis=1), np.max(ydis, axis=1), alpha=alpha_fill, color=line_c)
    # ax.plot(x, np.ones(len(x))*thred, 'k--', linewidth=2)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.arange(ylim[0], ylim[1]+0.1, gap))
    # ax.set_xtick(np.arange(0, 31, 5))
    if title is not None:
        ax.set_title(title, pad=20)
    ax.grid(visible=True)

# plot the background of the MJO phase diagram
def mjo_phase_background(ax, opt=None):
    if opt is not None and "axisExtent" in opt and opt["axisExtent"] > 0:
        axisExtent = opt["axisExtent"]
    else:
        axisExtent = 4

    nPhase = 8

    res = {
        # "gsnDraw": False,
        # "gsnFrame": False,
        "vpXF": 0.1,
        "vpYF": 0.8,
        "trYMinF": -axisExtent,
        "trYMaxF": axisExtent,
        "trXMinF": -axisExtent,
        "trXMaxF": axisExtent + 0.05,
        "vpWidthF": 0.45,
        "vpHeightF": 0.45,
        "tmXBFormat": "f",
        "tmYLFormat": "f",
        "tmXBLabelDeltaF": -0.75,
        "tmYLLabelDeltaF": -0.75,
        "tiXAxisFontHeightF": 0.0167,
        "tiYAxisFontHeightF": 0.0167,
        "tiDeltaF": 1.25,
        "xyLineThicknessF": 1
    }

    rad = 4. * np.arctan(1.0) / 180.
    if opt is not None and "radius" in opt and opt["radius"] > 0:
        radius = opt["radius"]
    else:
        radius = 1.0

    nCirc = 361
    theta = np.linspace(0, 360, nCirc) * rad
    xCirc = radius * np.cos(theta)
    yCirc = radius * np.sin(theta)

    if opt is not None and "tiMainString" in opt:
        res["tiMainString"] = opt["tiMainString"]


    ax.plot(xCirc, yCirc, color="black", linewidth=1.0)

    txres = {
        "fontsize": 8,
        "rotation": 90,
        "va": "center",
        "ha": "center"
    }
    # txid = ax.text(0, 0, "Phase 5 (Maritime) Phase 4", **txres)

    amres = {
        "xytext": (0.5, 0.5),
        "textcoords": "axes fraction",
        "ha": "center",
        "va": "center"
    }
    # ann3 = ax.annotate("Phase 7 (Western Pacific) Phase 6")

    plres = {
        "color": "black",
        "linewidth": 1.0,
        "linestyle": "-",
        "dashes": [8, 4]
    }
    if opt is not None and "gsLineDashPattern" in opt:
        plres["dashes"] = opt["gsLineDashPattern"]

    c45 = radius * np.cos(45 * rad)
    E = axisExtent
    R = radius

    phaLine = np.zeros((nPhase, 4))
    phaLine[0, :] = [R, E, 0, 0]
    phaLine[1, :] = [c45, E, c45, E]
    phaLine[2, :] = [0, 0, R, E]
    phaLine[3, :] = [-c45, -E, c45, E]
    phaLine[4, :] = [-R, -E, 0, 0]
    phaLine[5, :] = [-c45, -E, -c45, -E]
    phaLine[6, :] = [0, 0, -R, -E]
    phaLine[7, :] = [c45, E, -c45, -E]

    for i in range(nPhase):
        ax.plot([phaLine[i, 0], phaLine[i, 1]], [phaLine[i, 2], phaLine[i, 3]], **plres)

    plt.show()
