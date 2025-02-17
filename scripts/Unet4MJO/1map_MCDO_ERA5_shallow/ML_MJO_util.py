import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os 
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

def get_1varfiles_one(mjo_ind, lead=0, exp_num='', m=1, mflg='off', wnx=1, wnxflg='off',vn='olr', dataflg='', winter=False, output_dir='/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_shallow/output', lat_range=20):

    # lat_range = 20
    nmem = 1

    output_path = output_dir+str(exp_num)+'/'


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
    
    ds0 = xr.open_dataarray(Fnmjo).sel(time=slice(datasta, dataend))

    if winter:
        ds = ds0.where(ds0.time.dt.month.isin([12,1,2,3]), drop=True)

        phase = vectorized_get_phase(ds[:,0].values, ds[:,1].values)
        amp = np.sqrt(ds[:,0].values**2+ds[:,1].values**2)
        
    else:
        ds = ds0
        phase = vectorized_get_phase(ds[:,0].values, ds[:,1].values)
        amp = np.sqrt(ds[:,0].values**2+ds[:,1].values**2)
        # if mjo_ind == 'RMM':
        #     phase = vectorized_get_phase(ds['RMM'][:,0].values, ds['RMM'][:,1].values)
        # elif mjo_ind == 'ROMI':
        #     phase = vectorized_get_phase(ds['ROMI'][:,1].values, -ds['ROMI'][:,0].values)

        # amp = np.sqrt(ds[mjo_ind][:,0].values**2+ds[mjo_ind][:,1].values**2)

    return phase, amp

def get_1var_skill_one(mjo_ind, lead=0, exp_num='', m=1, mflg='off', wnx=1, wnxflg='off', rule='Iamp>1.0', vn='olr', dataflg='', winter=False, lat_range=20, output_dir='/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_shallow/output'):
    # mjo_ind: the index of MJO
    # lead: the lead time
    # fn: the number of channel to be zeroed. 

    fn = get_1varfiles_one(mjo_ind=mjo_ind, lead=lead, exp_num=exp_num, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, vn=vn, dataflg=dataflg, winter=winter, lat_range=lat_range, output_dir=output_dir)

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

def compute_get_1var_skill_one(mjo_ind, lead=0, exp_num='', m=1, mflg='off', wnx=1, wnxflg='off', rule='Iamp>1.0', vn='olr', dataflg='', winter=False, lat_range=20, output_dir='/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_shallow/output'):
    bcc, rmse = get_1var_skill_one(mjo_ind, lead=lead, exp_num=exp_num, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, rule=rule, vn=vn, dataflg=dataflg, winter=winter, lat_range=lat_range, output_dir=output_dir)
    return (lead, exp_num), {'bcc':bcc, 'rmse':rmse}

def get_1var_skill_parallel(mjo_ind, lead_list, exp_num_list=[''], m=1, mflg='off', wnx=1, wnxflg='off', rule='Iamp>1.0', vn='olr', dataflg='', winter=False, lat_range=20, output_dir='/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_shallow/output'):

    bcc_list = {}
    rmse_list = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_get_1var_skill_one, mjo_ind, lead=lead, exp_num=exp_num, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, rule=rule, vn=vn, dataflg=dataflg, winter=winter, lat_range=lat_range, output_dir=output_dir) 
                for lead in lead_list for exp_num in exp_num_list]
        
        for future in concurrent.futures.as_completed(futures):
            (lead, exp_num), result = future.result()
            bcc_list[(lead, exp_num)] = result['bcc']
            rmse_list[(lead, exp_num)] = result['rmse']
                     
    return bcc_list, rmse_list

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
