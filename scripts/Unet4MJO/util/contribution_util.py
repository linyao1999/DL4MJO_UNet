import numpy as np
import pandas as pd  
import xarray as xr 
import matplotlib.pyplot as plt 
import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
from datetime import date 
import pickle

import dask
from scipy import special
import math 
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from ML_MJO_util import get_1varfiles_one, get_phase_amp 


def get_hid26_std(channel, vn='olr',lat_lim=20,mjo_ind='RMM',lead=15,m=1,mflg='off',wnx=1,wnxflg='off',zmode=1,nmem=1, after_relu=False):
    flag = vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(lead)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)

    if after_relu:
        with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/std_hid26relu'+flag+'.pickle', 'rb') as file2:
            std_hid26 = pickle.load(file2)
    else:
        with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/std_hid26'+flag+'.pickle', 'rb') as file2:
            std_hid26 = pickle.load(file2)
        
    return np.squeeze(std_hid26[channel, :, :]) # (lat, lon)

def get_UNet(nmaps=1, nmem=1, lat_lim=20, zero_channel=None, perturb=None, after_relu=False, lead=15, vn='olr', mjo_ind='RMM'):
    # Default values are for OLR-20deg-1timestep model 

    # model's hyperparameters
    nhidden1=500
    nhidden2=200
    nhidden3=50

    dimx = int(1 + 2 * int(lat_lim / 2))
    dimy = 180

    num_filters_enc = 64
    num_filters_dec1 = 128
    num_filters_dec2 = 192

    featureDim=num_filters_dec2*dimx*dimy

    class CNN(nn.Module):
        def __init__(self,imgChannels=nmaps*nmem, out_channels=2):
            super().__init__()
            self.input_layer = (nn.Conv2d(imgChannels, num_filters_enc, kernel_size=5, stride=1, padding='same'))
            self.hidden1 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=5, stride=1, padding='same' ))
            self.hidden2 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=5, stride=1, padding='same' ))
            self.hidden3 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=5, stride=1, padding='same' ))
            self.hidden4 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=5, stride=1, padding='same' ))


            self.hidden5 = (nn.Conv2d(num_filters_dec1, num_filters_dec1, kernel_size=5, stride=1, padding='same' ))
            self.hidden6 = (nn.Conv2d(num_filters_dec2, num_filters_dec2, kernel_size=5, stride=1, padding='same' ))

            self.FC1 = nn.Linear(featureDim,nhidden1)
            self.FC2 = nn.Linear(nhidden1,nhidden2)
            self.FC3 = nn.Linear(nhidden2,nhidden3)
            self.FC4 = nn.Linear(nhidden3,out_channels)

            self.dropoutconv1 = nn.Dropout2d(p=0.1)
            self.dropoutconv2 = nn.Dropout2d(p=0.1)
            self.dropoutconv3 = nn.Dropout2d(p=0.1)
            self.dropoutconv4 = nn.Dropout2d(p=0.1)
            self.dropoutconv5 = nn.Dropout2d(p=0.1)
            self.dropoutconv6 = nn.Dropout2d(p=0.1)
            self.dropoutline1 = nn.Dropout(p=0.2)
            self.dropoutline2 = nn.Dropout(p=0.2)
            self.dropoutline3 = nn.Dropout(p=0.2)
            
        def forward (self,x):

            x1 = F.relu (self.dropoutconv1(self.input_layer(x)))
            x2 = F.relu (self.dropoutconv2(self.hidden1(x1)))
            x3 = F.relu (self.dropoutconv3(self.hidden2(x2)))
            x4 = F.relu (self.dropoutconv4(self.hidden3(x3)))

            x5 = torch.cat ((F.relu(self.dropoutconv5(self.hidden4(x4))),x3), dim =1)
            x6 = torch.cat ((F.relu(self.dropoutconv6(self.hidden5(x5))),x2), dim =1)
            
            if zero_channel is not None:
                x6[:, zero_channel, :, :] = 0.0

            # add perturbation to the hidden layer for a given channel
            # the perturbation is the standard deviation of the hidden layer * a random number between -1 and 1
            if perturb is not None:
                std26 = get_hid26_std(perturb, after_relu=after_relu, lead=lead, vn=vn, mjo_ind=mjo_ind, lat_lim=lat_lim)  # (channel, lat, lon).squeeze() -> (lat, lon)
                # repeat std26 to the same shape as x6
                std26 = np.repeat(std26[np.newaxis, :, :], np.shape(x6)[0], axis=0)
                pert = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=np.shape(std26)) * std26).float().cuda()  # (batch, lat, lon)
                # # reshape perturbation to the same shape as x6
                # pert = pert.view(-1, 1, dimx, dimy)
                x6[:, perturb, :, :] = x6[:, perturb, :, :] + pert 
                

            x6 = x6.view(-1,featureDim)
            x6 = F.relu(self.FC1(x6))
            x7 = F.relu(self.FC2(self.dropoutline1(x6)))
            x8 = F.relu(self.FC3(self.dropoutline2(x7)))

            out = (self.FC4(self.dropoutline3(x8)))

            return out

    net = CNN()  

    return net.cuda()  # send the net to GPU

def projection(vn, c=51, m=1, mflg='off', wnx=10, wnxflg='all', time_range=['1979-01-01','2019-12-31'], lat_range=[20, -20], pic_save='./',dataflg='new'):
    # zmode: the vertical mode, default m = 1
    # m: wave truncation
    # wnx: zonal wave number truncation [inclusive]
    # time_range: the time range of the data used in training and validating the model [inclusive]
    # lat_range: the latitude range (y) of the data used in projection

    # read data; any file with olr[time, lat, lon]
    if dataflg=='raw':
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/daily/'+vn+'.day.1978to2023.nc'
    elif dataflg=='new':
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/fltano120/2012/'+vn+'.fltano120.1978to2023based1979to2012.nc'
    elif dataflg=='new40':
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/fltano40/'+vn+'.fltano40.1978to2023based1979to2012.nc'
    else:
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5.'+vn+'GfltG.day.1901to2020.nc'
    
    ds = xr.open_dataset(fn)

    # ds1 = ds.sel(time=slice(time_range[0], time_range[1]), lat=slice(lat_range[0], lat_range[1])).fillna(0)
    ds1 = ds.sel(time=slice(time_range[0], time_range[1])).fillna(0)

    olr = ds1[vn].values  # (time, lat, lon)
    lat = ds1['lat']
    # find the index of the latitudes within the lat_range
    lat_ind = np.where((lat>=lat_range[1]) & (lat<=lat_range[0]))[0]

    lon = ds1['lon'].values
    time = ds1['time'].values

    if mflg=='off':
        olr_re = np.copy(olr)  # no dimension reduction on the meridional direction. 
    else:  
        # # parameters
        # N = 1e-2  # buoyancy frequency (s-1)
        # H = 1.6e4  # tropopause height (m)
        beta= 2.28e-11  # variation of coriolis parameter with latitude
        # g = 9.8  # gravity acceleration 
        # theta0 = 300  # surface potential temperature (K)
        # c = N * H / np.pi / zmode # gravity wave speed
        L = np.sqrt(c / beta)  # horizontal scale (m)
    
        # define y = lat * 110 km / L
        y = lat.values * 110 * 1000 / L # dimensionless

        # define meridianol wave structures
        phi = []

        # define which equatorial wave is included in the reconstructed map
        # m is analogous to a meridional wavenumber
        # m = 0: mostly Kelvin wave
        # m = 2: mostly Rossby wave

        if mflg=='odd':
            m_list = np.arange(1,m,2)  
        elif mflg=='even':
            m_list = np.arange(0,m,2)
        elif mflg=='all':
            m_list = np.arange(m)
        elif mflg=='one':
            m_list = [m-1]
        elif mflg=='1pls':
            m_list = [0,m-1]
        elif mflg=='no1':
            m_list = np.arange(1,m)
        elif mflg=='resi':
            m_list = np.arange(m)  # this is the part to be removed from filtered map
        else:
            print('wrong m flag!')
            exit()

        for i in m_list:
            p = special.hermite(i)
            Hm = p(y)
            phim = np.exp(- y**2 / 2) * Hm / np.sqrt((2**i) * np.sqrt(np.pi) * math.factorial(i))

            phi.append(np.reshape(phim, (1, len(y), 1)))

        # projection coefficients
        olrm = []

        dy = (lat[0].values - lat[1].values) * 110 * 1000 / L 

        for i in range(len(m_list)):
            um = np.sum(olr * phi[i] * dy, axis=1, keepdims=True)  # (time, 1, lon)
            olrm.append(um)

        # reconstruction 
        olr_re = np.zeros(np.shape(olr))  # (time, lat, lon)

        for i in range(len(m_list)):
            olr_re = olr_re + olrm[i] * phi[i]
        
        if mflg=='resi':
            olr_re1 = olr - olr_re
            del olr_re
            olr_re = np.copy(olr_re1)
            del olr_re1

    if wnxflg=='off':
        olr_re_fft = np.copy(olr_re)
    else:
        # do fourier transform along each latitude at each time step
        coef_fft = np.fft.rfft(olr_re, axis=2)
        # remove waves whose zonal wave cycles are larger than wnx
        if wnxflg=='all':
            coef_fft[:,:,wnx+1:] = 0.0 
        elif wnxflg=='one':
            if wnx==0:
                coef_fft[:,:,wnx+1:] = 0.0
            else:
                coef_fft[:,:,:wnx] = 0.0
                coef_fft[:,:,wnx+1:] = 0.0 
        elif wnxflg=='no0':  # include 1, 2, ..., wnx
            coef_fft[:,:,wnx+1:] = 0.0 
            coef_fft[:,:,0] = 0.0
        elif wnxflg=='no0p7': # include 1, 2, ..., wnx, 7
            coef_fft[:,:,wnx+1:7] = 0.0 
            coef_fft[:,:,8:] = 0.0 
            coef_fft[:,:,0] = 0.0
        elif wnxflg=='resi':  # resi of 0-wnx[inclusive]
             coef_fft[:,:,:wnx+1] = 0.0
        else:
            print('wrong wnx flag!')
            exit()
        # reconstruct OLR with selected zonal waves
        olr_re_fft = np.fft.irfft(coef_fft, np.shape(olr_re)[2], axis=2)



    if pic_save != 'None':
        fig_file_path = pic_save+dataflg+vn+str(lat_range[0])+'deg_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_global.jpg'
        if not os.path.exists(fig_file_path):  # create a snapshot of the input map if the file does not exist. 
            # save the OLR maps at the first time step
            fig, ax = plt.subplots(2,1)
            fig.set_figheight(6)
            fig.set_figwidth(12)

            # fig1: reconstructed OLR after zonal fft
            im = ax[0].contourf(lon, lat.values, olr_re_fft[0,:,:])
            ax[0].set_xlabel('longitude')
            ax[0].set_ylabel('latitude')
            plt.colorbar(im, ax=ax[0])
            ax[0].set_title('filtered+Yproj_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg)

            # fig2: filtered OLR - reconstructed OLR after zonal fft
            tmp = olr[0,:,:] - olr_re_fft[0,:,:] 
            im = ax[1].contourf(lon, lat.values, tmp)
            ax[1].set_xlabel('longitude')
            ax[1].set_ylabel('latitude')
            plt.colorbar(im, ax=ax[1])
            ax[1].set_title('information missing from the filtered map')

            plt.subplots_adjust(hspace=0.4)

            fig.savefig(fig_file_path)

    return olr_re_fft[:, lat_ind, :]  # (time, lat, lon)

def load_test_data(Fnmjo,leadmjo,testystat=2015,testyend=2020,c=51,m=1,mflg='off',wnx=1,wnxflg='off',lat_lim=20,mjo_ind='RMM',vn='olr', pic_save='None',mem_list=[0,],dataflg='new'):
    # set parameters
    nmem = len(mem_list)  # memory length
    dimx = int(1 + 2 * int( int(lat_lim) / int(2)))
    dimy = 180

    # make projection and do zonal fft (time, lat, lon)
    olr_re = projection(vn=vn, c=c, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, time_range=[str(testystat)+'-01-01', str(testyend-1)+'-12-31'], lat_range=[lat_lim,-lat_lim],pic_save=pic_save, dataflg=dataflg)

    ndays = int( ( np.datetime64(str(testyend)+'-01-01') - np.datetime64(str(testystat)+'-01-01') ) / np.timedelta64(1,'D') ) 

    psi_test_input = np.zeros((ndays-mem_list[-1],nmem,dimx,dimy))

    for i in range(ndays-mem_list[-1]):
        psi_test_input[i, :, :, :] = olr_re[i:i+nmem, :, :]


    print('combined input shape is: ' + str(np.shape(psi_test_input[i, :, :, :])))

    # read the MJO index
    FFmjo = xr.open_dataset(Fnmjo)
    FFmjo = FFmjo.sel(time=slice(str(testystat)+'-01-01', str(testyend)+'-03-31'))
    # FFmjo.fillna(0)
    pc = np.asarray(FFmjo[mjo_ind])

    Nlat=dimx
    Nlon=dimy

    psi_test_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays-mem_list[-1],:]

    psi_test_input_Tr=np.zeros([np.size(psi_test_input,0),nmem,Nlat,Nlon])   # vn input maps
    psi_test_label_Tr=np.zeros([np.size(psi_test_label,0),2])  # 2 PC labels

    psi_test_input_Tr = psi_test_input
    psi_test_label_Tr = psi_test_label

    ## convert to torch tensor
    psi_test_input_Tr_torch = torch.from_numpy(psi_test_input_Tr).float()
    psi_test_label_Tr_torch = torch.from_numpy(psi_test_label_Tr).float()

    return psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, ndays-mem_list[-1]

# calculate the root-mean-square difference between the original and new predictions. 
def get_rmsd_one(mjo_ind='RMM', lead=0, exp_num='', testystat=2015, testyend=2020, m=1, mflg='off', wnx=1, wnxflg='off', rule='Iamp>1.0', vn='olr', dataflg='', outputflg='', winter=False, lat_range=20, channel=None, zero_channel=True, ptb_channel=False, after_relu=False, normflg=False):
    if zero_channel:
        # new prediction with one channel zeroed out
        fn = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/zero_channel_lastconv/'+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_c51_mem1_zero_'+str(channel)+'.nc'
        # reference prediction
    elif ptb_channel:
        if after_relu:
            dataflg1 = '_ptb_aftrelu_' + str(channel) + '_exp' + str(exp_num)
            fn = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/ptb_channel_lastconv/'+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_c51_mem1'+dataflg1+'.nc'
        else:
            dataflg1 = '_ptb_' + str(channel) + '_exp' + str(exp_num)
            fn = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/ptb_channel_lastconv/'+'predicted_MCDO_UNET_'+vn+str(lat_range)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(lead)+'_dailyinput_c51_mem1'+dataflg1+'.nc'

    fn0 = get_1varfiles_one(mjo_ind=mjo_ind, lead=lead, exp_num='', m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, vn=vn, dataflg=dataflg, winter=winter, outputflg=outputflg, lat_range=lat_range)

    rmm = xr.open_dataset(fn)[mjo_ind+'p'].sel(time=slice(str(testystat), str(testyend)+'-01-01')) # (time, 2)
    datesta = rmm.time.values[0]
    dateend = rmm.time.values[-1]
    rmm0 = xr.open_dataset(fn0)[mjo_ind+'p'].sel(time=slice(datesta, dateend))  # (time, 2)

    phase, iamp = get_phase_amp(mjo_ind, datesta, dateend, winter=winter)

    # target amplitude
    rmmt = xr.open_dataset(fn0)[mjo_ind+'t'].sel(time=slice(datesta, dateend))
    tamp = np.sqrt(rmmt[:,0]**2 + rmmt[:,1]**2)

    # select amp>1.0
    if rule=='Iamp>1.0':
        ind = np.where(iamp>1.0)[0]
        rmm_values = rmm[ind, :].values
        rmm0_values = rmm0[ind, :].values
    elif rule=='Tamp>1.0':
        ind = np.where(tamp>1.0)[0]
        rmm_values = rmm[ind, :].values
        rmm0_values = rmm0[ind, :].values

    rmsd = np.sqrt(np.mean((rmm_values[:,0]-rmm0_values[:,0])**2 + (rmm_values[:,1]-rmm0_values[:,1])**2))

    if normflg:
        rmsd = rmsd / np.sqrt(np.mean(rmm0_values[:,0]**2 + rmm0_values[:,1]**2))
        # print(np.sqrt(np.mean(rmm0_values[:,0]**2 + rmm0_values[:,1]**2)))

    return rmsd


def compute_get_rmsd_one(mjo_ind='RMM', lead=0, exp_num='', testystat=2015, testyend=2020, m=1, mflg='off', wnx=1, wnxflg='off', rule='Iamp>1.0', vn='olr', dataflg='', outputflg='', winter=False, lat_range=20, channel=None, zero_channel=True, ptb_channel=False, after_relu=False, normflg=False):
    rmsd = get_rmsd_one(mjo_ind=mjo_ind, lead=lead, exp_num=exp_num,testystat=testystat, testyend=testyend, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, rule=rule, vn=vn, dataflg=dataflg, winter=winter, outputflg=outputflg, lat_range=lat_range, channel=channel, zero_channel=zero_channel, ptb_channel=ptb_channel, after_relu=after_relu, normflg=normflg)
    return (lead, exp_num, channel), rmsd

def get_rmsd_parallel(mjo_ind='RMM', lead_list=[0,], exp_num_list=['',], testystat=2015, testyend=2020, m=1, mflg='off', wnx=1, wnxflg='off', rule='Iamp>1.0', vn='olr', dataflg='', winter=False, outputflg='', lat_range=20, fn_list=None, zero_channel=True, ptb_channel=False, after_relu=False, normflg=False):
    # mjo_ind: the index of MJO
    # lead_list: the list of lead time
    # fn_list: the numbers of channels to be zeroed

    rmsd_list = {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_get_rmsd_one, mjo_ind=mjo_ind, lead=lead, exp_num=exp_num,testystat=testystat, testyend=testyend, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, rule=rule, vn=vn, dataflg=dataflg, winter=winter, outputflg=outputflg, lat_range=lat_range, channel=fn, zero_channel=zero_channel, ptb_channel=ptb_channel, after_relu=after_relu, normflg=normflg) 
                    for lead in lead_list for exp_num in exp_num_list for fn in fn_list]
        
        for future in concurrent.futures.as_completed(futures):
            (lead, exp_num, fn), result = future.result()
            rmsd_list[(lead, exp_num, fn)] = result
        
    return rmsd_list
   
def resort_descending(x):
    resorted_x_ind = np.argsort(-x)
    resorted_x = x[resorted_x_ind]

    return resorted_x, resorted_x_ind

# check the scale for given channels. 
def get_fft_power(vn='olr',lat_lim=20,mjo_ind='RMM',lead=15,m=1,mflg='off',wnx=1,wnxflg='off',zmode=1,nmem=1,dataflg='',order=None, lat_avg=20, relu=True, rm_wn0=False):
    flag = vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(lead)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+dataflg

    with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/outfields'+flag+'.pickle', 'rb') as file2:
        outfields = pickle.load(file2)

    latitude = np.arange(20,-22,-2)
    lat_ind = list(np.where((latitude>=-lat_avg) & (latitude<=lat_avg))[0])

    time, batch, channel, lat, lon = np.shape(np.asarray(outfields['hidden5'])[:,:,:,lat_ind,:])
    hid5 = np.reshape(np.asarray(outfields['hidden5'])[:,:,:,lat_ind,:], (time*batch, channel, lat, lon))
    time, batch, channel, lat, lon = np.shape(np.asarray(outfields['hidden1'])[:,:,:,lat_ind,:])
    hid1 = np.reshape(np.asarray(outfields['hidden1'])[:,:,:,lat_ind,:], (time*batch, channel, lat, lon))
    
    x6 = np.concatenate((hid5, hid1), axis=1)
    if relu:
        # apply ReLu
        x6 = np.maximum(x6, 0)
    feature_maps = x6[:, order, :, :]
    feature_maps_fft = np.fft.fft(feature_maps, axis=-1)
    feature_maps_fft_power = np.mean(np.abs(feature_maps_fft)**2, axis=(0,2))  # average over time and latitudes (channel, lon)

    if rm_wn0:
        feature_maps_fft_power[:, 0] = 0.0
    feature_maps_fft_power_norm = feature_maps_fft_power / np.sum(feature_maps_fft_power, axis=-1, keepdims=True)
    # corresponding frequency
    freq = np.fft.fftfreq(lon, d=1.0)* lon 

    return freq, feature_maps_fft_power, feature_maps_fft_power_norm

# check the scale for given channels. 
def get_fft_power_input(vn='olr',lat_lim=20,mjo_ind='RMM',lead=15,m=1,mflg='off',wnx=1,wnxflg='off',zmode=1,nmem=1,dataflg='', lat_avg=20, rm_wn0=False):
    flag = vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(lead)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+dataflg

    with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/outfields'+flag+'.pickle', 'rb') as file2:
        outfields = pickle.load(file2)

    latitude = np.arange(20,-22,-2)
    lat_ind = list(np.where((latitude>=-lat_avg) & (latitude<=lat_avg))[0])


    time, batch, channel, lat, lon = np.shape(np.asarray(outfields['input_map'])[:,:,:,lat_ind,:])
    feature_maps = np.reshape(np.asarray(outfields['input_map'])[:,:,:,lat_ind,:], (time*batch, channel, lat, lon))
    
    feature_maps_fft = np.fft.fft(feature_maps, axis=-1)
    feature_maps_fft_power = np.mean(np.abs(feature_maps_fft)**2, axis=(0,2))  # average over time and latitudes (channel, lon)

    if rm_wn0:
        feature_maps_fft_power[:, 0] = 0.0
    feature_maps_fft_power_norm = feature_maps_fft_power / np.sum(feature_maps_fft_power, axis=-1, keepdims=True)
    # corresponding frequency
    freq = np.fft.fftfreq(lon, d=1.0)* lon 

    return freq, feature_maps_fft_power, feature_maps_fft_power_norm

# check the scale change for mechanism-denial experiments E4 (small-scale)
def get_fft_power_E1E4(vn='olr',lat_lim=20,mjo_ind='RMM',lead=10,m=10,mflg='resi',wnx=9,wnxflg='resi',zmode=1,nmem=1,dataflg='',order=None, lat_avg=20, relu=False, rm_wn0=True):
    flag = vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(lead)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+dataflg

    with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/outfields'+flag+'.pickle', 'rb') as file2:
        outfields = pickle.load(file2)

    latitude = np.arange(20,-22,-2)
    lat_ind = list(np.where((latitude>=-lat_avg) & (latitude<=lat_avg))[0])

    time, batch, channel, lat, lon = np.shape(np.asarray(outfields['input_layer'])[:,:,:,lat_ind,:])
    hid1 = np.reshape(np.asarray(outfields['input_layer'])[:,:,:,lat_ind,:], (time*batch, channel, lat, lon))
    time, batch, channel, lat, lon = np.shape(np.asarray(outfields['hidden1'])[:,:,:,lat_ind,:])
    hid2 = np.reshape(np.asarray(outfields['hidden1'])[:,:,:,lat_ind,:], (time*batch, channel, lat, lon))
    time, batch, channel, lat, lon = np.shape(np.asarray(outfields['hidden2'])[:,:,:,lat_ind,:])
    hid3 = np.reshape(np.asarray(outfields['hidden2'])[:,:,:,lat_ind,:], (time*batch, channel, lat, lon))
    time, batch, channel, lat, lon = np.shape(np.asarray(outfields['hidden3'])[:,:,:,lat_ind,:])
    hid4 = np.reshape(np.asarray(outfields['hidden3'])[:,:,:,lat_ind,:], (time*batch, channel, lat, lon))
    time, batch, channel, lat, lon = np.shape(np.asarray(outfields['hidden4'])[:,:,:,lat_ind,:])
    hid5 = np.reshape(np.asarray(outfields['hidden4'])[:,:,:,lat_ind,:], (time*batch, channel, lat, lon))
    time, batch, channel, lat, lon = np.shape(np.asarray(outfields['hidden5'])[:,:,:,lat_ind,:])
    hid6 = np.reshape(np.asarray(outfields['hidden5'])[:,:,:,lat_ind,:], (time*batch, channel, lat, lon))

    if relu:
        # apply ReLu
        hid1 = np.maximum(hid1, 0)
        hid2 = np.maximum(hid2, 0)
        hid3 = np.maximum(hid3, 0)
        hid4 = np.maximum(hid4, 0)
        hid5 = np.maximum(hid5, 0)
        hid6 = np.maximum(hid6, 0)

    feature_maps = {}
    feature_maps['hid1'] = hid1
    feature_maps['hid2'] = hid2
    feature_maps['hid3'] = hid3
    feature_maps['hid4'] = hid4
    feature_maps['hid5'] = hid5
    feature_maps['hid6'] = hid6

    feature_maps_fft = {}
    feature_maps_fft_power = {}
    feature_maps_fft_power_norm = {}

    for key in feature_maps.keys():
        feature_maps_fft[key] = np.fft.fft(feature_maps[key], axis=-1)
        feature_maps_fft_power[key] = np.mean(np.abs(feature_maps_fft[key])**2, axis=(0,2))  # average over time and latitudes (channel, frequency)

        if rm_wn0:
            feature_maps_fft_power[key][:, 0] = 0.0

        feature_maps_fft_power_norm[key] = feature_maps_fft_power[key] / np.sum(feature_maps_fft_power[key], axis=-1, keepdims=True)

    # corresponding frequency
    freq = np.fft.fftfreq(lon, d=1.0)* lon 

    return freq, feature_maps_fft_power_norm
