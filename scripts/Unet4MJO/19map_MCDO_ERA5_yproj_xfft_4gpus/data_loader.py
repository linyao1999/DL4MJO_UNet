import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import numpy as np
import pandas as pd  
import xarray as xr 
from projection import projection

def load_test_data(vn, Fnmjo,leadmjo,mem_list,testystat,testyend,c,m,mflg,wnx,wnxflg,lat_lim,mjo_ind,pic_save,dataflg='new'):
    # set parameters
    nmem = len(mem_list)  # memory length
    nmaps = len(vn)  # number of variables
    dimx = int(1 + 2 * int(lat_lim / 2))
    dimy = 180
    ndays = int( ( np.datetime64(str(testyend)+'-01-01') - np.datetime64(str(testystat)+'-01-01') ) / np.timedelta64(1,'D') ) 

    # make projection and do zonal fft (time, lat, lon) one by one
    psi_test_input = np.zeros((ndays-mem_list[-1],nmem*nmaps,dimx,dimy))

    for i in np.arange(nmaps):
        # for each variable
        olr_re = projection(vn[i], c, m, mflg, wnx, wnxflg, [str(testystat)+'-01-01', str(testyend-1)+'-12-31'], [lat_lim,-lat_lim], pic_save,dataflg)
        # olr_re(time,lat,lon)
        
        # for each time step of a given variable
        for j in range(ndays-mem_list[-1]):
            psi_test_input[j, i*nmem:(i+1)*nmem, :, :] = olr_re[j:j+nmem, :, :]

        print('input shape inside the j loop: ', np.shape(psi_test_input[j, i*nmem:(i+1)*nmem, :, :]))
        print('olr_re shape inside the j loop: ', np.shape(olr_re[j:j+nmem, :, :]))

    # psi_test_input(time, nmaps*nmem, lat, lon)

    # print('combined input shape is: ' + str(np.shape(psi_test_input[i, :, :, :])))

    # read the MJO index
    FFmjo = xr.open_dataset(Fnmjo)
    FFmjo = FFmjo.sel(time=slice(str(testystat)+'-01-01', str(testyend)+'-03-31'))
    # FFmjo.fillna(0)
    pc = np.asarray(FFmjo[mjo_ind])

    Nlat=dimx
    Nlon=dimy

    psi_test_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays-mem_list[-1],:]

    psi_test_input_Tr=np.zeros([np.size(psi_test_input,0),nmem*nmaps,Nlat,Nlon])   # vn input maps
    psi_test_label_Tr=np.zeros([np.size(psi_test_label,0),2])  # 2 PC labels

    psi_test_input_Tr = psi_test_input
    psi_test_label_Tr = psi_test_label

    ## convert to torch tensor
    psi_test_input_Tr_torch = torch.from_numpy(psi_test_input_Tr).float()
    psi_test_label_Tr_torch = torch.from_numpy(psi_test_label_Tr).float()

    return psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, ndays-mem_list[-1]


def load_train_data(vn,Fnmjo,leadmjo,mem_list,ysta,yend,c,m,mflg,wnx,wnxflg,lat_lim,mjo_ind,pic_save,dataflg='new'):
    # set parameters
    nmem = len(mem_list)  # memory length
    nmaps = len(vn)  # number of variables
    dimx = int(1 + 2 * int(lat_lim / 2))
    dimy = 180
    ndays = int( ( np.datetime64(str(yend)+'-01-01') - np.datetime64(str(ysta)+'-01-01') ) / np.timedelta64(1,'D') ) 

    # make projection and do zonal fft (time, lat, lon) one by one
    psi_train_input = np.zeros((ndays-mem_list[-1],nmem*nmaps,dimx,dimy))

    for i in np.arange(nmaps):
        # for each variable
        olr_re = projection(vn[i], c, m, mflg, wnx, wnxflg, [str(ysta)+'-01-01', str(yend-1)+'-12-31'], [lat_lim,-lat_lim],pic_save,dataflg)
        # olr_re(time,lat,lon)
        
        # for each time step of a given variable
        for j in range(ndays-mem_list[-1]):
            psi_train_input[j, i*nmem:(i+1)*nmem, :, :] = olr_re[j:j+nmem, :, :]

    
    # read the MJO index
    FFmjo = xr.open_dataset(Fnmjo)
    FFmjo = FFmjo.sel(time=slice(str(ysta)+'-01-01', str(yend)+'-03-31'))
    FFmjo.fillna(0)
    pc = np.asarray(FFmjo[mjo_ind])

    Nlat=dimx
    Nlon=dimy

    psi_train_label = pc[mem_list[-1]+leadmjo:mem_list[-1]+leadmjo+ndays-mem_list[-1],:]
    print('label shape is: ' + str(np.shape(psi_train_label)))

    psi_train_input_Tr=np.zeros([np.size(psi_train_input,0),nmem*nmaps,Nlat,Nlon])   # vn input maps
    psi_train_label_Tr=np.zeros([np.size(psi_train_label,0),2])  # 2 PC labels

    psi_train_input_Tr = psi_train_input
    psi_train_label_Tr = psi_train_label

    ## convert to torch tensor
    psi_train_input_Tr_torch = torch.from_numpy(psi_train_input_Tr).float()
    psi_train_label_Tr_torch = torch.from_numpy(psi_train_label_Tr).float()

    return psi_train_input_Tr_torch, psi_train_label_Tr_torch, ndays-mem_list[-1]
