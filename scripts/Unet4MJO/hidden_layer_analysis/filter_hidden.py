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

import dask
from scipy import special
import math 

# module load pytorch/1.11.0
# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

# define parameters
mjo_ind = 'RMM' # os.environ["mjo_ind"]  # RMM or ROMI
leadmjo = 15 # lead for output (the MJO index)

filter_hidden = True  # filter out large scales or small scales in L2+L6 layer
relu_flg = True # if True, we filter the feature maps after ReLU; if False, we filter the feature maps before ReLU

cut_m = 3
cut_k = 2
cut_m_flg = 'resi'
cut_k_flg = 'resi'

testystat = 2015  # validation starts
dataflg = 'new'
nmem = int(1)  # the number of how many days we want to include into the input maps

testyend =  2020 # str(np.datetime64('2019-12-31'))- np.timedelta64(leadmjo+nmem,'D'))  # validation ends

c = int(51)

m = int(1)  # number of meridional modes
mflg = 'off'  # flaf of m. only use odd/even modes to reconstruct OLR. 
wnx = int(1)  # zonal wavenumber included
wnxflg = 'off'  # flag of wnx

lat_lim = int(20)  # maximum latitude in degree

# datadir = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/case_study/'

# if mjo_ind=='RMM':
#     Fnmjo = '/global/homes/l/linyaoly/ERA5/reanalysis/data_preproc/RMM_ERA5_case.nc'
# elif mjo_ind=='ROMI':
#     Fnmjo = '/global/homes/l/linyaoly/ERA5/reanalysis/data_preproc/ROMI_ERA5_case.nc'

# data_save = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/19maps_MCDO_ERA5_yproj_xfft_4gpus/'
# data_save = './test_'

datadir = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/fltano120/2012/'

if mjo_ind=='RMM':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
elif mjo_ind=='ROMI':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2021.nc'

# data_save = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/19maps_MCDO_ERA5_yproj_xfft_4gpus_new/'
path_forecasts = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/'
model_save = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/modelsave/'
pic_save = path_forecasts

os.makedirs(path_forecasts, exist_ok=True)
# os.makedirs(model_save, exist_ok=True)
# os.makedirs(pic_save, exist_ok=True)

def projection(vn, c=51, m=1, mflg='off', wnx=10, wnxflg='all', time_range=['1979-01-01','2019-12-31'], lat_range=[90, -90], pic_save='./',dataflg='new'):
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


def load_test_data(vn, Fnmjo,leadmjo,mem_list,testystat,testyend,c,m,mflg,wnx,wnxflg,lat_lim,mjo_ind,pic_save,dataflg='new'):
    # set parameters
    nmem = len(mem_list)  # memory length
    dimx = int(1 + 2 * int( int(lat_lim) / int(2)))
    dimy = 180

    # make projection and do zonal fft (time, lat, lon)
    olr_re = projection(vn, c, m, mflg, wnx, wnxflg, [str(testystat)+'-01-01', str(testyend-1)+'-12-31'], [lat_lim,-lat_lim],pic_save, dataflg=dataflg)

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


print('Code starts')

# variables used in the script
vn = 'olr'
# number of used variables
nmaps = 1

p = ''   # Monte Carlo Dropout rate = mcdp * 1%

mem_list = np.arange(nmem)

batch_size = 20
num_samples = 2
lambda_reg = 0.2

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

# Generate MCDO masks
mcdo_masks_conv1 = 1
mcdo_masks_conv2 = 1
mcdo_masks_conv3 = 1
mcdo_masks_conv4 = 1
mcdo_masks_conv5 = 1
mcdo_masks_conv6 = 1
mcdo_masks_outline1 = 1
mcdo_masks_outline2 = 1
mcdo_masks_outline3 = 1

def filter_hidden_layer(x2_np, cut_m, cut_k, cut_m_flg, cut_k_flg, c=51, lat_lim=20):
    # filter the activation maps of the hidden layer
    # x2_np: the activation maps of the hidden layer [batch_size, num_filters_enc, dimx, dimy]
    # cut_m: the number of meridional modes to be kept
    # cut_k: the number of zonal wavenumbers to be kept
    # cut_m_flg: 'all', 'odd', 'even', 'one', '1pls', 'no1', 'resi'
    # cut_k_flg: 'all', 'one', 'no0', 'no0p7', 'resi'

    if cut_m_flg=='off':
        olr_re = np.copy(x2_np)  # no dimension reduction on the meridional direction. 
    else:  
        beta= 2.28e-11  # variation of coriolis parameter with latitude
        L = np.sqrt(c / beta)  # horizontal scale (m)
    
        lat = np.arange(lat_lim, -lat_lim-2, -2)
        y = lat * 110 * 1000 / L # dimensionless

        # define meridianol wave structures
        phi = []

        if cut_m_flg=='odd':
            m_list = np.arange(1,cut_m,2)  
        elif cut_m_flg=='even':
            m_list = np.arange(0,cut_m,2)
        elif cut_m_flg=='all':
            m_list = np.arange(cut_m)
        elif cut_m_flg=='one':
            m_list = [cut_m-1]
        elif cut_m_flg=='1pls':
            m_list = [0,cut_m-1]
        elif cut_m_flg=='no1':
            m_list = np.arange(1,cut_m)
        elif cut_m_flg=='resi':
            m_list = np.arange(cut_m)  # this is the part to be removed from filtered map
        else:
            print('wrong m flag!')
            exit()

        for i in m_list:
            p = special.hermite(i)
            Hm = p(y)
            phim = np.exp(- y**2 / 2) * Hm / np.sqrt((2**i) * np.sqrt(np.pi) * math.factorial(i))

            phi.append(np.reshape(phim, (1, 1, len(y), 1)))

        # projection coefficients
        olrm = []

        dy = (lat[0] - lat[1]) * 110 * 1000 / L 

        for i in range(len(m_list)):
            um = np.sum(x2_np * phi[i] * dy, axis=-2, keepdims=True)  # (time,channel, 1, lon)
            olrm.append(um)

        # reconstruction 
        olr_re = np.zeros(np.shape(x2_np))  # (time, channel, lat, lon)

        for i in range(len(m_list)):
            olr_re = olr_re + olrm[i] * phi[i]
        
        if cut_m_flg=='resi':
            olr_re1 = x2_np - olr_re
            del olr_re
            olr_re = np.copy(olr_re1)
            del olr_re1

    if cut_k_flg=='off':
        olr_re_fft = np.copy(olr_re)
    else:
        # do fourier transform along each latitude at each time step
        coef_fft = np.fft.rfft(olr_re, axis=-1)
        # remove waves whose zonal wave cycles are larger than wnx
        if cut_k_flg=='all':
            coef_fft[:,:,:,cut_k+1:] = 0.0 
        elif cut_k_flg=='one':
            if wnx==0:
                coef_fft[:,:,:,cut_k+1:] = 0.0
            else:
                coef_fft[:,:,:,:cut_k] = 0.0
                coef_fft[:,:,:,cut_k+1:] = 0.0 
        elif cut_k_flg=='no0':  # include 1, 2, ..., wnx
            coef_fft[:,:,:,cut_k+1:] = 0.0 
            coef_fft[:,:,:,0] = 0.0
        elif cut_k_flg=='no0p7': # include 1, 2, ..., wnx, 7
            coef_fft[:,:,:,cut_k+1:7] = 0.0 
            coef_fft[:,:,:,8:] = 0.0 
            coef_fft[:,:,:,0] = 0.0
        elif cut_k_flg=='resi':  # resi of 0-wnx[inclusive]
             coef_fft[:,:,:,:cut_k+1] = 0.0
        else:
            print('wrong cut_k flag!')
            exit()
        # reconstruct OLR with selected zonal waves
        olr_re_fft = np.fft.irfft(coef_fft, np.shape(olr_re)[-1], axis=-1)

    return olr_re_fft # (time, lat, lon)


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

        x1 = F.relu(self.dropoutconv1(self.input_layer(x)))
        conv_x1 = self.dropoutconv2(self.hidden1(x1))
        x2 = F.relu(conv_x1)
        x3 = F.relu(self.dropoutconv3(self.hidden2(x2)))
        x4 = F.relu(self.dropoutconv4(self.hidden3(x3)))
        x5 = torch.cat((F.relu(self.dropoutconv5(self.hidden4(x4))),x3), dim =1)
        conv_x5 = self.dropoutconv6(self.hidden5(x5)) 

        # filter the activation maps of the hidden1 layer
        if filter_hidden:
            if relu_flg:
                x6 = torch.cat((F.relu(conv_x5),x2), dim =1)
                # convert to numpy array for filtering
                x2_np = x6.data.cpu().numpy()
                # channel = np.shape(x2_np)[1]
                # print('shape of x2_np',np.shape(x2_np)) [batch_size, num_filters_enc, dimx, dimy]
                filtered_x6_np = filter_hidden_layer(x2_np, cut_m, cut_k, cut_m_flg, cut_k_flg)
                # convert back to torch tensor
                x6 = torch.from_numpy(filtered_x6_np).float().cuda()
            else:
                conv_x6 = torch.cat((conv_x5,conv_x1), dim =1)
                x2_np = conv_x6.data.cpu().numpy()
                filtered_x6_np = filter_hidden_layer(x2_np, cut_m, cut_k, cut_m_flg, cut_k_flg)
                x6 = torch.from_numpy(filtered_x6_np).float().cuda()
                x6 = F.relu(x6)
        else:
            x6 = torch.cat((F.relu(conv_x5),x2), dim =1)

        x6 = x6.view(-1,featureDim)
        x6 = F.relu(self.FC1(x6))
        x7 = F.relu(self.FC2(self.dropoutline1(x6) * mcdo_masks_outline1))
        x8 = F.relu(self.FC3(self.dropoutline2(x7) * mcdo_masks_outline2))

        out = (self.FC4(self.dropoutline3(x8) * mcdo_masks_outline3))

        return out

net = CNN()  

net.cuda()  # send the net to GPU

print('Model starts')

# vn,Fnmjo,leadmjo,mem_list,testystat,testyend,m,mflg,wnx,wnxflg,lat_lim,mjo_ind,c=51,pic_save='./',dataflg=''

psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, M = load_test_data(vn,Fnmjo,leadmjo,mem_list,testystat,testyend,c,m,mflg,wnx,wnxflg,lat_lim,mjo_ind,pic_save,dataflg=dataflg)
psi_test_input_Tr_torch_norm = np.zeros(np.shape(psi_test_input_Tr_torch))

for leveln in np.arange(0,nmem):
        M_test_level = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
        STD_test_level = torch.std(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
        psi_test_input_Tr_torch_norm[:,leveln,None,:,:] = ((psi_test_input_Tr_torch[:,leveln,None,:,:]-M_test_level)/STD_test_level)

psi_test_input_Tr_torch  = torch.from_numpy(psi_test_input_Tr_torch_norm).float()

print('shape of normalized input test',psi_test_input_Tr_torch.shape)
print('shape of normalized label test',psi_test_label_Tr_torch.shape)
###############################################################################

net.load_state_dict(torch.load(model_save+'predicted_MCDO_UNET_olr20deg_RMM_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_c'+str(c)+'_nmem'+str(nmem)+'.pt'))
net.eval()

print('Model loaded')

batch_size = 20
testing_data_loader = torch.utils.data.DataLoader(psi_test_input_Tr_torch, batch_size=batch_size, drop_last=True)

M = int( M // batch_size) * batch_size
autoreg_pred = []
autoreg_true = psi_test_label_Tr[:M,:]

# Disable MCDO during testing
for batch in testing_data_loader:
        batch = batch.cuda()
        net.eval()
        autoreg_pred.append(net(batch).data.cpu().numpy()) # Nlat changed to 1 for hovmoller forecast

autoreg_pred1 = np.concatenate(autoreg_pred, axis=0)

t0 = np.datetime64(str(testystat)) + np.timedelta64(mem_list[-1], 'D')
tb = t0 + np.timedelta64(M,'D')

t1 = np.arange(t0,tb)

print('prediction starts')

# create prediction RMM time series
RMMp = xr.DataArray(
        data=autoreg_pred1,
        dims=['time','mode'],
        coords=dict(
                time=t1,
                mode=[0,1]
        ),
        attrs=dict(
                description=mjo_ind+' prediction'
        ),
        name=mjo_ind+'p',
)

# create true RMM time series
RMMt = xr.DataArray(
        data=autoreg_true,
        dims=['time','mode'],
        coords=dict(
                time=t1,
                mode=[0,1]
        ),
        attrs=dict(
                description=mjo_ind+' truth'
        ),
        name=mjo_ind+'t',
)

ds = xr.merge([RMMp, RMMt])

if filter_hidden:
    if relu_flg:
        dataflg = f'_mcut{cut_m}kcut{cut_k}_{cut_m_flg}{cut_k_flg}'
        ds.to_netcdf(path_forecasts+'flt_hid26_relu_predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(leadmjo)+dataflg+'.nc', mode='w')
    else:
        dataflg = f'_mcut{cut_m}kcut{cut_k}_{cut_m_flg}{cut_k_flg}'
        ds.to_netcdf(path_forecasts+'flt_hid26_predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(leadmjo)+dataflg+'.nc', mode='w')
else:
    ds.to_netcdf(path_forecasts+'predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(leadmjo)+'_dailyinput_c'+str(c)+'_mem'+str(nmem)+'d'+dataflg+'.nc', mode='w')
