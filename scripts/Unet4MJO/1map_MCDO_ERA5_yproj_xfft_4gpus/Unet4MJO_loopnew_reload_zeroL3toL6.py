# This job only takes 1-1.5 hours to run. 
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import numpy as np
import sys
# import netCDF4 as nc
# from saveNCfile import savenc
# from saveNCfile_for_activations import savenc_for_activations
from data_loader_loopnew import load_test_data
from data_loader_loopnew import load_train_data
# from prettytable import PrettyTable
from count_trainable_params import count_parameters
# import hdf5storage
import pandas as pd 
import xarray as xr 
import dask 
import os 
from datetime import date 
from projection_global import projection

print('Code starts')
# April 18: project OLR onto meridional wave structures
# Updates Nov 13, 2022: Add 2dMonte
# input: 6 global maps at three time steps
# output: RMMERA5 at one time step

# parameters to be set
vn = os.environ["varn"]  # variable name

ysta = int(os.environ["ysta_train"])  # training starts
yend = int(os.environ["yend_train"])  # training ends

testystat = int(os.environ["ysta_test"])  # validation starts
testyend = int(os.environ["yend_test"])  # validation ends

leadmjo = int(os.environ["lead30d"]) # lead for output (the MJO index)
nmem = int(os.environ["memlen"])  # the number of how many days we want to include into the input maps
print('leadmjo: '+str(leadmjo))
print('nmem: '+str(nmem))

c = int(os.environ["c"])  # selected verticla mode 
m = int(os.environ["m"])  # number of meridional modes
mflg = os.environ["mflg"]  # flaf of m. only use odd/even modes to reconstruct OLR. 
wnx = int(os.environ["wnx"])  # zonal wavenumber included
wnxflg = os.environ["wnxflg"]  # flag of wnx

dataflg = os.environ["dataflg"]  # flag of data input; defult is new for RMM and new40 for ROMI

lat_lim = int(os.environ["lat_lim"])  # maximum latitude in degree

mjo_ind = os.environ["mjo_ind"]  # RMM or ROMI

exp_num = os.environ["exp_num"]  # experiment number

nmaps = 1

# num_epochs = 2
# Nsamp = 2

num_epochs = 100   # how many loops we want to train the model. 
# Nsamp = 100  # MC runs applied to forecast of testing data

datadir = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'
# vn = 'olr'
# Fn = datadir+'ERA5.'+vn+'GfltG.day.1901to2020.nc'

# # Fnmjo = '/global/homes/l/linyaoly/ERA5/reanalysis/RMM_ERA5_daily.nc'
# Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1901to2020.nc'

if mjo_ind=='RMM':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
elif mjo_ind=='ROMI':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'

data_save = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/'
# data_save = './test_'

path_forecasts = data_save + 'output'+str(exp_num)+'/'
# # path_forecasts = './test_output/'
model_save = data_save + 'modelsave'+str(exp_num)+'/'
pic_save = data_save + 'picsave/'

os.makedirs(path_forecasts, exist_ok=True)
os.makedirs(model_save, exist_ok=True)
os.makedirs(pic_save, exist_ok=True)

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
mcdo_masks_conv6 = 0  # zero out L3 to L6
mcdo_masks_outline1 = 1
mcdo_masks_outline2 = 1
mcdo_masks_outline3 = 1

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

        x1 = F.relu (self.dropoutconv1(self.input_layer(x)) * mcdo_masks_conv1)
        x2 = F.relu (self.dropoutconv2(self.hidden1(x1)) * mcdo_masks_conv2)
        x3 = F.relu (self.dropoutconv3(self.hidden2(x2)) * mcdo_masks_conv3)
        x4 = F.relu (self.dropoutconv4(self.hidden3(x3)) * mcdo_masks_conv4)

        x5 = torch.cat ((F.relu(self.dropoutconv5(self.hidden4(x4)) * mcdo_masks_conv5),x3), dim =1)
        x6 = torch.cat ((F.relu(self.dropoutconv6(self.hidden5(x5)) * mcdo_masks_conv6),x2), dim =1)
        x6 = x6.view(-1,featureDim)
        x6 = F.relu(self.FC1(x6))
        x7 = F.relu(self.FC2(self.dropoutline1(x6) * mcdo_masks_outline1))
        x8 = F.relu(self.FC3(self.dropoutline2(x7) * mcdo_masks_outline2))

        out = (self.FC4(self.dropoutline3(x8) * mcdo_masks_outline3))

        return out

net = CNN()  

net.cuda()  # send the net to GPU

print('Model starts')

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

if dataflg=='new':
     dataflg=''

# # NOTE: COMMENT STARTS
# net.eval()
# torch.save(net.state_dict(), model_save+'predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_c'+str(c)+'_nmem'+str(nmem)+dataflg+'.pt')
# # NOTE: COMMENT ENDS

net.load_state_dict(torch.load(model_save+'predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_c'+str(c)+'_nmem'+str(nmem)+dataflg+'.pt'))
net.eval()

print('BNN Model Saved')

net.eval()

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


dataflg ='ZeroL3toL6'
ds.to_netcdf(path_forecasts+'predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(leadmjo)+'_dailyinput_'+str(ysta)+'to'+str(yend)+'_c'+str(c)+'_mem'+str(nmem)+'d'+dataflg+'.nc', mode='w')

