import sys
sys.path.append('../util/')
import os 
import torch 
import numpy as np  
import xarray as xr
import matplotlib.pyplot as plt
import contribution_util as ctr


# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
# export lead=0
# module load pytorch/1.11.0
# ############ define parameters ############
vn = 'olr'
testystat = 2015  # validation starts
testyend =  2020 # str(np.datetime64('2019-12-31'))- np.timedelta64(leadmjo+nmem,'D'))  # validation ends
dataflg = 'new'
leadmjo = int(os.environ["lead"]) # lead for output (the MJO index)
print('leadmjo',leadmjo)


# ############### change parameters ###############
zero_channel = False 
if zero_channel:
     channel = int(os.environ["channel"])
        # channel = np.arange(128,192)
        # channel = np.arange(0,128)
#         channel = [183, 171, 163,
#        147, 164, 185, 155, 129, 151, 190, 181, 177, 133, 134, 131, 178,
#        135, 145, 153, 165, 160, 128, 158, 130, 169, 188, 132, 161, 172,
#        152, 186, 187, 170, 142, 154, 174, 149, 144, 137, 162, 148, 157,
#        136, 138, 156, 179, 184, 173, 176,  85,  36, 139, 146,  17,  21,
#        150,  53,  32,  42, 107,  49, 159, 123,  44, 141, 121,  80,  15,
#         99, 105, 102,  19, 113, 115, 125,  31,  58,  57, 124, 114,  26,
#         54,  63,  56,  48,  98,  61,  13,  46,  92,  81, 101,  41,  67,
#         71,  23,  29, 120,  91,  51,  33,  90, 117,  34, 100,  12,  22,
#         18,  76, 127,   4,  40,  95, 126, 106,  74,  59,  60, 108, 109,
#         27,  65,  83,   6,  45,  96,  37,  79, 111,  43,  73,   7,  24,
#        104,  52,  77,  10,  87,  68,  89,  64,  69,   5,  88,   8,  35,
#         14,   1,  94, 119, 110,   9,  66,  62,   2,  55, 112,  70, 116,
#         86, 118,  84,  20,  11, 122,   3,  97,  16,  72,  50,  25,  47,
#         93,   0,  28,  82,  38,  39, 103,  78,  30,  75]

ptb_channel = True
after_relu = True

if ptb_channel:
       channel = int(os.environ["channel"])
       exp = int(os.environ["exp"])
# ###################################################


m = int(1)  # number of meridional modes
mflg = 'off'  # flaf of m. only use odd/even modes to reconstruct OLR. 
wnx = int(1)  # zonal wavenumber included
wnxflg = 'off'  # flag of wnx

lat_lim = int(20)  # maximum latitude in degree

mjo_ind = 'ROMI' # os.environ["mjo_ind"]  # RMM or ROMI

datadir = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/fltano120/2012/'

if mjo_ind=='RMM':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
elif mjo_ind=='ROMI':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'

# data_save = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/19maps_MCDO_ERA5_yproj_xfft_4gpus_new/'
if zero_channel:
        path_forecasts = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/zero_channel_lastconv/'
elif ptb_channel:
        path_forecasts = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/ptb_channel_lastconv/'

model_save = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/modelsave/'
pic_save = path_forecasts

os.makedirs(path_forecasts, exist_ok=True)

############################ load data ############################
nmem = 1 
mem_list = np.arange(nmem)
c=51

psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, M = ctr.load_test_data(vn=vn,Fnmjo=Fnmjo,leadmjo=leadmjo,testystat=testystat,testyend=testyend,m=m,mflg=mflg,wnx=wnx,wnxflg=wnxflg,lat_lim=lat_lim,mjo_ind=mjo_ind,dataflg=dataflg)
psi_test_input_Tr_torch_norm = np.zeros(np.shape(psi_test_input_Tr_torch))

for leveln in np.arange(0,nmem):
        M_test_level = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
        STD_test_level = torch.std(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
        psi_test_input_Tr_torch_norm[:,leveln,None,:,:] = ((psi_test_input_Tr_torch[:,leveln,None,:,:]-M_test_level)/STD_test_level)

psi_test_input_Tr_torch  = torch.from_numpy(psi_test_input_Tr_torch_norm).float()

print('shape of normalized input test',psi_test_input_Tr_torch.shape)
print('shape of normalized label test',psi_test_label_Tr_torch.shape)
###############################################################################

# load model
if zero_channel:
        net = ctr.get_UNet(nmaps=1, lat_lim=lat_lim, zero_channel=channel)
elif ptb_channel:
        net = ctr.get_UNet(nmaps=1, lat_lim=lat_lim, perturb=channel, after_relu=after_relu, lead=leadmjo, vn=vn, mjo_ind=mjo_ind)
else: 
        net = ctr.get_UNet(nmaps=1, lat_lim=lat_lim)

if wnxflg=='off':
    dataflg = ''
else:
    dataflg = 'glb'
net.load_state_dict(torch.load(model_save+'predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_c'+str(c)+'_nmem'+str(nmem)+dataflg+'.pt'))
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

if zero_channel:
        dataflg = '_zero_' + str(channel)
        # dataflg = '_zero_' + str(channel[0]) + '_' + str(channel[-1])
        # dataflg = '_zero_all10pls'
elif ptb_channel:
        if after_relu:
                dataflg = '_ptb_aftrelu_' + str(channel) + '_exp' + str(exp) 
        else:
                dataflg = '_ptb_' + str(channel) + '_exp' + str(exp)
else:
        dataflg = ''

ds.to_netcdf(path_forecasts+'predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(leadmjo)+'_dailyinput_c'+str(c)+'_mem'+str(nmem)+dataflg+'.nc', mode='w')
