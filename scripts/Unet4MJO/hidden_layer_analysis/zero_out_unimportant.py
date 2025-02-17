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
mjo_ind = os.environ["mjo_ind"]  # RMM or ROMI
zero_num = int(os.environ["zero_num"])  # number of channels to be maintained

# ############### change parameters ###############
zero_channel = True 

# define which channel to zero out
channel_list=list(np.arange(0,192))  # which channel to be zeroed out 
rule = 'Iamp>1.0'
# rule = 'Tamp>1.0'
rmsd_list = ctr.get_rmsd_parallel(mjo_ind=mjo_ind, lead_list=[leadmjo,],fn_list=channel_list, zero_channel=True, ptb_channel=False, rule=rule)
rmsd_zero = np.empty((len(channel_list)))

for i in range(len(channel_list)):
    rmsd_zero[i] = rmsd_list[(leadmjo,'',channel_list[i])]

print('rmsd_zero shape: ',rmsd_zero.shape)

_, order_zero_list = ctr.resort_descending(np.squeeze(rmsd_zero))

channel = order_zero_list[zero_num:]  # which channel to be zeroed out

print('channel',channel)        

m = int(1)  # number of meridional modes
mflg = 'off'  # flaf of m. only use odd/even modes to reconstruct OLR. 
wnx = int(1)  # zonal wavenumber included
wnxflg = 'off'  # flag of wnx

lat_lim = int(20)  # maximum latitude in degree


datadir = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/fltano120/2012/'

if mjo_ind=='RMM':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
elif mjo_ind=='ROMI':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'

# data_save = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/19maps_MCDO_ERA5_yproj_xfft_4gpus_new/'
path_forecasts = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/zero_channel_lastconv/'

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

# load model: selected default parameters: zero_channel=None, perturb=None, after_relu=False, lead=15
net = ctr.get_UNet(nmaps=1, lat_lim=lat_lim, zero_channel=channel)

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
        dataflg = '_zero_all' + str(zero_num) + 'pls'
        # dataflg = '_zero_' + str(channel[0]) + '_' + str(channel[-1])
        # dataflg = '_zero_all10pls'
else:
        dataflg = ''

ds.to_netcdf(path_forecasts+'predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(leadmjo)+'_dailyinput_c'+str(c)+'_mem'+str(nmem)+dataflg+'.nc', mode='w')
