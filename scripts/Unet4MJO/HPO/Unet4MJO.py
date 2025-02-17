# This job only takes 1-1.5 hours to run. 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import load_test_data
from data_loader import load_train_data

import numpy as np
import xarray as xr 
import os 

print('Code starts')

# parameters to be set
batch_size = int(os.environ["batch_size"])  # batch size
kernel_size = int(os.environ["kernel_size"])  # kernel size
drop_prob = float(os.environ["drop_prob"])  # dropout probability
optimizer_type = os.environ["optimizer_type"]  # optimizer type
learning_rate = float(os.environ["learning_rate"])  # learning rate

print('batch_size',batch_size)
print('kernel_size',kernel_size)
print('drop_prob',drop_prob)
print('optimizer_type',optimizer_type)
print('learning_rate',learning_rate)

# parameters not changed
vn = 'olr'  # variable name
ysta = 1979  # training starts
yend = 2015  # training ends
testystat = 2015  # validation starts
testyend = 2022  # validation ends
leadmjo = int(os.environ["lead30d"]) # lead for output (the MJO index)
nmem = 1  # the number of how many days we want to include into the input maps
c = 51  # selected verticla mode 
m = int(os.environ["m"])  # number of meridional modes
mflg = os.environ["mflg"]  # flaf of m. only use odd/even modes to reconstruct OLR. 
wnx = int(os.environ["wnx"])  # zonal wavenumber included
wnxflg = os.environ["wnxflg"]  # flag of wnx
lat_lim = 20  # maximum latitude in degree
mjo_ind = os.environ["mjo_ind"]  # RMM or ROMI
exp_num = ''  # experiment number

nmaps = 1
num_epochs = 100   # how many loops we want to train the model. 
Nsamp = 100  # MC runs applied to forecast of testing data

datadir = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/'

if mjo_ind=='RMM':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
elif mjo_ind=='ROMI':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'

data_save = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/HPO/'
path_forecasts = data_save + 'output'+str(exp_num)+'/'
pic_save = data_save + 'picsave/'
os.makedirs(path_forecasts, exist_ok=True)
os.makedirs(pic_save, exist_ok=True)

mem_list = np.arange(nmem)
num_samples = batch_size

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
        self.input_layer = (nn.Conv2d(imgChannels, num_filters_enc, kernel_size=kernel_size, stride=1, padding='same'))
        self.hidden1 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=kernel_size, stride=1, padding='same' ))
        self.hidden2 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=kernel_size, stride=1, padding='same' ))
        self.hidden3 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=kernel_size, stride=1, padding='same' ))
        self.hidden4 = (nn.Conv2d(num_filters_enc, num_filters_enc, kernel_size=kernel_size, stride=1, padding='same' ))


        self.hidden5 = (nn.Conv2d(num_filters_dec1, num_filters_dec1, kernel_size=kernel_size, stride=1, padding='same' ))
        self.hidden6 = (nn.Conv2d(num_filters_dec2, num_filters_dec2, kernel_size=kernel_size, stride=1, padding='same' ))

        self.FC1 = nn.Linear(featureDim,nhidden1)
        self.FC2 = nn.Linear(nhidden1,nhidden2)
        self.FC3 = nn.Linear(nhidden2,nhidden3)
        self.FC4 = nn.Linear(nhidden3,out_channels)

        self.dropoutconv1 = nn.Dropout2d(p=drop_prob)
        self.dropoutconv2 = nn.Dropout2d(p=drop_prob)
        self.dropoutconv3 = nn.Dropout2d(p=drop_prob)
        self.dropoutconv4 = nn.Dropout2d(p=drop_prob)
        self.dropoutconv5 = nn.Dropout2d(p=drop_prob)
        self.dropoutconv6 = nn.Dropout2d(p=drop_prob)
        self.dropoutline1 = nn.Dropout(p=drop_prob)
        self.dropoutline2 = nn.Dropout(p=drop_prob)
        self.dropoutline3 = nn.Dropout(p=drop_prob)
        

    def forward (self,x):

        x1 = F.relu (self.dropoutconv1(self.input_layer(x)))
        x2 = F.relu (self.dropoutconv2(self.hidden1(x1)))
        x3 = F.relu (self.dropoutconv3(self.hidden2(x2)))
        x4 = F.relu (self.dropoutconv4(self.hidden3(x3)))

        x5 = torch.cat ((F.relu(self.dropoutconv5(self.hidden4(x4))),x3), dim =1)
        x6 = torch.cat ((F.relu(self.dropoutconv6(self.hidden5(x5))),x2), dim =1)
        x6 = x6.view(-1,featureDim)
        x6 = F.relu(self.FC1(x6))
        x7 = F.relu(self.FC2(self.dropoutline1(x6)))
        x8 = F.relu(self.FC3(self.dropoutline2(x7)))

        out = (self.FC4(self.dropoutline3(x8)))

        return out

net = CNN()  

net.cuda()  # send the net to GPU

loss_fn = nn.MSELoss()

if optimizer_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
elif optimizer_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

print('Model starts')

psi_train_input_Tr_torch, psi_train_label_Tr_torch, trainN = load_train_data(vn,Fnmjo,leadmjo,mem_list,ysta,yend,c,m,mflg,wnx,wnxflg,lat_lim,mjo_ind,pic_save)
psi_train_input_Tr_torch_norm = np.zeros(np.shape(psi_train_input_Tr_torch))

for levelnloop in np.arange(0,nmem):
        M_train_level = torch.mean(torch.flatten(psi_train_input_Tr_torch[:,levelnloop,:,:]))
        STD_train_level = torch.std(torch.flatten(psi_train_input_Tr_torch[:,levelnloop,:,:]))
        psi_train_input_Tr_torch_norm[:,levelnloop,None,:,:] = ((psi_train_input_Tr_torch[:,levelnloop,None,:,:]-M_train_level)/STD_train_level)

psi_train_input_Tr_torch = torch.from_numpy(psi_train_input_Tr_torch_norm).float()

print('shape of normalized input test',psi_train_input_Tr_torch.shape)
print('shape of normalized label test',psi_train_label_Tr_torch.shape)


psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, M = load_test_data(vn,Fnmjo,leadmjo,mem_list,testystat,testyend,c,m,mflg,wnx,wnxflg,lat_lim,mjo_ind,pic_save)
psi_test_input_Tr_torch_norm = np.zeros(np.shape(psi_test_input_Tr_torch))

for leveln in np.arange(0,nmem):
        M_test_level = torch.mean(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
        STD_test_level = torch.std(torch.flatten(psi_test_input_Tr_torch[:,leveln,:,:]))
        psi_test_input_Tr_torch_norm[:,leveln,None,:,:] = ((psi_test_input_Tr_torch[:,leveln,None,:,:]-M_test_level)/STD_test_level)

psi_test_input_Tr_torch  = torch.from_numpy(psi_test_input_Tr_torch_norm).float()

print('shape of normalized input test',psi_test_input_Tr_torch.shape)
print('shape of normalized label test',psi_test_label_Tr_torch.shape)
###############################################################################
# net.eval()
net.train()

for epoch in range(0, num_epochs):  # loop over the dataset multiple times
    
        for step in range(0,trainN-batch_size,batch_size):
            # get the inputs; data is a list of [inputs, labels]
            indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
            input_batch, label_batch = psi_train_input_Tr_torch[indices,:,:,:], psi_train_label_Tr_torch[indices,:]
            print('shape of input', input_batch.shape)
            print('shape of output', label_batch.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            output= net(input_batch.cuda())

            loss = loss_fn(output, label_batch.cuda())

            loss.backward()
            optimizer.step()
            output_val= net(psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,nmaps*nmem,dimx,dimy]).cuda()) # Nlat changed to 1 for hovmoller forecast
            val_loss = loss_fn(output_val, psi_test_label_Tr_torch[0:num_samples].reshape([num_samples,2]).cuda()) # Nlat changed to 1 for hovmoller forecast
            # print statistics

            if step % 50 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, loss))
                print('[%d, %5d] val_loss: %.3f' %
                    (epoch + 1, step + 1, val_loss))
                running_loss = 0.0

            del input_batch
            del label_batch


print('Finished Training')

net.eval()

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

dataflg = '_'+str(batch_size)+'_'+str(kernel_size)+'_'+str(drop_prob)+'_'+optimizer_type+'_'+str(learning_rate)+'_'+str(num_epochs)+'_'


ds.to_netcdf(path_forecasts+'predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'ERA5_'+str(m)+'modes'+mflg+'_wnx'+str(wnx)+wnxflg+'_lead'+str(leadmjo)+dataflg+'.nc', mode='w')

