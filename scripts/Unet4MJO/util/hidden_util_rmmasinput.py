import numpy as np 
import xarray as xr
import os
import sys
from datetime import date 
import matplotlib.pyplot as plt
from scipy import special
import math 
import torch

# get um 
# function to get m coefficients
def get_um(u, lat_lim=20, c=51, max_m=20):
    # u has a shape of [batch_size, filters, lat, longitude]
    # do meridional projection on each longitude for each activations
    lat = np.arange(lat_lim,-lat_lim-2,-2)

    beta= 2.28e-11  # variation of coriolis parameter with latitude

    L = np.sqrt(c / beta)  # horizontal scale (m)

    # define y = lat * 110 km / L
    y = lat * 110 * 1000 / L # dimensionless

    # define meridianol wave structures
    phi = []

    for i in np.arange(max_m):
        p = special.hermite(i)
        Hm = p(y)
        phim = np.exp(- y**2 / 2) * Hm / np.sqrt((2**i) * np.sqrt(np.pi) * math.factorial(i))

        if len(u.shape)==4:
            phi.append(np.reshape(phim, (1, 1, len(y), 1)))
        elif len(u.shape)==3:
            phi.append(np.reshape(phim, (1, len(y), 1)))
        else:
            print('wrong input shape!')
            exit()

    # projection coefficients
    if len(u.shape)==4:
        um = np.zeros((u.shape[0], u.shape[1], max_m, u.shape[-1]))

        dy = (lat[0] - lat[1]) * 110 * 1000 / L 

        for i in range(max_m):
            um0 = np.sum(u * phi[i] * dy, axis=2, keepdims=True)  # (time, 1, lon)
            um[:,:,i,None,:] = um0
    elif len(u.shape)==3:
        um = np.zeros((u.shape[0], max_m, u.shape[-1]))

        dy = (lat[0] - lat[1]) * 110 * 1000 / L 

        for i in range(max_m):
            um0 = np.sum(u * phi[i] * dy, axis=1, keepdims=True)
            um[:,i,None,:] = um0

    return um

# load projection 
def projection(vn='olr', c=51, m=1, mflg='off', wnx=1, wnxflg='off', time_range=['1979-01-01','2021-12-31'], lat_range=[90, -90], pic_save='./',dataflg='new'):
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

    olr_re_fft = xr.DataArray(
        np.zeros((len(ds1.time), len(lat_ind), len(ds1.lon)), dtype=np.complex128),
        coords=[ds1['time'], lat[lat_ind], ds1['lon']],
        dims=['time', 'lat', 'lon']
    )

    return olr_re_fft # (time, lat, lon)

# load test data
def load_test_data(vn, Fnmjo,leadmjo,mem_list,testystat,testyend,c,m,mflg,wnx,wnxflg,lat_lim,mjo_ind,pic_save,dataflg='new'):
    # set parameters
    nmem = len(mem_list)  # memory length
    dimx = int(1 + 2 * int(lat_lim / 2))
    dimy = 180

    # make projection and do zonal fft (time, lat, lon)
    olr_re = projection(vn, c, m, mflg, wnx, wnxflg, [str(testystat)+'-01-01', str(testyend-1)+'-12-31'], [lat_lim,-lat_lim],pic_save, dataflg=dataflg)

    ndays = int( ( np.datetime64(str(testyend)+'-01-01') - np.datetime64(str(testystat)+'-01-01') ) / np.timedelta64(1,'D') ) 

    psi_test_input = np.zeros((ndays-mem_list[-1],nmem,dimx,dimy))

    for i in range(ndays-mem_list[-1]):
        psi_test_input[i, :, :, :] = olr_re[i:i+nmem, :, :] * 0

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

    psi_test_input[:,0,0,:2] = psi_test_label

    psi_test_input_Tr = psi_test_input
    psi_test_label_Tr = psi_test_label

    ## convert to torch tensor
    psi_test_input_Tr_torch = torch.from_numpy(psi_test_input_Tr).float()
    psi_test_label_Tr_torch = torch.from_numpy(psi_test_label_Tr).float()

    return psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, ndays-mem_list[-1]

# CNN
def get_net(nmaps=1, lat_lim=20, nmem=1):

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    print('Code starts')

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
    return net

# calculate and store the mk field and the raw field for each layer in the given model
def get_mkfield(**kwargs):
    # give the default values of the kwargs
    leadmjo = kwargs.get('leadmjo', 1)
    vn = kwargs.get('vn', 'olr')
    mjo_ind = kwargs.get('mjo_ind', 'RMM')
    c = kwargs.get('c', 51)
    m = kwargs.get('m', 1)
    mflg = kwargs.get('mflg', 'off')
    wnx = kwargs.get('wnx', 1)
    wnxflg = kwargs.get('wnxflg', 'off')
    nmem = kwargs.get('nmem', 1)
    mem_list = kwargs.get('mem_list', [0])
    Fnmjo = kwargs.get('Fnmjo', '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc')
    testystat = kwargs.get('testystat', 2015)
    testyend = kwargs.get('testyend', 2022)
    zmode = kwargs.get('zmode', 1)
    lat_lim = kwargs.get('lat_lim', 20)
    pic_save = kwargs.get('pic_save', '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/pic_save/')
    model_save = kwargs.get('model_save', '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/modelsave/')
    dataflg = kwargs.get('dataflg', 'new')
    outputflg = kwargs.get('outputflg', '')
    max_m = kwargs.get('max_m', 20)

    # look the (m,k) field at each layer
    # 1. extract the target activations whose size is [batch_size, channels, latitude, longitude] on one layer 
    # 2. get the (m,k) field for the target activations 
    # 3. iterate #1-2 for each layer
    net = get_net(lat_lim=lat_lim, nmem=nmem)

    # each layer's name: 
    layer_name = ['input_map','input_layer','hidden1','hidden2','hidden3','hidden4','hidden5']
    nlayer = len(layer_name)
    mkfields = {}  # name of each layer: its mk fields (dtype: list)
    outfields = {}

    # add hook
    hook_outputs = {}

    for layer in range(nlayer):
        mkfields[layer_name[layer]] = []
        outfields[layer_name[layer]] = []

    # Define hook functions to retrieve the hidden layer outputs for each input
    def hook_fn(module, input, output):
        hook_outputs[module] = output.data.cpu().numpy()

    # Register hooks on desired layers for each input
    net.input_layer.register_forward_hook(hook_fn)
    net.hidden1.register_forward_hook(hook_fn)
    net.hidden2.register_forward_hook(hook_fn)
    net.hidden3.register_forward_hook(hook_fn)
    net.hidden4.register_forward_hook(hook_fn)
    net.hidden5.register_forward_hook(hook_fn)

    net.cuda()  # send the net to GPU
    # net.cuda(device=gpuid)
    # gpuid = os.environ["SLURM_LOCALID"]

    print('Model starts')

    psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr, M  = load_test_data(vn,Fnmjo,leadmjo,mem_list,testystat,testyend,zmode,m,mflg,wnx,wnxflg,lat_lim,mjo_ind,pic_save,dataflg=dataflg)
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
        dataflg = ''
        if outputflg=='glb':
            dataflg = outputflg
    
    fn = model_save+'predicted_MCDO_UNET_'+vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_c'+str(c)+'_nmem'+str(nmem)+dataflg+'.pt'
    print(fn)
    net.load_state_dict(torch.load(fn))
    net.eval()

    batch_size = 20
    testing_data_loader = torch.utils.data.DataLoader(psi_test_input_Tr_torch, batch_size=batch_size, drop_last=True)

    for batch in testing_data_loader:
        batch = batch.cuda()
        tmp = net(batch)

        # hidden_outputs contains activations on all layers {name: activations}
        # ####### input map ##########
        data = batch.data.cpu().numpy()  # [batch_size, channels, latitude, longitude]
        # meridional projection
        outfields['input_map'].append(data)

        # ####### input layer ##########
        data = hook_outputs[net.input_layer]  # [batch_size, channels, latitude, longitude]

        outfields['input_layer'].append(data)

        # ####### hidden1 ##########
        data = hook_outputs[net.hidden1]  # [batch_size, channels, latitude, longitude]

        outfields['hidden1'].append(data)

        # ####### hidden2 ##########
        data = hook_outputs[net.hidden2]  # [batch_size, channels, latitude, longitude]

        outfields['hidden2'].append(data)
        
        # ####### hidden3 ##########
        data = hook_outputs[net.hidden3]  # [batch_size, channels, latitude, longitude]

        outfields['hidden3'].append(data)

        # ####### hidden4 ##########
        data = hook_outputs[net.hidden4]  # [batch_size, channels, latitude, longitude]

        outfields['hidden4'].append(data)

        # ####### hidden5 ##########
        data = hook_outputs[net.hidden5]  # [batch_size, channels, latitude, longitude]
        # meridional projection

        outfields['hidden5'].append(data)

    # The structure of mkfields
    # 7 items ['input_map','input_layer','hidden1','hidden2','hidden3','hidden4','hidden5']
    # each item is a list store activations for the item layer
    # each list has 91 arrays to store activations of 91 batches
    # each array contains activations for one batch

    # store it
    import pickle


    with open('/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_yproj_xfft_4gpus_new/hidden_analysis/outfields'+vn+str(lat_lim)+'deg_'+mjo_ind+'_ERA5_lead'+str(leadmjo)+'_dailyinput_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'_zmode'+str(zmode)+'_nmem'+str(nmem)+dataflg+str(max_m)+'.pickle', 'wb') as file2:
        pickle.dump(outfields, file2)

def get_feature_matrix_weighted(mkfields, layer_name=['input_map','input_layer','hidden1','hidden2','hidden3','hidden4','hidden5'], powerflg=False, timeavg=True): 
    # cluster filters on layers using k-means
    from sklearn.cluster import KMeans
    epsilon = 1e-8
    k_err = []

    # store the maps in each layer as a list [time*filters, m, k]
    feature_matrix = []

    for layer in layer_name:
        if powerflg: 
            tmp = np.asarray(mkfields[layer])**2
        else:
            tmp = np.asarray(mkfields[layer])

        if timeavg:
            if layer == 'input_map':
                tmp1 = np.mean(tmp, axis=(0,1)).squeeze()  # [latitude, longitude]
                tmp2 = tmp1.reshape((1, -1))  # [1, latitude*longitude]
            else:
                tmp1 = np.mean(tmp, axis=(0,1)).squeeze()  # [filters, latitude, longitude]
                tmp2 = tmp1.reshape((tmp1.shape[0],tmp1.shape[1]*tmp1.shape[2]))  # [filters, latitude*longitude]
        else:
            # reshape into [time*filters, latitude*longitude]
            tmp2 = tmp.reshape(-1, tmp.shape[-1]*tmp.shape[-2])  # [time*filters, latitude*longitude]
        
        feature_matrix.append(tmp2)
        
    # concatenate the feature matrix
    feature_matrix1 = np.concatenate(feature_matrix, axis=0)  # [time*filters*layer_num, m*k]
    # normalize the feature matrix
    power = np.sum(feature_matrix1, axis=1, keepdims=True)
    # M_feature = np.min(feature_matrix1, axis=1, keepdims=True)
    # STD_feature = np.max(feature_matrix1, axis=1, keepdims=True) - M_feature
    feature_matrix_norm = feature_matrix1 / power

    return feature_matrix_norm

def get_feature_matrix(mkfields, layer_name=['input_map','input_layer','hidden1','hidden2','hidden3','hidden4','hidden5']): 
    # cluster filters on layers using k-means
    from sklearn.cluster import KMeans
    epsilon = 1e-8
    k_err = []

    # store the maps in each layer as a list [time*filters, m, k]
    feature_matrix = []

    # for layer in layer_name:
    #     tmp = np.asarray(mkfields[layer])  # [batch_num, batch_size, filters, m, k]
    #     # reshape into [time*filters, m, k]
    #     tmp_reshape = np.reshape(tmp, (tmp.shape[0]*tmp.shape[1]*tmp.shape[2], tmp.shape[3], tmp.shape[4]), order='F')
    #     # reshape into [time*filters, m*k]
    #     tmp_reshape1 = np.reshape(tmp_reshape, (tmp_reshape.shape[0], tmp_reshape.shape[1]*tmp_reshape.shape[2]), order='C')
    #     feature_matrix.append(tmp_reshape1)

    for layer in layer_name:
        tmp = np.asarray(mkfields[layer])
        tmp1 = np.mean(tmp, axis=(0,1)).squeeze()
        tmp2 = tmp1.reshape((tmp1.shape[0],-1))  # [filters, latitude*longitude]
        feature_matrix.append(tmp2)
        
    # concatenate the feature matrix
    feature_matrix1 = np.concatenate(feature_matrix, axis=0)  # [time*filters*layer_num, m*k]
    # normalize the feature matrix
    M_feature = np.min(feature_matrix1, axis=1, keepdims=True)
    STD_feature = np.max(feature_matrix1, axis=1, keepdims=True) - M_feature
    feature_matrix_norm = (feature_matrix1 - M_feature) / STD_feature

    return feature_matrix_norm

def get_outfield_matrix(mkfields, layer_name=['input_map','input_layer','hidden1','hidden2','hidden3','hidden4','hidden5']): 
    # cluster filters on layers using k-means
    from sklearn.cluster import KMeans
    epsilon = 1e-8
    k_err = []

    # store the maps in each layer as a list [time*filters, m, k]
    feature_matrix = []

    # for layer in layer_name:
    #     tmp = np.asarray(mkfields[layer])  # [batch_num, batch_size, filters, m, k]
    #     # reshape into [time*filters, m, k]
    #     tmp_reshape = np.reshape(tmp, (tmp.shape[0]*tmp.shape[1]*tmp.shape[2], tmp.shape[3], tmp.shape[4]), order='F')
    #     # reshape into [time*filters, m*k]
    #     tmp_reshape1 = np.reshape(tmp_reshape, (tmp_reshape.shape[0], tmp_reshape.shape[1]*tmp_reshape.shape[2]), order='C')
    #     feature_matrix.append(tmp_reshape1)

    for layer in layer_name:
        tmp = np.asarray(mkfields[layer])
        tmp1 = np.mean(tmp, axis=(0,1)).squeeze()
        tmp2 = tmp1.reshape((tmp1.shape[0],-1))  # [filters, latitude*longitude]
        feature_matrix.append(tmp2)
        
    # concatenate the feature matrix
    feature_matrix1 = np.absolute( np.concatenate(feature_matrix, axis=0) ) # [time*filters*layer_num, m*k]
    # normalize the feature matrix
    M_feature = np.sum(feature_matrix1, axis=1, keepdims=True)
    
    feature_matrix_norm = (feature_matrix1 ) / M_feature

    return feature_matrix_norm

# find the suitable k for given layer(s)
def find_suit_kmeans(mkfields, kmax=30, layer_name=['input_map','input_layer','hidden1','hidden2','hidden3','hidden4','hidden5'], weighted=False):
    # cluster filters on layers using k-means
    from sklearn.cluster import KMeans
    epsilon = 1e-8
    k_err = []

    if weighted:
        feature_matrix_norm = get_feature_matrix_weighted(mkfields, layer_name)
    else:
        feature_matrix_norm = get_feature_matrix(mkfields, layer_name)

    sse = []

    for k in range(1, kmax):
        # create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
        kmeans = KMeans(n_clusters=k, max_iter=100).fit(feature_matrix_norm)
        # get the cluster labels for each data point.
        labels = kmeans.labels_
        # get the cluster centers.
        centers = kmeans.cluster_centers_

        # compute the sum of squared distances from each point to its assigned center.
        err = 0
        for i in range(k):
            # select just the data points in the given cluster.
            points = feature_matrix_norm[labels == i, :]
            # compute the distance from each point to the cluster center.
            distances = np.linalg.norm(points - centers[i, :], axis=1)
            # add up all these distances for the cluster.
            err += np.sum(distances)

        sse.append(err)

    return np.asarray(sse)

def plot_suit_kmeans(k_err, title=None):
    fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    # Plot with logarithmic scale
    axs.plot(range(1, len(np.asarray(k_err).squeeze()) + 1), np.asarray(k_err).squeeze(), 'o-', color='black', alpha= 0.5)
    if title is not None:
        axs.set_title(title)
    axs.set_yscale('log')

    # Create a twin axes for linear scale
    ax2 = axs.twinx()
    ax2.plot(range(1, len(np.asarray(k_err).squeeze()) + 1), np.asarray(k_err).squeeze(), 'o-', color='red', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_yscale('linear')

    fig.supxlabel('k')

    # Place the legend on the right-y axis
    axs.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def plot_kmeans_outfields(mkfields, k_cluster=2, layer_name=['input_map','input_layer','hidden1','hidden2','hidden3','hidden4','hidden5'], kmax=20, title=None, xlab=None, ylab=None, xlim=None, ylim=None):
    # choose k_cluster = 6
    # cluster filters using k-means
    from sklearn.cluster import KMeans
    import matplotlib.ticker as ticker

    # kmax = 10

    feature_matrix_norm = get_outfield_matrix(mkfields, layer_name)

    # create kmeans algorithm
    kmeans = KMeans(n_clusters=k_cluster, n_init=30, max_iter=100).fit(feature_matrix_norm)
    # get the cluster labels for each sample
    labels = kmeans.labels_
    # get the cluster centroids
    centroids = kmeans.cluster_centers_

    epsilon = 1e-8
    x = np.bincount(labels)

    # Visualize the cluster centroids
    
    plt.figure(figsize=(15, 3*k_cluster))

    for i in range(k_cluster):
        plt.rcParams.update({'font.size': 24})
        # Calculate the logarithm of the centroid values using ln (natural logarithm)
        centroid_ln = centroids[i].reshape((21, -1)) + epsilon
        
        plt.subplot(int(k_cluster/2), 2, i+1)
        tmp = centroid_ln
        # tmp1 = np.where(tmp <-0.9, np.nan, tmp)
        # tmp_mask = np.ma.masked_invalid(tmp1)

        im = plt.pcolormesh(tmp, cmap='Blues')

        print(f'log(Cluster {i} Centroid), num {x[i]}')

        if title:
            plt.title(title)
        if xlab:
            plt.xlabel(xlab)
        if ylab:
            plt.ylabel(ylab)
 
        plt.colorbar(im)   

        # if xlim:
        #     plt.xlim(xlim)

        # if ylim:
        #     plt.ylim(ylim)
        # else:
        #     plt.xticks([0, 2, 4, 6, 8, 10]) 
        #     plt.yticks([0, 2, 4, 6, 8, 10]) 

    plt.tight_layout()
    plt.show()


def plot_kmeans(mkfields, k_cluster=2, layer_name=['input_map','input_layer','hidden1','hidden2','hidden3','hidden4','hidden5'], kmax=20, title=None, xlab=None, ylab=None, xlim=None, ylim=None):
    # choose k_cluster = 6
    # cluster filters using k-means
    from sklearn.cluster import KMeans
    import matplotlib.ticker as ticker

    # kmax = 10

    feature_matrix_norm = get_feature_matrix(mkfields, layer_name)

    # create kmeans algorithm
    kmeans = KMeans(n_clusters=k_cluster, n_init=30, max_iter=100).fit(feature_matrix_norm)
    # get the cluster labels for each sample
    labels = kmeans.labels_
    # get the cluster centroids
    centroids = kmeans.cluster_centers_

    epsilon = 1e-8
    x = np.bincount(labels)

    # Visualize the cluster centroids
    
    plt.figure(figsize=(15, 3*k_cluster))

    for i in range(k_cluster):
        plt.rcParams.update({'font.size': 24})
        # Calculate the logarithm of the centroid values using ln (natural logarithm)
        centroid_ln = centroids[i].reshape((21, 91)) + epsilon
        
        plt.subplot(int(k_cluster/2), 2, i+1)
        tmp = np.log10(centroid_ln[:,:kmax])
        tmp1 = np.where(tmp <-0.9, np.nan, tmp)
        tmp_mask = np.ma.masked_invalid(tmp1)

        im = plt.pcolormesh(tmp_mask[:kmax,:kmax], cmap='Blues', vmin=-0.9, vmax=0.1)

        print(f'log(Cluster {i} Centroid), num {x[i]}')

        if title:
            plt.title(title)
        if xlab:
            plt.xlabel(xlab)
        if ylab:
            plt.ylabel(ylab)
 
        plt.colorbar(im)   

        if xlim:
            plt.xlim(xlim)

        if ylim:
            plt.ylim(ylim)
        else:
            plt.xticks([0, 2, 4, 6, 8, 10]) 
            plt.yticks([0, 2, 4, 6, 8, 10]) 

    plt.tight_layout()
    plt.show()

def plot_kmeans_weighted(mkfields, k_cluster=2, layer_name=['input_map','input_layer','hidden1','hidden2','hidden3','hidden4','hidden5'], kmax=20, title=None, xlab=None, ylab=None, xlim=None, ylim=None, powerflg=False, timeavg=True):
    # choose k_cluster = 6
    # cluster filters using k-means
    from sklearn.cluster import KMeans
    import matplotlib.ticker as ticker

    # kmax = 10

    feature_matrix_norm = get_feature_matrix_weighted(mkfields, layer_name, powerflg=powerflg, timeavg=timeavg)

    (km, kk) = np.shape(mkfields[layer_name[0]])[-2:]

    # create kmeans algorithm
    kmeans = KMeans(n_clusters=k_cluster, n_init=30, max_iter=100).fit(feature_matrix_norm)
    # get the cluster labels for each sample
    labels = kmeans.labels_
    # get the cluster centroids
    centroids = kmeans.cluster_centers_

    epsilon = 1e-8
    x = np.bincount(labels)

    # Visualize the cluster centroids
    
    plt.figure(figsize=(15, 3*k_cluster))

    for i in range(k_cluster):
        plt.rcParams.update({'font.size': 24})
        # Calculate the logarithm of the centroid values using ln (natural logarithm)
        centroid_ln = centroids[i].reshape((km, kk)) # + epsilon
        
        plt.subplot(int(k_cluster/2), 2, i+1)
        tmp = np.log10(centroid_ln[:,:kmax])
        thred = int(np.max(tmp) * 10 ) / 10 
        tmp1 = np.where(tmp <thred-1.0, np.nan, tmp)
        tmp_mask = np.ma.masked_invalid(tmp1)

        im = plt.pcolormesh(tmp_mask[:kmax,:kmax], cmap='Blues', vmin=thred-1.0, vmax=thred)

        print(f'log(Cluster {i} Centroid), num {x[i]}, sum {np.sum(centroid_ln)}')

        if title:
            plt.title(title)
        if xlab:
            plt.xlabel(xlab)
        if ylab:
            plt.ylabel(ylab)
 
        plt.colorbar(im, ticks=np.linspace(thred-1.0, thred, 6))   

        if xlim:
            plt.xlim(xlim)

        if ylim:
            plt.ylim(ylim)
        else:
            plt.xticks([0, 2, 4, 6, 8, 10]) 
            plt.yticks([0, 2, 4, 6, 8, 10]) 

    plt.tight_layout()
    plt.show()
