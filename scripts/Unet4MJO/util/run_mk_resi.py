import hidden_util as hid 
# salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

vn = 'olr'
lat_lim = 90
mjo_ind = 'RMM' 
leadmjo = 10
m = 5
mflg = 'resi' 
wnx = 15
wnxflg = 'resi'
zmode = 1 
nmem = 1
max_m = 5

if mjo_ind == 'RMM':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
else:
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'
    
model_save = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_globalmap/modelsave/'
dataflg = 'new'
outputflg = 'glb'
# calculate the m-k field for a given model
hid.get_mkfield(vn=vn, lat_lim=lat_lim, mjo_ind=mjo_ind, leadmjo=leadmjo,max_m=max_m, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, zmode=zmode, nmem=nmem, Fnmjo=Fnmjo, dataflg=dataflg, outputflg=outputflg, model_save=model_save)

vn = 'olr'
lat_lim = 90
mjo_ind = 'ROMI' 
leadmjo = 10
m = 5
mflg = 'resi' 
wnx = 15
wnxflg = 'resi'
zmode = 1 
nmem = 1
max_m = 5

if mjo_ind == 'RMM':
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
else:
    Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'
    
model_save = '/pscratch/sd/l/linyaoly/ERA5/Unet4MJO/1map_MCDO_ERA5_globalmap/modelsave/'
dataflg = 'new'
outputflg = 'glb'
# calculate the m-k field for a given model
hid.get_mkfield(vn=vn, lat_lim=lat_lim, mjo_ind=mjo_ind, leadmjo=leadmjo,max_m=max_m, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, zmode=zmode, nmem=nmem, Fnmjo=Fnmjo, dataflg=dataflg, outputflg=outputflg, model_save=model_save)


# print('rmm')

# vn = 'olr'
# lat_lim = 20
# mjo_ind = 'ROMI' 
# leadmjo = 25
# m = 1
# mflg = 'off' 
# wnx = 1
# wnxflg = 'off'
# zmode = 1
# nmem = 1
# Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'
# dataflg = 'new'
# # calculate the m-k field for a given model
# hid.get_mkfield(vn=vn, lat_lim=lat_lim, mjo_ind=mjo_ind, leadmjo=leadmjo, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, zmode=zmode, nmem=nmem, Fnmjo=Fnmjo, dataflg=dataflg)

# vn = 'olr'
# lat_lim = 20
# mjo_ind = 'RMM' 
# leadmjo = 10
# m = 10
# mflg = 'resi' 
# wnx = 9
# wnxflg = 'resi'
# zmode = 1 
# nmem = 1
# Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/RMM_ERA5_daily_1979to2012.nc'
# dataflg = 'new'
# # calculate the m-k field for a given model
# hid.get_mkfield(vn=vn, lat_lim=lat_lim, mjo_ind=mjo_ind, leadmjo=leadmjo, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, zmode=zmode, nmem=nmem, Fnmjo=Fnmjo, dataflg=dataflg)

# print('rmm')

# vn = 'olr'
# lat_lim = 20
# mjo_ind = 'ROMI' 
# leadmjo = 15
# m = 10
# mflg = 'resi' 
# wnx = 9
# wnxflg = 'resi'
# zmode = 1
# nmem = 1
# Fnmjo = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/full/ROMI_ERA5_daily_1979to2014.nc'
# dataflg = 'new'
# # calculate the m-k field for a given model
# hid.get_mkfield(vn=vn, lat_lim=lat_lim, mjo_ind=mjo_ind, leadmjo=leadmjo, m=m, mflg=mflg, wnx=wnx, wnxflg=wnxflg, zmode=zmode, nmem=nmem, Fnmjo=Fnmjo, dataflg=dataflg)

