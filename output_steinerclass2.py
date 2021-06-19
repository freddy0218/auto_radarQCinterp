import pyart 
import numpy as np
import warnings
from numpy import dtype
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=RuntimeWarning)
warnings.filterwarnings(action='ignore',category=UserWarning)
from datetime import datetime
from netCDF4 import Dataset
import glob, os

# Setting relevant paths
#-------------------------------------------------------------------------------------------------
input_gridpath = '/scra6/ft21894/radartrunk/0701/grid/'
radargridlist = sorted(glob.glob(input_gridpath+'*.nc'))
output_gridpath = '/scra6/ft21894/radartrunk/0701/grid/'
# Read in grid files and save data time
#-------------------------------------------------------------------------------------------------
base_f = pyart.io.read_grid(radargridlist[0]) #8
base_l = pyart.io.read_grid(radargridlist[-1])
dt_f = datetime.strptime(base_f.time['units'][-20:], '%Y-%m-%dT%H:%M:%SZ')
dt_l = datetime.strptime(base_l.time['units'][-20:], '%Y-%m-%dT%H:%M:%SZ')
del base_l
# Convective-Stratiform Partition (cite: Steiner et al. 1995)
# ------------------------------------------------------------------------------------------------
data_example = base_f.fields['reflectivity']['data']
classification_storage = np.zeros((len(radargridlist),data_example.shape[1],
								   data_example.shape[2]))
for i in range(len(radargridlist)):
	print(i)
	base = pyart.io.read_grid(radargridlist[i])
	x,y = base.x['data'], base.y['data']
	dx = np.abs(x[1]-x[0])
	dy = np.abs(y[1]-y[0])
	eclass = pyart.retrieve.steiner_conv_strat(base, dx=dx, dy=dy, 
											work_level=2000, refl_field='reflectivity')
	classification_storage[i,:,:] = eclass['data'][:]
	del base

# open a netCDF file to write
out_fn  = '_'.join([base_f.metadata['instrument_name'],dt_f.strftime('%Y%m%d%H%M%S'),
					dt_l.strftime('%Y%m%d%H%M%S'),'convstrat']) + '.nc'
out_ffn = ''.join([output_gridpath,out_fn])

ncout = Dataset(out_ffn, 'w', format='NETCDF4')
# define axis size
ncout.createDimension('time', None)  # unlimited
ncout.createDimension('lat',data_example.shape[1])
ncout.createDimension('lon',data_example.shape[2])
	
# create variable array
steiner_eclass = ncout.createVariable('steiner_eclass', dtype('double').char, ('time', 'lat', 'lon'))
	
# copy axis from original dataset
steiner_eclass[:] = classification_storage[:]
# close files
ncout.close()


