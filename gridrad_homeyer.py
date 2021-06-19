#+
# Name:
#		GRIDRAD Python Module
# Purpose:
#		This module contains three functions for dealing with Gridded NEXRAD WSR-88D Radar
#		(GridRad) data: reading (read_file), filtering (filter), and decluttering (remove_clutter).
# Author and history:
#		Cameron R. Homeyer  2017-07-03.
# Warning:
#		The authors' primary coding language is not Python. This code works, but may not be
#      the most efficient or proper approach. Please suggest improvements by sending an email
#		 to chomeyer@ou.edu.
#-

# GridRad read routine
def read_file(infile):
	
	# Import python libraries
	import sys
	import os
	import numpy as np
	import netCDF4
	
	# Check to see if file exists
	if not os.path.isfile(infile):
		print('File "' + infile + '" does not exist.  Returning -2.')
		return -2
	
	# Check to see if file has size of zero
	if os.stat(infile).st_size == 0:
		print('File "' + infile + '" contains no valid data.  Returning -1.')
		return -1
	
	from netCDF4 import Dataset
	from netCDF4 import Variable
	# Open GridRad netCDF file
	id = Dataset(infile, "r", format="NETCDF4")
	
	# Read global attributes
	Analysis_time           = str(id.getncattr('Analysis_time'          ))
	Analysis_time_window    = str(id.getncattr('Analysis_time_window'   ))
	File_creation_date      = str(id.getncattr('File_creation_date'     ))
	Grid_scheme             = str(id.getncattr('Grid_scheme'            ))
	Algorithm_version       = str(id.getncattr('Algorithm_version'      ))
	Algorithm_description   = str(id.getncattr('Algorithm_description'  ))
	Data_source             = str(id.getncattr('Data_source'            ))
	Data_source_URL         = str(id.getncattr('Data_source_URL'        ))
	NOAA_wct_export_Version = str(id.getncattr('NOAA_wct-export_Version'))
	Authors                 = str(id.getncattr('Authors'                ))
	Project_sponsor         = str(id.getncattr('Project_sponsor'        ))
	Project_name            = str(id.getncattr('Project_name'           ))
	
	# Read list of merged files
	file_list    = (id.variables['files_merged'])[:]
	files_merged = ['']*(id.dimensions['File'].size)
	for i in range(0,id.dimensions['File'].size):
		for j in range(0,id.dimensions['FileRef'].size):
			files_merged[i] += str(file_list[i,j])
	
	# Read longitude dimension
	x = id.variables['Longitude']
	x = {'values'    : x[:],             \
		  'long_name' : str(x.long_name), \
		  'units'     : str(x.units),     \
		  'delta'     : str(x.delta),     \
		  'n'         : len(x[:])}
	
	# Read latitude dimension
	y = id.variables['Latitude']
	y = {'values'    : y[:],             \
		  'long_name' : str(y.long_name), \
		  'units'     : str(y.units),     \
		  'delta'     : str(y.delta),     \
		  'n'         : len(y[:])}
	
	# Read altitude dimension
	z = id.variables['Altitude']
	z = {'values'    : z[:],             \
		  'long_name' : str(z.long_name), \
		  'units'     : str(z.units),     \
		  'delta'     : str(z.delta),     \
		  'n'         : len(z[:])}
	
	# Read observation and echo counts
	nobs  = (id.variables['Nradobs' ])[:]
	necho = (id.variables['Nradecho'])[:]
	index = (id.variables['index'   ])[:]
	
	# Read reflectivity variables	
	Z_H  = id.variables['Reflectivity' ]
	wZ_H = id.variables['wReflectivity']

	# Create arrays to store binned values	
	values    = np.zeros(x['n']*y['n']*z['n'])
	wvalues   = np.zeros(x['n']*y['n']*z['n'])
	values[:] = float('nan')

	# Add values to arrays
	values[index[:]]  =  (Z_H)[:]
	wvalues[index[:]] = (wZ_H)[:]
	
	# Reshape arrays to 3-D GridRad domain
	values  =  values.reshape((z['n'], y['n'] ,x['n']))
	wvalues = wvalues.reshape((z['n'], y['n'] ,x['n']))

	Z_H = {'values'     : values,              \
			 'long_name'  : str(Z_H.long_name),  \
			 'units'      : str(Z_H.units),      \
			 'missing'    : float('nan'),        \
			 'wvalues'    : wvalues,             \
			 'wlong_name' : str(wZ_H.long_name), \
			 'wunits'     : str(wZ_H.units),     \
			 'wmissing'   : wZ_H.missing_value,  \
			 'n'          : values.size}
	
	# Close netCDF4 file
	id.close()
	
	# Return data dictionary	
	return {'name'                    : 'GridRad analysis for ' + Analysis_time, \
			  'x'                       : x, \
			  'y'                       : y, \
			  'z'                       : z, \
			  'Z_H'                     : Z_H, \
			  'nobs'                    : nobs, \
			  'necho'                   : necho, \
			  'file'                    : infile, \
			  'files_merged'            : files_merged, \
			  'Analysis_time'           : Analysis_time, \
			  'Analysis_time_window'    : Analysis_time_window, \
			  'File_creation_date'      : File_creation_date, \
			  'Grid_scheme'             : Grid_scheme, \
			  'Algorithm_version'       : Algorithm_version, \
			  'Algorithm_description'   : Algorithm_description, \
			  'Data_source'             : Data_source, \
			  'Data_source_URL'         : Data_source_URL, \
			  'NOAA_wct_export_Version' : NOAA_wct_export_Version, \
			  'Authors'                 : Authors, \
			  'Project_sponsor'         : Project_sponsor, \
			  'Project_name'            : Project_name}


# GridRad filter routine
def filter(data0):
	
	# Import python libraries
	import sys
	import os
	import numpy as np	
	
	#Extract year from GridRad analysis time string
	year = long((data0['Analysis_time'])[0:4])

	wmin        = 0.1												# Set absolute minimum weight threshold for an observation (dimensionless)
	wthresh     = 1.33 - 1.0*(year < 2009)					# Set default bin weight threshold for filtering by year (dimensionless)
	freq_thresh = 0.6												# Set echo frequency threshold (dimensionless)
	Z_H_thresh  = 18.5											# Reflectivity threshold (dBZ)
	nobs_thresh = 2												# Number of observations threshold
	
	# Extract dimension sizes
	nx = (data0['x'])['n']
	ny = (data0['y'])['n']
	nz = (data0['z'])['n']
	
	echo_frequency = np.zeros((nz,ny,nx))					# Create array to compute frequency of radar obs in grid volume with echo

	ipos = np.where(data0['nobs'] > 0)						# Find bins with obs 
	npos = len(ipos[0])											# Count number of bins with obs

	if (npos > 0):
		echo_frequency[ipos] = (data0['necho'])[ipos]/(data0['nobs'])[ipos]		# Compute echo frequency (number of scans with echo out of total number of scans)

	inan = np.where(np.isnan((data0['Z_H'])['values']))				# Find bins with NaNs 
	nnan = len(inan[0])														# Count number of bins with NaNs
	
	if (nnan > 0): ((data0['Z_H'])['values'])[inan] = 0.0

	# Find observations with low weight
	ifilter = np.where( ((data0['Z_H'])['wvalues'] < wmin       )                                            | \
							 (((data0['Z_H'])['wvalues'] < wthresh    ) & ((data0['Z_H'])['values'] <= Z_H_thresh)) |
							  ((echo_frequency           < freq_thresh) &  (data0['nobs'] > nobs_thresh)))
	
	nfilter = len(ifilter[0])									# Count number of bins that need to be removed
	
	# Remove low confidence observations
	if (nfilter > 0): ((data0['Z_H'])['values'])[ifilter] = float('nan')
	
	# Replace NaNs that were previously removed
	if (nnan > 0): ((data0['Z_H'])['values'])[inan] = float('nan')
	
	# Return filtered data0
	return data0

	
def remove_clutter(data0, **kwargs):

	# Set defaults for optional parameters
	if ('skip_weak_ll_echo' not in kwargs): skip_weak_ll_echo = 0
	
	# Import python libraries
	import sys
	import os
	import numpy as np	
	
	# Set fractional areal coverage threshold for speckle identification
	areal_coverage_thresh = 0.32
	
	# Extract dimension sizes
	nx = (data0['x'])['n']
	ny = (data0['y'])['n']
	nz = (data0['z'])['n']
	
	# Copy altitude array to 3 dimensions
	zzz = ((((data0['z'])['values']).reshape(nz,1,1)).repeat(ny, axis = 1)).repeat(nx, axis = 2)

	# First pass at removing speckles
	fin = np.isfinite((data0['Z_H'])['values'])
	
	# Compute fraction of neighboring points with echo
	cover = np.zeros((nz,ny,nx))
	for i in range(-2,3):
		for j in range(-2,3):
			cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
	cover = cover/25.0
	
	# Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
	ibad = np.where(cover <= areal_coverage_thresh)
	nbad = len(ibad[0])
	if (nbad > 0): ((data0['Z_H'])['values'])[ibad] = float('nan')

	# Attempts to mitigate ground clutter and biological scatterers
	if (skip_weak_ll_echo == 0):
		# First check for weak, low-level echo
		inan = np.where(np.isnan((data0['Z_H'])['values']))				# Find bins with NaNs 
		nnan = len(inan[0])															# Count number of bins with NaNs
	
		if (nnan > 0): ((data0['Z_H'])['values'])[inan] = 0.0

		# Find weak low-level echo and remove (set to NaN)
		ibad = np.where(((data0['Z_H'])['values'] < 10.0) & (zzz <= 4.0))
		nbad = len(ibad[0])
		if (nbad > 0): ((data0['Z_H'])['values'])[ibad] = float('nan')
		
		# Replace NaNs that were removed
		if (nnan > 0): ((data0['Z_H'])['values'])[inan] = float('nan')

		# Second check for weak, low-level echo
		inan = np.where(np.isnan((data0['Z_H'])['values']))				# Find bins with NaNs 
		nnan = len(inan[0])															# Count number of bins with NaNs
	
		if (nnan > 0): ((data0['Z_H'])['values'])[inan] = 0.0

		refl_max   = np.nanmax( (data0['Z_H'])['values'],             axis=0)
		echo0_max  = np.nanmax(((data0['Z_H'])['values'] >  0.0)*zzz, axis=0)
		echo0_min  = np.nanmin(((data0['Z_H'])['values'] >  0.0)*zzz, axis=0)
		echo5_max  = np.nanmax(((data0['Z_H'])['values'] >  5.0)*zzz, axis=0)
		echo15_max = np.nanmax(((data0['Z_H'])['values'] > 15.0)*zzz, axis=0)

		# Replace NaNs that were removed
		if (nnan > 0): ((data0['Z_H'])['values'])[inan] = float('nan')
		
		# Find weak and/or shallow echo
		ibad = np.where(((refl_max   <  20.0) & (echo0_max  <= 4.0) & (echo0_min  <= 3.0)) | \
							 ((refl_max   <  10.0) & (echo0_max  <= 5.0) & (echo0_min  <= 3.0)) | \
							 ((echo5_max  <=  5.0) & (echo5_max  >  0.0) & (echo15_max <= 3.0)) | \
							 ((echo15_max <   2.0) & (echo15_max >  0.0)))
		nbad = len(ibad[0])
		if (nbad > 0):
			kbad = (np.zeros((nbad))).astype(int)
			for k in range(0,nz):
				((data0['Z_H'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')


	# Find clutter below convective anvils
	k4km = ((np.where((data0['z'])['values'] >= 4.0))[0])[0]
	fin  = np.isfinite((data0['Z_H'])['values'])
	ibad = np.where((          fin[k4km         ,:,:]          == 0) & \
							 (np.sum(fin[k4km:(nz  -1),:,:], axis=0) >  0) & \
							 (np.sum(fin[   0:(k4km-1),:,:], axis=0) >  0))
	nbad = len(ibad[0])
	if (nbad > 0):
		kbad = (np.zeros((nbad))).astype(int)
		for k in range(0,k4km+1):
			((data0['Z_H'])['values'])[(k+kbad),ibad[0],ibad[1]] = float('nan')
	
	# Second pass at removing speckles
	fin = np.isfinite((data0['Z_H'])['values'])
	
	# Compute fraction of neighboring points with echo
	cover = np.zeros((nz,ny,nx))
	for i in range(-2,3):
		for j in range(-2,3):
			cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
	cover = cover/25.0
			
	# Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
	ibad = np.where(cover <= areal_coverage_thresh)
	nbad = len(ibad[0])
	if (nbad > 0): ((data0['Z_H'])['values'])[ibad] = float('nan')
	
	return data0
