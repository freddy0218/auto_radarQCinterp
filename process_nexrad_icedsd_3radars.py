import pyart 
import numpy as np
import warnings
from siphon.simplewebservice.iastate import IAStateUpperAir
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=RuntimeWarning)
warnings.filterwarnings(action='ignore',category=UserWarning)
from datetime import datetime
import glob,os
import auto_filter,snr_noise,attenuation,calc_kdp,tools,retrieval,echo_classification
from csu_radartools import csu_liquid_ice_mass
import faulthandler

faulthandler.enable()
# Setting relevant paths
#-------------------------------------------------------------------------------------------------
input_volpath = '/scra6/ft21894/radartrunk/0617/0200-0400/'
input_volpath2 = '/scra6/ft21894/radartrunk/0617/0200-0400/'
input_volpath3 = '/scra6/ft21894/radartrunk/0617/0200-0400/'
radarvollist = sorted(glob.glob(input_volpath+'KUEX20150617*'))
radarvollist2 = sorted(glob.glob(input_volpath2+'KGLD20150617*'))
radarvollist3 = sorted(glob.glob(input_volpath3+'KLNX20150617*'))
output_gridpath = '/scra6/ft21894/radartrunk/0617/'

# Read in radar files and save data time
#-------------------------------------------------------------------------------------------------
for i in range(len(radarvollist)):
	print(i)
	base = pyart.io.read_nexrad_archive(radarvollist[i])
	base2 = pyart.io.read_nexrad_archive(radarvollist2[i])
	base3 = pyart.io.read_nexrad_archive(radarvollist3[i])
	dt = datetime.strptime(base.time['units'][-20:], '%Y-%m-%dT%H:%M:%SZ')

	# Read in sounding and interpolate onto radar gates
	#-------------------------------------------------------------------------------------------------
	date_name = datetime(2015,6,17,0)
	station_name = 'KLBF' #KOAX
	df0620 = IAStateUpperAir.request_data(date_name,station_name)
	T0620,z0620 = tools.interpolate_sounding_to_radar(df0620.dropna(),base)
	T0620b,z0620b = tools.interpolate_sounding_to_radar(df0620.dropna(),base2)
	T0620c,z0620c = tools.interpolate_sounding_to_radar(df0620.dropna(),base3)
	base = tools.add_field_to_radar_object(T0620,base,'temp_radiosonde','degree','Temperature','Temperature','reflectivity')
	base2 = tools.add_field_to_radar_object(T0620b,base2,'temp_radiosonde','degree','Temperature','Temperature','reflectivity')
	base3 = tools.add_field_to_radar_object(T0620c,base3,'temp_radiosonde','degree','Temperature','Temperature','reflectivity')
	base = tools.add_field_to_radar_object(z0620,base,'z_radiosonde','m','gate_alt','gate_alt','reflectivity')
	base2 = tools.add_field_to_radar_object(z0620b,base2,'z_radiosonde','m','gate_alt','gate_alt','reflectivity')
	base3 = tools.add_field_to_radar_object(z0620c,base3,'z_radiosonde','m','gate_alt','gate_alt','reflectivity')

	# Retrieve SNR and filter noise
	#-------------------------------------------------------------------------------------------------
	snr = pyart.retrieve.calculate_snr_from_reflectivity(base,'reflectivity','SNR')
	snr2 = pyart.retrieve.calculate_snr_from_reflectivity(base2,'reflectivity','SNR')
	snr3 = pyart.retrieve.calculate_snr_from_reflectivity(base3,'reflectivity','SNR')
	base = tools.add_field_to_radar_object(snr['data'], base, 'SNR','','SNR','SNR','reflectivity')
	base2 = tools.add_field_to_radar_object(snr2['data'], base2, 'SNR','','SNR','SNR','reflectivity')
	base3 = tools.add_field_to_radar_object(snr3['data'], base3, 'SNR','','SNR','SNR','reflectivity')

	zdr_corr = snr_noise.correct_zdr(base, zdr_name="differential_reflectivity", snr_name="SNR")
	zdr_corr2 = snr_noise.correct_zdr(base2, zdr_name="differential_reflectivity", snr_name="SNR")
	zdr_corr3 = snr_noise.correct_zdr(base3, zdr_name="differential_reflectivity", snr_name="SNR")
	base = tools.add_field_to_radar_object(zdr_corr,base,'zdr_corr','dB',
										'ZDR_nonoise','ZDR_nonoise','reflectivity')
	base2 = tools.add_field_to_radar_object(zdr_corr2,base2,'zdr_corr','dB',
										 'ZDR_nonoise','ZDR_nonoise','reflectivity')
	base3 = tools.add_field_to_radar_object(zdr_corr3,base3,'zdr_corr','dB','ZDR_nonoise','ZDR_nonoise','reflectivity')
	# Simple Gatefilter
	#-------------------------------------------------------------------------------------------------
	gatefilter = pyart.filters.GateFilter(base)
	gatefilter.exclude_below('cross_correlation_ratio',0.75)
	gatefilter2=pyart.filters.GateFilter(base2)
	gatefilter2.exclude_below('cross_correlation_ratio', 0.75)
	gatefilter3=pyart.filters.GateFilter(base3)
	gatefilter3.exclude_below('cross_correlation_ratio', 0.75)

	# Unfold phidp and calculate Kdp with csu_radartools
	#-------------------------------------------------------------------------------------------------
	unfphidic = pyart.correct.dealias_unwrap_phase(base,skip_checks=True,\
												vel_field='differential_phase',nyquist_vel=90)
	unfphidic2 = pyart.correct.dealias_unwrap_phase(base2,skip_checks=True,\
												 vel_field='differential_phase',nyquist_vel=90)
	unfphidic3 = pyart.correct.dealias_unwrap_phase(base3,skip_checks=True,vel_field='differential_phase',nyquist_vel=90)
	
	#recalculate phidp
	base,kdp,fdp,sdp = calc_kdp.kdp_bringi(base,gatefilter,raw_phidp_name="differential_phase",refl_name="reflectivity")
	base2,kdp2,fdp2,sdp2 = calc_kdp.kdp_bringi(base2,gatefilter2,raw_phidp_name="differential_phase",refl_name="reflectivity")
	base3,kdp3,fdp3,sdp3 = calc_kdp.kdp_bringi(base3,gatefilter3,raw_phidp_name="differential_phase",refl_name="reflectivity")
	
	# Create Fake NCP Field
	#-------------------------------------------------------------------------------------------------
	try:
		base.fields['NCP']
		base2.fields['NCP']
		base3.fields['NCP']
	except KeyError:
		ncp = pyart.config.get_metadata('normalized_coherent_power')
		ncp['data'] = np.ones_like(base.fields['reflectivity']['data'])
		ncp['description'] = "THIS FIELD IS FAKE. SHOULD BE REMOVED!"
		ncp2 = pyart.config.get_metadata('normalized_coherent_power')
		ncp2['data'] = np.ones_like(base2.fields['reflectivity']['data'])
		ncp3 = pyart.config.get_metadata('normalized_coherent_power')
		ncp3['data'] = np.ones_like(base3.fields['reflectivity']['data'])
		base.add_field('NCP', ncp)
		base2.add_field('NCP',ncp2)
		base3.add_field('NCP',ncp3)

	# Correct reflectivity and Zdr attenuation
	#-------------------------------------------------------------------------------------------------
	#ZDR attenuation correction
	base = attenuation.correct_attenuation_zdr(base,gatefilter=gatefilter,zdr_name='zdr_corr',phidp_name='KDP',\
			refl_field="reflectivity",alpha=0.016)
	base2 = attenuation.correct_attenuation_zdr(base2,gatefilter=gatefilter2,zdr_name='zdr_corr',phidp_name='KDP',\
			refl_field="reflectivity",alpha=0.016)
	base3 = attenuation.correct_attenuation_zdr(base3,gatefilter=gatefilter3,zdr_name='zdr_corr',phidp_name='KDP',\
			refl_field="reflectivity",alpha=0.016)

	# Retrieve Ice/Liquid Mass
	#-------------------------------------------------------------------------------------------------
	dz = tools.extract_unmasked_data(base,'reflectivity')
	dr = tools.extract_unmasked_data(base,'zdr_acorr')
	dz2 = tools.extract_unmasked_data(base2,'reflectivity')
	dr2 = tools.extract_unmasked_data(base2,'zdr_acorr')
	dz3 = tools.extract_unmasked_data(base3,'reflectivity')
	dr3 = tools.extract_unmasked_data(base3,'zdr_acorr')

	mw, mi = csu_liquid_ice_mass.calc_liquid_ice_mass(dz,dr,z0620/1000.0,T=T0620)
	mw2, mi2 = csu_liquid_ice_mass.calc_liquid_ice_mass(dz2,dr2,z0620b/1000.0,T=T0620b)
	mw3, mi3 = csu_liquid_ice_mass.calc_liquid_ice_mass(dz3,dr3,z0620c/1000.0,T=T0620c)
	base = tools.add_field_to_radar_object(mw,base,'MW','g m-3',\
			'Liquid Water Mass','Liquid Water Mass','reflectivity')
	base = tools.add_field_to_radar_object(mi,base,'MI','g m-3',\
			'Ice Water Mass','Ice Water Mass','reflectivity')
	base2 = tools.add_field_to_radar_object(mw2,base2,'MW','g m-3','Liquid Water Mass','Liquid Water Mass','reflectivity')
	base2 = tools.add_field_to_radar_object(mi2,base2,'MI','g m-3','Ice Water Mass','Ice Water Mass','reflectivity')
	base3 = tools.add_field_to_radar_object(mw3,base3,'MW','g m-3','Liquid Water Mass','Liquid Water Mass','reflectivity')
	base3 = tools.add_field_to_radar_object(mi3,base3,'MI','g m-3','Ice Water Mass','Ice Water Mass','reflectivity')

	#Filter insect and ground clutter with csu_radartools and wradlib
	#-------------------------------------------------------------------------------------------------
#	base, gf_noinsect, gf_gclutter = auto_filter.insect_clutter_filter(base,refl_field="reflectivity",\
#																	zdr_field="zdr_corr")
#	base2, gf_noinsect2, gf_gclutter2 = auto_filter.insect_clutter_filter(base2,refl_field="reflectivity",\
#																	zdr_field="zdr_corr")
	
	# Retrieve DSD, rainrate, Ice DSD
	#-------------------------------------------------------------------------------------------------
	base = retrieval.retrieve_dsd(base,gatefilter=gatefilter,kdp_field='KDP',\
			zdr_field='zdr_acorr',refl_field='reflectivity')
	base = retrieval.retrieve_rainrate(base,gatefilter=gatefilter,kdp_field='KDP',zdr_field='zdr_acorr',\
			refl_field='reflectivity',hid_field=None,method='CSU_KDP')
	base2 = retrieval.retrieve_dsd(base2,gatefilter=gatefilter2,kdp_field='KDP',\
			zdr_field='zdr_acorr',refl_field='reflectivity')
	base2 = retrieval.retrieve_rainrate(base2,gatefilter=gatefilter2,kdp_field='KDP',zdr_field='zdr_acorr',\
			refl_field='reflectivity',hid_field=None,method='CSU_KDP')
	base3 = retrieval.retrieve_dsd(base3,gatefilter=gatefilter3,kdp_field='KDP',zdr_field='zdr_acorr',refl_field='reflectivity')
	base3 = retrieval.retrieve_rainrate(base3,gatefilter=gatefilter3,kdp_field='KDP',zdr_field='zdr_acorr',\
			refl_field='reflectivity',hid_field=None,method='CSU_KDP')
	base = retrieval.retrieve_icedsd_ryzhkov18(base,gatefilter=gatefilter,zdp_field='ZDP',kdp_field='KDP',\
			zdr_field='zdr_acorr',refl_field='reflectivity',temp_field='temp_radiosonde',band='S')
	base2 = retrieval.retrieve_icedsd_ryzhkov18(base2,gatefilter=gatefilter2,zdp_field='ZDP',kdp_field='KDP',\
			zdr_field='zdr_acorr',refl_field='reflectivity',temp_field='temp_radiosonde',band='S')
	base3 = retrieval.retrieve_icedsd_ryzhkov18(base3,gatefilter=gatefilter3,zdp_field='ZDP',kdp_field='KDP',\
			zdr_field='zdr_acorr',refl_field='reflectivity',temp_field='temp_radiosonde',band='S')

	gatefilter.exclude_above('zdr_acorr', 6)
	gatefilter.exclude_above('D0', 4.25)
	gatefilter = pyart.correct.despeckle.despeckle_field(base, 'D0', gatefilter=gatefilter)
	gatefilter = pyart.correct.despeckle.despeckle_field(base, 'zdr_acorr', gatefilter=gatefilter)
	gatefilter2.exclude_above('zdr_acorr', 6)
	gatefilter2.exclude_above('D0', 4.25)
	gatefilter2 = pyart.correct.despeckle.despeckle_field(base2, 'D0', gatefilter=gatefilter2)
	gatefilter2 = pyart.correct.despeckle.despeckle_field(base2, 'zdr_acorr', gatefilter=gatefilter2)
	gatefilter2.exclude_above('zdr_acorr', 6)
	gatefilter2.exclude_above('D0', 4.25)
	gatefilter3 = pyart.correct.despeckle.despeckle_field(base3, 'D0', gatefilter=gatefilter3)
	gatefilter3 = pyart.correct.despeckle.despeckle_field(base3, 'zdr_acorr', gatefilter=gatefilter3)

        
	#grid
	# 1st batch: -200,100; -150,150; 2nd batch -150,200,-200,-200
	# 0620; 0300-0430: -50,250;-150,200
	grid_shape  = (41, 501, 551) 
	grid_limits = ((0, 20000), (-200000.0, 300000.0), (-350000.0, 200000.0))
	grid_roi    = 2000
	grid_lat0 = (base.latitude['data'][0]+base2.latitude['data'][0])/2
	grid_lon0 = (base.longitude['data'][0]+base2.longitude['data'][0])/2

	###########################################################
	# Gridded Processing and Output
	###########################################################
	out_fn  = '_'.join([base.metadata['instrument_name'],base2.metadata['instrument_name'],base3.metadata['instrument_name'],
		dt.strftime('%Y%m%d_%H%M%S'),'grid_mass_comp_v2']) + '.nc'
	out_ffn = ''.join([output_gridpath,out_fn])
	
	#genreate grid object for cts fields
	grid = pyart.map.grid_from_radars((base,base2,base3),grid_shape = grid_shape,grid_limits = grid_limits,\
			grid_origin=(base.latitude['data'][0],base.longitude['data'][0]),gatefilters=(gatefilter,gatefilter2,gatefilter3),\
			weighting_function = 'BARNES2',gridding_algo = 'map_gates_to_grid',roi_func='constant', constant_roi = grid_roi,\
			fields=['reflectivity', 'zdr_acorr', 'KDP','cross_correlation_ratio','Dm_ice','Nt_ice',\
					'D0','Nw','mu','MI','MW','rain_blend', 'ZDP','FI','temp_radiosonde','z_radiosonde','spectrum_width', 'SNR'])
	pyart.io.write_grid(out_ffn, grid, format='NETCDF4', write_proj_coord_sys=True,proj_coord_sys=None,\
			arm_time_variables=False, arm_alt_lat_lon_variables=True,\
			write_point_x_y_z=True, write_point_lon_lat_alt=True)

	del grid,base,base2,base3,snr,snr2,snr3,zdr_corr,zdr_corr2,zdr_corr3,\
			unfphidic,unfphidic2,unfphidic3,dz,dz2,dz3,dr,dr2,dr3,mw,mw2,mw3,mi,mi2,mi3,kdp,fdp,sdp,kdp2,fdp2,sdp2,kdp3,fdp3,sdp3

