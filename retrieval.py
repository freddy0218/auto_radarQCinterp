import pyart
import numpy as np
from csu_radartools import csu_dsd, csu_blended_rain
import tools

def retrieve_dsd(radar,gatefilter,kdp_field=None,zdr_field=None,refl_field=None,
				 band="S"):
	"""
	Ref: Bringi et al. (2004; JTECH), Bringi et al. (2013) - S band retrieval
	Bringi et al. (2009; JTECH) - C band retrieval
	"""
	dbz = tools.extract_unmasked_data(radar, refl_field, bad=-9999)
	zdr = tools.extract_unmasked_data(radar, zdr_field, bad=-9999)
	kdp = tools.extract_unmasked_data(radar, kdp_field, bad=-9999)
	
	d0, Nw, mu = csu_dsd.calc_dsd(dz=dbz, zdr=zdr, kdp=kdp, band=band)
	Nw = np.log10(Nw)
	radar = tools.add_field_to_radar_object(d0, radar, field_name='D0', units='mm', 
										 long_name='Median Volume Diameter',
										 standard_name='Median Volume Diameter', 
										 dz_field=refl_field)
	radar = tools.add_field_to_radar_object(Nw, radar, field_name='Nw', units='', 
										 long_name='Normalized Intercept Parameter',
										 standard_name='Normalized Intercept Parameter', 
										 dz_field=refl_field)
	radar = tools.add_field_to_radar_object(mu, radar, field_name='mu', units='', 
										 long_name = 'Mu',
										 dz_field=refl_field)
	return radar

def retrieve_rainrate(radar,gatefilter,kdp_field=None,zdr_field=None,refl_field=None,
					  hid_field=None,method='CSU_KDP'):
	if method=='CSU_KDP':
		dbz = tools.extract_unmasked_data(radar, refl_field, bad=-9999)
		zdr = tools.extract_unmasked_data(radar, zdr_field, bad=-9999)
		kdp = tools.extract_unmasked_data(radar, kdp_field, bad=-9999)
		rain, method, zdp, fi = csu_blended_rain.calc_blended_rain(dz=dbz,zdr=zdr,
															 kdp=kdp,ice_flag=True)
		radar = tools.add_field_to_radar_object(rain, radar, field_name='rain_blend', units='mm h-1',
										  long_name='Blended Rainfall Rate', 
										  standard_name='Rainfall Rate',
										  dz_field=refl_field)
		radar = tools.add_field_to_radar_object(method, radar, 
										  field_name='method_blend', units='',
										  long_name='Blended Rainfall Method', 
										  standard_name='Rainfall Method',
										  dz_field=refl_field)
		radar = tools.add_field_to_radar_object(zdp, radar, field_name='ZDP', units='dB',
										  long_name='Difference Reflectivity',
										  standard_name='Difference Reflectivity',
										  dz_field=refl_field)
		radar = tools.add_field_to_radar_object(fi, radar, field_name='FI', units='', 
										  long_name='Ice Fraction',
										  standard_name='Ice Fraction',
										  dz_field=refl_field)
		return radar	

def retrieve_icedsd_ryzhkov18(radar,gatefilter,zdp_field=None,kdp_field=None,zdr_field=None,
							  refl_field=None,temp_field=None,band='S'):
	"""
	Ref: Ryzhkov et al. (2018), Ryzhkov and Zrnic (2019), Murphy et al. (2020)
	"""
	from copy import deepcopy
	zdp = deepcopy(radar.fields[zdp_field]['data'][:])
	kdp = deepcopy(radar.fields[kdp_field]['data'][:])
	refl= deepcopy(radar.fields[refl_field]['data'][:])
	zdr = deepcopy(radar.fields[zdr_field]['data'][:])
	temp = deepcopy(radar.fields[temp_field]['data'][:])
	
	# Filter data gates (Murphy et al. 2020)
	condition = np.logical_or(np.logical_or(np.logical_or(refl<0,zdr<0.1),kdp<0.01),temp>0)
	masked_zdp = np.ma.masked_where(condition,zdp)
	masked_kdp = np.ma.masked_where(condition,kdp)
	masked_refl = np.ma.masked_where(condition,refl)
	
	if band=='S':
		wavelength=10
		Dm_ice = -0.1+2*(masked_zdp/(masked_kdp*wavelength*10))**(0.5)
		Nt_ice = 0.1*masked_refl-2*np.log10(masked_zdp/(masked_kdp*wavelength*10))-1.11
	
	radar = tools.add_field_to_radar_object(Dm_ice, radar, field_name='Dm_ice', 
										 units='mm',
										 long_name='Ice Medium Volume Diameter', 
										 standard_name='Ice Medium Volume Diameter',
										 dz_field=refl_field)
	radar = tools.add_field_to_radar_object(Nt_ice, radar, field_name='Nt_ice', 
										 units='mm',
										 long_name='Total Ice Number Concentration', 
										 standard_name='Total Ice Number Concentration',
										 dz_field=refl_field)
	return radar

