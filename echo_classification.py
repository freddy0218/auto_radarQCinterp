"""
Perform echo classification
(a) Convective-Stratiform partition (Ref: Steiner et al. 1995)

"""
import pyart
import numpy as np
import wradlib as wrb
import tools
import auto_filter
from csu_radartools import csu_fhc

#def steiner_convstrat(grid, alt=2500., refl_name='DBZ'):
#    """
#    
#    Ref: Steiner et al. (1995)
#    """
    
#    # Extract coordinates
#    x,y = radar.gate_x['data'], radar.gate_y['data']
#    dx = np.abs(x[1]-x[0])
#    dy = np.abs(y[1]-y[0])
    
    # Perform convective-stratiform partition with Py-ART
#    constrat_class = pyart.retrieve.steiner_conv_strat()

def thurai_convstrat(radar,refl_field=None,rain_field=None,d0_field=None,Nw_field=None):
    """
    Convective-stratiform partition based on radar-retrieved DSD and rain rate by 
    Thurai et al. (2010; J. Atmos. Ocean. Tech.). User should only use this method
    after running retrieve_dsd and retrieve_rainrate

    Returns
    -------
    class_meta: dict
    1: Stratiform, 2: Convective, 3: Mixed, 0: No Rain
    """
    
    # Extracting data.
    d0 = radar.fields[d0_field]['data'].copy()
    nw = radar.fields[Nw_field]['data'].copy()
    dbz = radar.fields[refl_field]['data'].copy()

    classification = np.zeros(dbz.shape, dtype=np.int16)

    # Invalid data
    pos0 = (d0 >= -5) & (d0 <= 100)
    pos1 = (nw >= -10) & (nw <= 100)

    # Classification index.
    indexa = nw - 6.4 + 1.7 * d0

    # Classifying
    classification[(indexa > 0.1) & (dbz > 20)] = 2
    classification[(indexa > 0.1) & (dbz <= 20)] = 1
    classification[indexa < -0.1] = 1
    classification[(indexa >= -0.1) & (indexa <= 0.1)] = 3

    # Masking invalid data.
    classification = np.ma.masked_where(~pos0 | ~pos1 | dbz.mask, classification)

    # Generate metada.
    class_meta = {
        "data": classification,
        "long_name": "thurai_echo_classification",
        "valid_min": 0,
        "valid_max": 3,
        "comment_1": "Convective-stratiform echo classification based on Merhala Thurai",
        "comment_2": "0 = Undefined, 1 = Stratiform, 2 = Convective, 3 = Mixed",
    }
    
    return class_meta

def csu_pid(radar,gatefilter,refl_field='reflectivity',zdr_field='zdr_acorr',
            rhohv_field='RHOHV',kdp_field='KDP',T_field=None,use_temp=True):
    zdr_c = auto_filter.use_csu_wradlib_filters(radar,field_name=zdr_field,
                                            filter_array=gatefilter,TYPE='CSU')
    rhohv_c = auto_filter.use_csu_wradlib_filters(radar,field_name=rhohv_field,
                                             filter_array=gatefilter,TYPE='CSU')
    kdp_c = auto_filter.use_csu_wradlib_filters(radar,field_name=kdp_field,
                                             filter_array=gatefilter,TYPE='CSU')
    refl_c = auto_filter.use_csu_wradlib_filters(radar,field_name=refl_field,
                                             filter_array=gatefilter,TYPE='CSU')
    
    if use_temp is True:
        pidscore_kabr = csu_fhc.csu_fhc_summer(dz=refl_c,zdr=zdr_c,
                                               rho=rhohv_c,kdp=kdp_c,
                                               use_temp=True,band='S',T=T_field)
    else:        
        pidscore_kabr = csu_fhc.csu_fhc_summer(dz=refl_c,zdr=zdr_c,
                                               rho=rhohv_c,kdp=kdp_c,
                                               use_temp=False,band='S')
        
    pid = np.argmax(pidscore_kabr,axis=0)+1
    radar = tools.add_field_to_radar_object(pid, radar, field_name='PID', units='', 
                                           long_name='PID',standard_name='PID', 
                                           dz_field=refl_field)
    return radar
    
    
