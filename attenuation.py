import pyart
import numpy as np
from scipy.integrate import cumtrapz
from copy import deepcopy
import tools
"""
Correcting attenuations in radar data

Ref: "cpol-processing" package by Valentin Louf (Monash U., Australian BOM)

"""

def correct_attenuation_ref(radar,gatefilter,refl_field=None,ncp_field="NCP",rhv_field=None,phidp_field=None):
    spec_atten, _ = pyart.correct.calculate_attenuation(
        radar,0,rhv_min=0.8,refl_field=refl_field,ncp_field=rhv_field,
        rhv_field=rhv_field,phidp_field=phidp_field,)
    
    specific_attenuation = np.ma.masked_invalid(spec_atten['data'])
    r = radar.range['data'] / 1000
    dr = r[2] - r[1]
    
    na, nr = radar.fields[refl_field]['data'].shape
    attenuation = np.zeros((na,nr))
    attenuation[:,:-1] = 2 * cumtrapz(specific_attenuation,dx=dr)
    refl_corr = radar.fields[refl_field]['data'].copy() + attenuation
    refl_corr = np.ma.masked_where(gatefilter.gate_excluded, refl_corr)
    
    radar = tools.add_field_to_radar_object(refl_corr, radar, field_name='refl_corr', units='dBZ', 
                                            long_name='Corrected reflectivity',
                                            standard_name='Corrected reflectivity', 
                                            dz_field=refl_field)
    radar = tools.add_field_to_radar_object(specific_attenuation, radar, 
                                            field_name='atten_spec', units='dBZ', 
                                           long_name='Specific Attenuation (Reflectivity)',
                                           standard_name='Specific Attenuation (Reflectivity)', 
                                           dz_field=refl_field)
    return radar

def correct_attenuation_zdr(
    radar, gatefilter, zdr_name="ZDR_CORR", phidp_name="PHIDP_VAL", refl_field=None,alpha=0.016):
    """
    V. N. Bringi, T. D. Keenan and V. Chandrasekar, "Correcting C-band radar
    reflectivity and differential reflectivity data for rain attenuation: a
    self-consistent method with constraints," in IEEE Transactions on Geoscience
    and Remote Sensing, vol. 39, no. 9, pp. 1906-1915, Sept. 2001.
    doi: 10.1109/36.951081
    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    zdr_name: str
        Differential reflectivity field name.
    gatefilter:
        Filter excluding non meteorological echoes.
    kdp_name: str
        KDP field name.
    Returns:
    ========
    zdr_corr: array
        Attenuation corrected differential reflectivity.
    """
    zdr = radar.fields[zdr_name]["data"].copy()
    phi = radar.fields[phidp_name]["data"].copy()

    zdr_corr = zdr + 0.016 * phi
    zdr_corr[gatefilter.gate_excluded] = np.NaN
    zdr_corr = np.ma.masked_invalid(zdr_corr)
    np.ma.set_fill_value(zdr_corr, np.NaN)
    # Z-PHI coefficient from Bringi et al. 2001
    
    radar = tools.add_field_to_radar_object(zdr_corr, radar, field_name='zdr_acorr', units='dB', 
                                            long_name='Corrected differential reflectivity',
                                            standard_name='Corrected differential reflectivity', 
                                            dz_field=refl_field)
    return radar
