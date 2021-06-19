"""
Process differential phase and calculate KDP

Ref. cpol_processing (Valentin Louf), csu_radartools
"""

import pyart
import scipy
import numpy as np
from scipy import integrate
from csu_radartools import csu_kdp
import tools

def unfold_raw_phidp(radar,gatefilter,phidp_field='differential_phase'):
    unfphidic = pyart.correct.dealias_unwrap_phase(radar,gatefilter=gatefilter,skip_checks=True,
                                               vel_field=phidp_field,nyquist_vel=90)
    radar.add_field_like(phidp_field, 'PHIDP_U', unfphidic['data'])
    return radar

def _fix_phidp_from_kdp(phidp, kdp, r, gatefilter):
    """
    Correct PHIDP and KDP from spider webs.
    Parameters
    ==========
    r:
        Radar range.
    gatefilter:
        Gate filter.
    kdp_name: str
        Differential phase key name.
    phidp_name: str
        Differential phase key name.
    Returns:
    ========
    phidp: ndarray
        Differential phase array.
    """
    kdp[gatefilter.gate_excluded] = 0
    kdp[(kdp < -4)] = 0
    kdp[kdp > 15] = 0
    interg = integrate.cumtrapz(kdp, r, axis=1)

    phidp[:, :-1] = interg / (len(r))
    return phidp, kdp

def kdp_bringi(radar,gatefilter,raw_phidp_name="differential_phase",
               refl_name="DBZ"):
    """
    Parameters
    ----------
    radar : Py-ART data structure
    gatefilter : Filter for all radar gates
    unfold_phidp_name : Differential phase key name (str)
    refl_name : Reflectivity field name (str)

    Returns
    -------
    phidpb : Bringi differential phase field
    kdpb : KDP field computed with Bringi method
    """
    unfphidic = pyart.correct.dealias_unwrap_phase(radar,gatefilter=gatefilter,skip_checks=True,
                                               vel_field=raw_phidp_name,nyquist_vel=90)
    radar.add_field_like(raw_phidp_name, 'PHITMP', unfphidic['data'])
    dp = tools.extract_unmasked_data(radar, 'PHITMP', bad=-9999)
    dz = tools.extract_unmasked_data(radar, refl_name, bad=-9999)
        
    # Dimensions
    rng,azi = radar.range['data'],radar.azimuth['data']
    drng = rng[1] - rng[0]
    [R, A] = np.meshgrid(rng,azi)
    
    kdpb, phidpb, sdnb = csu_kdp.calc_kdp_bringi(dp, dz, R/1e3, gs=drng,
                                              bad=-9999, thsd=12, window=3.0, std_gate=11)
    radar = tools.add_field_to_radar_object(kdpb, radar, field_name='KDP', units='deg/km', 
                                           long_name='Specific Differential Phase',
                                           standard_name='Specific Differential Phase', 
                                           dz_field=refl_name)
    radar = tools.add_field_to_radar_object(phidpb, radar, field_name='FDP', units='deg', 
                                           long_name='Filtered Differential Phase',
                                           standard_name='Filtered Differential Phase', 
                                           dz_field=refl_name)
    radar = tools.add_field_to_radar_object(sdnb, radar, field_name='SDP', units='deg', 
                                           long_name='Standard Deviation of Differential Phase',
                                           standard_name='Standard Deviation of Differential Phase', 
                                           dz_field=refl_name)

    return radar,kdpb,phidpb,sdnb

def phidp_giangrande(radar, gatefilter, refl_field='DBZ', ncp_field='NCP',
                     rhv_field='RHOHV_CORR', phidp_field='PHIDP'):
    """
    Phase processing using the LP method in Py-ART. A LP solver is required,
    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    gatefilter:
        Gate filter.
    refl_field: str
        Reflectivity field label.
    ncp_field: str
        Normalised coherent power field label.
    rhv_field: str
        Cross correlation ration field label.
    phidp_field: str
        Differential phase label.
    Returns:
    ========
    phidp_gg: dict
        Field dictionary containing processed differential phase shifts.
    kdp_gg: dict
        Field dictionary containing recalculated differential phases.
    """
    unfphidic = pyart.correct.dealias_unwrap_phase(radar,
                                                   gatefilter=gatefilter,
                                                   skip_checks=True,
                                                   vel_field=phidp_field,
                                                   nyquist_vel=90)

    radar.add_field_like(phidp_field, 'PHITMP', unfphidic['data'])

    phidp_gg, kdp_gg = pyart.correct.phase_proc_lp(radar, 0.0,
                                                   LP_solver='pyglpk',
                                                   ncp_field=ncp_field,
                                                   refl_field=refl_field,
                                                   rhv_field=rhv_field,
                                                   phidp_field='PHITMP')

    phidp_gg['data'], kdp_gg['data'] = _fix_phidp_from_kdp(phidp_gg['data'],
                                                           kdp_gg['data'],
                                                           radar.range['data'],
                                                           gatefilter)

    try:
        # Remove temp variables.
        radar.fields.pop('unfolded_differential_phase')
        radar.fields.pop('PHITMP')
    except Exception:
        pass

    phidp_gg['data'] = phidp_gg['data'].astype(np.float32)
    phidp_gg['_Least_significant_digit'] = 4
    kdp_gg['data'] = kdp_gg['data'].astype(np.float32)
    kdp_gg['_Least_significant_digit'] = 4

    return phidp_gg, kdp_gg

