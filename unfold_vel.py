"""
Correct Radar Radial Velocity field

Ref: cpol_processing (Valentin Louf; BOM, Monash U.), pyart (Helmus and Collis),
"""
import pyart
import numpy as np

def _check_nyquist(radar,vel_field="VEL"):
    """
    Purpose: Check if radar data contains nyquist velocity information, create
    this parameter if not provided in radar data
    
    """
    try:
        nyvel = radar.instrument_parameters["nyquist_velocity"]["data"]
        if nyvel is None:
            raise KeyError("Nyquist velocity does not exist in radar data")
    except KeyError:
        nyvel = np.nanmax(radar.fields[vel_field]["data"])
        nray = len(radar.azimuth["data"])
        vnyq_array = np.array([nyvel]*nray, dtype=np.float32)
        nyquist_vel = pyart.config.get_metadata("nyquist_velocity")
        nyquist_vel['data'] = vnyq_array
        nyquist_vel['_Least_significant_digit'] = 2
        radar.instrument_parameters["nyquist_velocity"] = nyquist_vel
    return nyvel

def unfold3d(radar,gatefilter,vel_field="VEL",dbz_field="DBZ",nyquist=None):
    """
    Ref: Louf et al. (2020) [J. Atmos. Ocean. Tech.]

    """
    import unravel
    
    vnyq = _check_nyquist(radar,vel_field)
    if nyquist is None:
        if np.isscalar(vnyq):
            nyquist = vnyq

    unfvel = unravel.unravel_3D_pyart(
        radar,vel_field,dbz_field,gatefilter,nyquist,"long_range")
    
    vel_meta = pyart.config.get_metadata("velocity")
    vel_meta['data'] = np.ma.masked_where(gatefilter.gate_excluded, unfvel).astype(np.float32)
    vel_meta['_Least_significant_digit'] = 2
    vel_meta['_FillValue'] = np.nan
    vel_meta['comment'] = "UNRAVEL Algorithm"
    vel_meta['long_name'] = "Unfolded radial velocity of scatterers away from instrument"
    vel_meta['units'] = 'm s-1'
    
    return vel_meta

