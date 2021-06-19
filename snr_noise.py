"""
Retrieve signal-to-noise ratio and remove data noise

(Ref: cpol-processing, Py-ART)
"""
import pyart
import scipy
import numpy as np

def _my_snr_from_reflectivity(radar, refl_field="DBZ"):
    """
    Just in case pyart.retrieve.calculate_snr_from_reflectivity, I can calculate
    it 'by hand'.
    Parameter:
    ===========
    radar:
        Py-ART radar structure.
    refl_field_name: str
        Name of the reflectivity field.
    Return:
    =======
    snr: dict
        Signal to noise ratio.
    """
    range_grid, _ = np.meshgrid(radar.range["data"], radar.azimuth["data"])
    range_grid += 1  # Cause of 0

    # remove range scale.. This is basically the radar constant scaled dBm
    pseudo_power = radar.fields[refl_field]["data"] - 20.0 * np.log10(range_grid / 1000.0)
    # The noise_floor_estimate can fail sometimes in pyart, that's the reason
    # why this whole function exists.
    noise_floor_estimate = -40

    snr_field = pyart.config.get_field_name("signal_to_noise_ratio")
    snr_dict = pyart.config.get_metadata(snr_field)
    snr_dict["data"] = pseudo_power - noise_floor_estimate

    return snr_dict

def correct_rhohv(radar, rhohv_name="RHOHV", snr_name="SNR"):
    """
    Correct cross correlation ratio (RHOHV) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 5)
    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        rhohv_name: str
            Cross correlation field name.
        snr_name: str
            Signal to noise ratio field name.
    Returns:
    ========
        rho_corr: array
            Corrected cross correlation ratio.
    """
    rhohv = radar.fields[rhohv_name]["data"].copy()
    snr = radar.fields[snr_name]["data"].copy()

    natural_snr = 10 ** (0.1 * snr)
    natural_snr = natural_snr.filled(-9999)
    rho_corr = rhohv * (1 + 1 / natural_snr)

    # Not allowing the corrected RHOHV to be lower than the raw rhohv
    rho_corr[np.isnan(rho_corr) | (rho_corr < 0) | (rho_corr > 1)] = 1
    try:
        rho_corr = rho_corr.filled(1)
    except Exception:
        pass

    return rho_corr


def correct_zdr(radar, zdr_name="ZDR", snr_name="SNR"):
    """
    Correct differential reflectivity (ZDR) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 6)
    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        zdr_name: str
            Differential reflectivity field name.
        snr_name: str
            Signal to noise ratio field name.
    Returns:
    ========
        corr_zdr: array
            Corrected differential reflectivity.
    """
    zdr = radar.fields[zdr_name]["data"].copy()
    snr = radar.fields[snr_name]["data"].copy()
    alpha = 1.48
    natural_zdr = 10 ** (0.1 * zdr)
    natural_snr = 10 ** (0.1 * snr)
    corr_zdr = 10 * np.log10((alpha * natural_snr * natural_zdr) / (alpha * natural_snr + alpha - natural_zdr))

    return corr_zdr

def calc_snr(radar, refl_field_name="DBZ"):
    """
    Parameters:
    ===========
        radar:
        refl_field_name: str
            Name of the reflectivity field.
    Returns:
    ========
        z_dict: dict
            Altitude in m, interpolated at each radar gates.

        snr: dict
            Signal to noise ratio.
    """
    # Calculate SNR
    snr = pyart.retrieve.calculate_snr_from_reflectivity(radar, refl_field=refl_field_name)
    #if snr["data"].count() == 0:
    #    snr = pyart.retrieve.calculate_snr_from_reflectivity(radar, refl_field=refl_field_name, toa=20000)
    #if snr["data"].count() == 0:
        # If it fails again, then we compute the SNR with the noise value
        # given by the CPOL radar manufacturer.
    #    snr = _my_snr_from_reflectivity(radar, refl_field=refl_field_name)
    field_dict = {'data': snr['data'],
                  'units': "dB",
                  'long_name': 'Signal-to-noise ratio',
                  'standard_name': 'Sigal-to-noise ratio',
                  '_FillValue': radar.fields[refl_field_name]['_FillValue']}
    radar.add_field("SNR", field_dict, replace_existing=True)
    return snr