import pyart
import numpy as np
import pandas as pd
import tools

def data_texture(inputdata=None):
    """
    Compute the spatial variability of input fields in the azimuthal and beam direction
    
    (Ref: Gourley et al. 2007 [J. Atmos. Ocean. Tech.], wradlib package)
    """
    
    x1 = np.roll(inputdata,1,-2)
    x2 = np.roll(inputdata,1,-1)
    x3 = np.roll(inputdata,-1,-2)
    x4 = np.roll(inputdata,-1,-1)
    x5 = np.roll(x1,1,-1)
    x6 = np.roll(x4,1,-2)
    x7 = np.roll(x3,-1,-1)
    x8 = np.roll(x2,-1,-2)
    
    xa = np.array([x1,x2,x3,x4,x5,x6,x7,x8])
    
    xa_valid = np.ones(np.shape(xa))
    xa_valid[np.isnan(xa)] = 0
    xa_valid_count = np.sum(xa_valid,axis=0)
    
    num = np.zeros(inputdata.shape)
    for xarr in xa:
        diff = inputdata-xarr
        diff[np.isnan(diff)] = 0
        num += diff**2
        
    num[np.isnan(inputdata)] = np.nan
    
    return np.sqrt(num/xa_valid_count)

def basic_filter(radar,refl_field="DBZ",phidp_field="PHIDP",
                 rhohv_field="RHOHV_CORR",zdr_field="ZDR",snr_field="SNR"):
    
    gf = pyart.filters.GateFilter(radar)
    
    #ZDR and DBZ filtering
    gf.exclude_outside(zdr_field,-4.0,7.0)
    gf.exclude_outside(refl_field,-20.0,80.0)
    
    #Compute PHIDP texture and remove noise based on the results
    dphi = data_texture(radar.fields[phidp_field]['data'])
    radar.add_field_like(phidp_field,"PHIDPTEX",dphi)
    gf.exclude_above("PHIDPTEX",20)
    
    #Remove non-meteorological echoes with RhoHV
    gf.exclude_below(rhohv_field,0.7)
    
    #Despeckle
    gf_despeckled = pyart.correct.despeckle_field(radar,refl_field,gatefilter=gf)
    
    try:
        radar.field.pop("PHIDPTEX")
    except Exception:
        pass
    
    return gf_despeckled, gf

def insect_clutter_filter(radar,refl_field="DBZ",zdr_field="ZDR"):
    """
    Reference: Lang et al. (2007; J. Clim.), Gabella et al. (2002)
    """
    from csu_radartools import csu_misc
    import wradlib.clutter as clutter
    
    dbz = radar.fields[refl_field]['data'].copy()
    zdr = radar.fields[zdr_field]['data'].copy()
    insect_mask = csu_misc.insect_filter(dbz,zdr,bad=-9999)
    
    dbz[insect_mask] = -9999
    dbz_insect = np.ma.masked_where(dbz==-9999,dbz)
    
    clutter_ref = clutter.filter_gabella(dbz_insect)
    dbz_insectc = np.ma.masked_where(clutter_ref!=0, dbz_insect)
    
    field_dict = {'data': dbz_insectc,
                  'units': "dBZ",
                  'long_name': 'Insect-and-clutter-filtered reflectivity',
                  'standard_name': 'dbz_insect',
                  '_FillValue': radar.fields[refl_field]['_FillValue']}
    radar.add_field("refl_insect", field_dict, replace_existing=True)
    return radar, insect_mask, clutter_ref

def use_csu_wradlib_filters(radar,field_name=None,filter_array=None,TYPE='CSU'):
    field_c = radar.fields[field_name]['data'].copy()
    if TYPE=='CSU':
        field_c[filter_array] = -9999
    elif TYPE=='wradlib':
        field_c = np.ma.masked_where(filter_array!=0,field_c)
    return field_c

def filter_hardcode(radar,field_name=None,gatefilter=None,filtfield_name=None,
                    filtfield_unit=None):
    dbzf = radar.fields[field_name]['data'].copy()
    dbzf[gatefilter.gate_excluded] = np.nan
    radar = tools.add_field_to_radar_object(dbzf, radar, 
                                           filtfield_name,filtfield_unit,filtfield_name,
                                           filtfield_name,'reflectivity')
    return radar
    
    
        