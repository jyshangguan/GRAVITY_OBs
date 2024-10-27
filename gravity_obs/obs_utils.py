import numpy as np
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import CIRS, GCRS
from astropy.coordinates import SkyCoord, Distance
from astropy.table import Table
import bisect

try:
    from astroquery.xmatch import XMatch
    from astroquery.simbad import Simbad
except ImportError:
    XMatch = None
    Simbad = None

 
__all__ = ['coord_offset', 'cal_offset', 'cal_coord_motion', 'sc_offset', 
           'coord_colon_to_degree', 'coord_degree_to_colon',
           'get_coord_plain', 'get_coord_colon', 'get_pos_current', 
           'get_pos_J2000', 'get_pos_ao', 'read_coordinate', 
           'coordinate_convert_epoch', 'search_gaia_single', 
           'search_gaia_2table', 'xmatch_gaiadr2', 'xmatch_gaiadr3', 
           'xmatch_2mass_psc', 'search_simbad']


def cal_coord_motion(c, pma=None, pmd=None, plx=None, radvel=None,
                     time_ref='J2000.0', time_cal='now', frame=GCRS):
    """
    Calculate the current coordinates considering the motion of the source.

    Parameters
    ----------
    c : Astropy SkyCoord
        The coordinate to calculate the motion.
    pma (optional) : Astropy Quantity (angular velocity)
        The proper motion of RA.
    pmd (optional) : Astropy Quantity (angular velocity)
        The proper motion of DEC.
    plx (optional) : Astropy Quantity (angle)
        The parallex.
    radvel (optional) : Astropy Quantity (velocity)
        The radial velocity.
    time_ref : string (default: 'J2000.0')
        The reference time of the input data.
    time_cal : string (default: 'now')
        The time to calculate the motion of the coordinate.
    frame : coordinate systems (default: GCRS)
        The coordinate system, which need to consider the orbit of the Earth
        around the Sun.

    Returns
    -------
    c_c : Astropy SkyCoord
        The calculated coordinate.
    """
    if pma is None:
        pma = 1e-16*u.mas/u.yr
    if pmd is None:
        pmd = 1e-16*u.mas/u.yr
    if plx is None:
        plx = 0 * u.mas
    if radvel is None:
        radvel = 0 * u.km/u.s
    c_m = SkyCoord(ra=c.ra, dec=c.dec, pm_ra_cosdec=pma, pm_dec=pmd,
                   distance=Distance(parallax=plx), radial_velocity=radvel,
                   frame=c.frame.name, obstime=time_ref)
    if time_cal == 'now':
        time_cal = Time.now()
    else:
        time_cal = Time(time_cal)
    c_c = c_m.apply_space_motion(time_cal).transform_to(GCRS(obstime=time_cal))
    return c_c


def sc_offset(c_sc, c_ft, pma_sc=None, pmd_sc=None, plx_sc=None, radvel_sc=None,
              pma_ft=None, pmd_ft=None, plx_ft=None, radvel_ft=None,
              time_ref="J2000.0", time_cal='now', frame=GCRS):
    '''
    Calculate the offset of the SC fiber from the FT target.

    Parameters
    ----------
    c_sc : Astropy SkyCoord
        The coordinate of the SC target.
    c_ft : Astropy SkyCoord
        The coordinate of the FT target.
    pma_sc (optional) : float
        The proper motion of RA, units: mas/yr.
    pmd_sc (optional) : float
        The proper motion of DEC, units: mas/yr.
    plx_sc (optional) : float
        The parallex, units: mas.
    radvel_sc (optional) : float
        The radial velocity, units: km/s.
    pma_ft (optional) : float
        The proper motion of RA, units: mas/yr.
    pmd_ft (optional) : float
        The proper motion of DEC, units: mas/yr.
    plx_ft (optional) : float
        The parallex, units: mas.
    radvel_ft (optional) : float
        The radial velocity, units: km/s.

    Returns
    -------
    delta_ra : float
        The offset of RA, units: mas.
    delta_dec : float
        The offset of DEC, units: mas.
    '''
    if not pma_sc is None:
        pma_sc = pma_sc * u.mas / u.yr
    if not pmd_sc is None:
        pmd_sc = pmd_sc * u.mas / u.yr
    if not plx_sc is None:
        plx_sc = plx_sc * u.mas
    if not radvel_sc is None:
        radvel_sc = radvel_sc * u.km/u.s
    if not pma_ft is None:
        pma_ft = pma_ft * u.mas / u.yr
    if not pmd_ft is None:
        pmd_ft = pmd_ft * u.mas / u.yr
    if not plx_ft is None:
        plx_ft = plx_ft * u.mas
    if not radvel_ft is None:
        radvel_ft = radvel_ft * u.km/u.s
    c_sc_now = cal_coord_motion(c_sc, pma=pma_sc, pmd=pmd_sc, plx=plx_sc, radvel=radvel_sc,
                                time_ref=time_ref, time_cal=time_cal, frame=frame)
    c_ft_now = cal_coord_motion(c_ft, pma=pma_ft, pmd=pmd_ft, plx=plx_ft, radvel=radvel_ft,
                                time_ref=time_ref, time_cal=time_cal, frame=frame)
    delta_ra, delta_dec = cal_offset(c_sc_now, c_ft_now)
    return delta_ra, delta_dec


def coord_colon_to_degree(ra, dec):
    '''
    Convert the coordinate from HH:MM:SS to degree.

    Parameters
    ----------
    ra : string or float
        The right ascension (HH:MM:SS).
    dec : string or float
        The declination (DD:MM:SS).

    Returns
    -------
    ra_deg, dec_deg : float
        The coordinates in degree.
    '''
    raList = np.atleast_1d(ra)
    decList = np.atleast_1d(dec)

    ra_deg = []
    dec_deg = []
    for loop, (r, d) in enumerate(zip(raList, decList)):
        c = read_coordinate(r, d)
        ra_deg.append(c.ra.degree)
        dec_deg.append(c.dec.degree)
    ra_deg = np.squeeze(ra_deg)
    dec_deg = np.squeeze(dec_deg)
    return ra_deg, dec_deg


def coord_degree_to_colon(ra, dec):
    '''
    Convert the coordinate from degree to HH:MM:SS.

    Parameters
    ----------
    ra : float
        The right ascension (degree).
    dec : float
        The declination (degree).
    
    Returns
    -------
    ra_colon, dec_colon : string
        The coordinates in HH:MM:SS and DD:MM:SS.
    '''
    raList = np.atleast_1d(ra)
    decList = np.atleast_1d(dec)

    ra_colon = []
    dec_colon = []
    for loop, (r, d) in enumerate(zip(raList, decList)):
        c = read_coordinate(r, d)
        ra_tmp, dec_tmp = get_coord_colon(c)
        ra_colon.append(ra_tmp)
        dec_colon.append(dec_tmp)
    ra_colon = np.squeeze(ra_colon)
    dec_colon = np.squeeze(dec_colon)
    return ra_colon, dec_colon


def coord_offset(delta_ra, delta_dec, c0, frame='icrs'):
    """
    Calculate the coordinate, that is offset from a reference coordinate.

    Parameters
    ----------
    delta_ra : float
        The ra offset of the target, units: arcsec.
    delta_dec : float
        The dec offset of the target, units: arcsec.
    c0 : Astropy SkyCoord
        The reference coordinate.
    frame : string (default: 'icrs')
        The coordinate frame.

    Returns
    -------
    c : Astropy SkyCoord
        The coordinates of the target.
    """
    ra_t  = c0.ra.arcsec + delta_ra / np.cos(c0.dec.radian)
    dec_t = c0.dec.arcsec + delta_dec
    c_t = SkyCoord(ra=ra_t*u.arcsec, dec=dec_t*u.arcsec, frame=frame)
    return c_t


def cal_pa_sep(offset):
    '''
    Calculate the PA and separation of the offset from the input (dx, dy)

    Parameters
    ----------
    offset : tuple
        The offset of the target in linear scale, (dx, dy).

    Returns
    -------
    pa : float
        The position angle of the offset, units: degree.
    sep : float
        The separation of the offset, units: the input unit.
    '''
    dx, dy = offset
    pa = np.arctan2(dx, dy) * 180/np.pi
    sep = np.sqrt(dx**2 + dy**2)
    return pa, sep


def cal_offset(c, c_ref, units='mas'):
    """
    Calculate the coordinate offset betweenn the two coordinates.

    Parameters
    ----------
    c : Astropy SkyCoord
        The coordinate to calculate the offset.
    c_ref : Astropy SkyCoord
        The reference coordinate.

    Returns
    -------
    (delta_ra, delta_dec) : (float, float)
        The offsets of right ascension and declination, units: arcsec.
    """
    sep = c_ref.separation(c)
    pa = c_ref.position_angle(c)
    delta_ra = (sep * np.sin(pa)).to(units)
    delta_dec = (sep * np.cos(pa)).to(units)
    return (delta_ra, delta_dec)


def get_coord_plain(c):
    '''
    Convert the coordinate to the (HHMMSS.SSS, DDMMSS.SSS) format.
    
    Parameters
    ----------
    c : SkyCoord
        The coordinate object.
    
    Returns
    -------
    ra_plain, dec_plain : string
        The coordinates in HHMMSS.SSS and DDMMSS.SSS
    '''
    ra, dec = c.to_string('hmsdms').split(' ')
    ra_h = ra[:2]
    ra_m = ra[3:5]
    ra_s = float(ra[6:-1])
    dec_d, dec_tmp = dec.split('d')
    dec_m, dec_s = dec_tmp.split('m')
    dec_s = float(dec_s[:-1])
    ra_plain = '{0}{1}{2}'.format(ra_h, ra_m, '{0:.3f}'.format(ra_s).zfill(6))
    dec_plain = '{0}{1}{2}'.format(dec_d, dec_m, '{0:.3f}'.format(dec_s).zfill(6))
    if dec_plain[0] == '+':
        dec_plain = dec_plain[1:]
    return ra_plain, dec_plain


def get_coord_colon(c):
    '''
    Convert the coordinate to the (HH:MM:SS.SSS, DD:MM:SS.SSS) format.
    
    Parameters
    ----------
    c : SkyCoord
        The coordinate object.
    
    Returns
    -------
    ra_colon, dec_colon : string
        The coordinates in HH:MM:SS.SSS and DD:MM:SS.SSS
    '''
    try:
        ra, dec = c.to_string('hmsdms').split(' ')
    except ValueError:
        return '00:00:00.000', '00:00:00.000'
    
    if ra == 'nan':
        return '00:00:00.000', '00:00:00.000'

    ra_h = ra[:2]
    ra_m = ra[3:5]
    ra_s = float(ra[6:-1])
    dec_d, dec_tmp = dec.split('d')
    dec_m, dec_s = dec_tmp.split('m')
    dec_s = float(dec_s[:-1])
    ra_colon = '{0}:{1}:{2}'.format(ra_h, ra_m, '{0:.3f}'.format(ra_s).zfill(6))
    dec_colon = ' {0}:{1}:{2}'.format(dec_d, dec_m, '{0:.3f}'.format(dec_s).zfill(6))
    return ra_colon, dec_colon


def get_pos_current(ra, dec, pma, pmd, parallax, radvel):
    '''
    Get the target position dict for the sequencer.
    
    Parameters
    ----------
    ra : float
        RA in degree.
    dec : float
        DEC in degree.
    pma : float
        Proper motion in mas.
    pmd : float
        Proper motion in mas.
    plx : float
        Parallax in mas.
    radvel : float
        Radial velocity in km/s.
        
    Returns
    -------
    pos : dict
        The dict of the position information of the target.
    '''
    c = read_coordinate(ra, dec)
    c_cur = cal_coord_motion(c, pma*u.mas/u.yr, pmd*u.mas/u.yr, parallax*u.mas, radvel*u.km/u.s)
    ra_hms, dec_dms = get_coord_plain(c_cur)
    pos = dict(ra=ra_hms, 
               dec=dec_dms, 
               pma='{0:.5f}'.format(pma*1e-3), 
               pmd='{0:.5f}'.format(pmd*1e-3), 
               parallax='{0:.5f}'.format(parallax*1e-3), 
               radvel=radvel)
    return pos


def get_pos_J2000(ra, dec, pma, pmd, parallax, radvel):
    '''
    Get the target position dict for the sequencer.
    
    Parameters
    ----------
    ra : float
        RA in degree.
    dec : float
        DEC in degree.
    pma : float
        Proper motion in mas.
    pmd : float
        Proper motion in mas.
    plx : float
        Parallax in mas.
    radvel : float
        Radial velocity in km/s.
        
    Returns
    -------
    pos : dict
        The dict of the position information of the target.
    '''
    c = read_coordinate(ra, dec)
    ra_hms, dec_dms = get_coord_plain(c)
    pos = dict(ra=ra_hms, 
               dec=dec_dms, 
               pma='{0:.5f}'.format(pma*1e-3), 
               pmd='{0:.5f}'.format(pmd*1e-3), 
               parallax='{0:.5f}'.format(parallax*1e-3), 
               radvel=radvel)
    return pos


def get_pos_ao(ra, dec, pma, pmd, parallax, radvel):
    '''
    Get the AO source position dict for the sequencer.
    
    Parameters
    ----------
    ra : float
        RA in degree.
    dec : float
        DEC in degree.
    pma : float
        Proper motion in mas.
    pmd : float
        Proper motion in mas.
    plx : float
        Parallax in mas.
    radvel : float
        Radial velocity in km/s.
        
    Returns
    -------
    pos : dict
        The dict of the position information of the target.
    '''
    c = read_coordinate(ra, dec)
    #c_cur = cal_coord_motion(c, pma*u.mas/u.yr, pmd*u.mas/u.yr, parallax*u.mas, radvel*u.km/u.s)
    ra_hms, dec_dms = get_coord_colon(c)
    pos = dict(ra=ra_hms, 
               dec=dec_dms, 
               pma='{0:.5f}'.format(pma*1e-3), 
               pmd='{0:.5f}'.format(pmd*1e-3), 
               parallax='{0:.5f}'.format(parallax*1e-3), 
               radvel=radvel)
    return pos
    
    
def coordinate_convert_epoch(ra, dec, pma, pmd, parallax, radial_velocity, 
                             ref_epoch, target_epoch):
    '''
    Convert the coordinate to the targeted epoch.
    
    Parameters
    ----------
    ra : float
        R.A. in degree.
    dec : float
        Decl. in degree.
    pma : float
        Proper motion of R.A., units: mas/yr.
    pmd : float
        Proper motion of Decl., units: mas/yr.
    parallax : float
        Parallax, in mas.
    radial_velocity : float
        Radial velocity, units: km/s.
    ref_epoch : string
        The reference epoch. J2015.5 for Gaia.
    target_epoch : string
        The targeted epoch. Typically, J2000.
    
    Returns
    -------
    c_t : SkyCoord
        The converted coordinate.
    '''
    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, 
                 pm_ra_cosdec=pma*u.mas/u.yr, pm_dec=pmd*u.mas/u.yr, 
                 distance=Distance(parallax=parallax*u.mas), 
                 radial_velocity=radial_velocity*u.km/u.s, 
                 obstime=ref_epoch)
    c_t = c.apply_space_motion(Time(target_epoch))
    return c_t
    
    
def search_gaia_single(ra, dec, radius):
    '''
    Search the target in Gaia.
    
    Parameters
    ----------
    ra : string or float
        R.A. in 'h:m:s' or degree.
    dec : string or float
        Decl. in 'd:m:s' or degree.
    radius : float
        Searching radius in arcsec.
        
    Returns
    -------
    Result table of the closest target.
    '''
    coord = read_coordinate(ra, dec)
    radius = u.Quantity(radius, u.arcsec)
    j = Gaia.cone_search_async(coordinate=coord, radius=radius)
    r = j.get_results()
    
    len_tb = len(r)
    if len_tb == 0:
        print(ra, dec)
        raise RuntimeError('Cannot find any results!')
    
    if len_tb > 1:
        fltr = np.arange(len_tb) == np.argmin(r['dist'])
        r = r[fltr]
    
    return r
    
    
def read_coordinate(ra, dec):
    '''
    Read in the coordinate, either in degree or hourangle. Only use ICRS frame.
    
    Parameters
    ----------
    ra : float or string
        The right ascension (degree or HH:MM:SS).
    dec : float or string
        The declination (degree or DD:MM:SS).
    
    Returns
    -------
    c : SkyCoord
        The coordinate object.
    '''
    if isinstance(ra, str):
        assert isinstance(dec, str)
        c = SkyCoord('{0} {1}'.format(ra, dec), frame='icrs', unit=(u.hourangle, u.deg))
    else:
        c = SkyCoord(ra, dec, frame='icrs', unit='deg')
    return c
    
    
def search_gaia_2table(ra, dec, radius):
    '''
    Search for Gaia info and convert them to J2000 into a table.

    Parameters
    ----------
    ra : float or string
        The right ascension (degree or HH:MM:SS).
    dec : float or string
        The declination (degree or DD:MM:SS).
    radius : float
        Searching radius in arcsec.
    
    Returns
    -------
    tb : Astropy Table
        The table of the target.
    '''
    r = search_gaia_single(ra, dec, radius)
    
    ra = r['ra'][0]
    dec = r['dec'][0]
    pma = r['pmra'][0]
    pmd = r['pmdec'][0]
    plx = r['parallax'][0]
    rv = r['radial_velocity'][0]
    ref_epoch = 'J{}'.format(r['ref_epoch'][0])
    target_epoch = 'J2000'
    
    if plx < 0:
        plx = 0
    c2000 = coordinate_convert_epoch(ra, dec, pma, pmd, plx, rv, ref_epoch, target_epoch)
    
    ra_c, dec_c = get_coord_colon(c2000)
    
    tb = Table([[ra_c], [dec_c], [pma], [pmd], [plx], [rv], r['phot_g_mean_mag']], 
               names=['ra', 'dec', 'pma', 'pmd', 'plx', 'rv', 'G'])
    tb['pma'].format = '%.3f'
    tb['pmd'].format = '%.3f'
    tb['plx'].format = '%.3f'
    tb['rv'].format = '%.2f'
    tb['G'].format = '%.1f'
    return tb


def xmatch_gaiadr2(t, radius, colRA, colDec):
    '''
    Cross match the table with Gaia.
    
    Parameters
    ----------
    t : Astropy Table
        The table of targets.
    radius : float
        The cross match radius, units: arcsec.
    colRA : string
        The column name of the RA.
    colDec : string
        The column name of the Dec.
    vizier_code : string
        The vizieR code of the Gaia table.
        
    Returns
    -------
    t_f : Astropy Table
        The table of cross matched results.
    '''
    assert XMatch is not None, 'The astroquery is not installed!'

    for cn in ['ra_gaia_J2000', 'dec_gaia_J2000', 'pma', 'pmd', 'plx', 'rv', 'G']:
        assert cn not in t.colnames, 'The input table has {} as a column!'.format(cn)
    
    t_o = XMatch.query(cat1=t,
                       cat2='vizier:I/345/gaia2',
                       max_distance=radius * u.arcsec, colRA1=colRA,
                       colDec1=colDec, colRA2='RAJ2000', colDec2='DECJ2000')
    
    ra_j2000 = []
    dec_j2000 = []
    ra_j2015 = []
    dec_j2015 = []
    pma = []
    pmd = []
    plx = []
    rv = []
    G = []
    Grp = []
    Gbp = []
    sourceID = []
    for loop in range(len(t)):
        fltr = np.isclose(t_o[colRA], t[colRA][loop]) & np.isclose(t_o[colDec], t[colDec][loop])
        if np.sum(fltr) == 0:
            ra_j2000.append(np.nan)
            dec_j2000.append(np.nan)
            ra_j2015.append(np.nan)
            dec_j2015.append(np.nan)
            pma.append(np.nan)
            pmd.append(np.nan)
            plx.append(np.nan)
            rv.append(np.nan)
            G.append(np.nan)
            Grp.append(np.nan)
            Gbp.append(np.nan)
            sourceID.append(np.nan)
        else:
            t_f = t_o[fltr]
            idx = np.argmin(t_f['angDist'])
            ra_j2000.append(t_f['ra_epoch2000'][idx])
            dec_j2000.append(t_f['dec_epoch2000'][idx])
            ra_j2015.append(t_f['ra'][idx])
            dec_j2015.append(t_f['dec'][idx])
            pma.append(t_f['pmra'][idx])
            pmd.append(t_f['pmdec'][idx])
            plx.append(t_f['parallax'][idx])
            rv.append(t_f['radial_velocity'][idx])
            G.append(t_f['phot_g_mean_mag'][idx])
            Grp.append(t_f['phot_rp_mean_mag'][idx])
            Gbp.append(t_f['phot_bp_mean_mag'][idx])
            sourceID.append(t_f['source_id'][idx])
    t_f = t.copy()
    t_f.add_columns([ra_j2000, dec_j2000, ra_j2015, dec_j2015, pma, pmd, plx, rv, G, Grp, Gbp, sourceID], 
                   names=['ra_gaia_J2000', 'dec_gaia_J2000', 'ra_gaia_J2015', 'dec_gaia_J2015',
                          'pma', 'pmd', 'plx', 'rv', 'G', 'Grp', 'Gbp', 'sourceID'])
    return t_f


def xmatch_gaiadr3(t, radius, colRA, colDec):
    '''
    Cross match the table with Gaia.
    
    Parameters
    ----------
    t : Astropy Table
        The table of targets.
    radius : float
        The cross match radius, units: arcsec.
    colRA : string
        The column name of the RA.
    colDec : string
        The column name of the Dec.
    vizier_code : string
        The vizieR code of the Gaia table.
        
    Returns
    -------
    t_f : Astropy Table
        The table of cross matched results.
    '''
    assert XMatch is not None, 'The astroquery is not installed!'

    for cn in ['ra_gaia_J2000', 'dec_gaia_J2000', 'pma', 'pmd', 'plx', 'rv', 'G']:
        assert cn not in t.colnames, 'The input table has {} as a column!'.format(cn)
    
    t_o = XMatch.query(cat1=t,
                       cat2='vizier:I/355/gaiadr3',
                       max_distance=radius * u.arcsec, colRA1=colRA, colDec1=colDec)
    
    ra_j2000 = []
    ra_j2000_err = []
    dec_j2000 = []
    dec_j2000_err = []
    ra_j2016 = []
    ra_j2016_err = []
    dec_j2016 = []
    dec_j2016_err = []
    pma = []
    pma_err = []
    pmd = []
    pmd_err = []
    plx = []
    plx_err = []
    rv = []
    rv_err = []
    G = []
    G_err = []
    Grp = []
    Grp_err = []
    Gbp = []
    Gbp_err = []
    sourceID = []
    ruwe = []
    for loop in range(len(t)):
        fltr = np.isclose(t_o[colRA], t[colRA][loop]) & np.isclose(t_o[colDec], t[colDec][loop])
        if np.sum(fltr) == 0:
            ra_j2000.append(np.nan)
            ra_j2000_err.append(np.nan)
            dec_j2000.append(np.nan)
            dec_j2000_err.append(np.nan)
            ra_j2016.append(np.nan)
            ra_j2016_err.append(np.nan)
            dec_j2016.append(np.nan)
            dec_j2016_err.append(np.nan)
            pma.append(np.nan)
            pma_err.append(np.nan)
            pmd.append(np.nan)
            pmd_err.append(np.nan)
            plx.append(np.nan)
            plx_err.append(np.nan)
            rv.append(np.nan)
            rv_err.append(np.nan)
            G.append(np.nan)
            G_err.append(np.nan)
            Grp.append(np.nan)
            Grp_err.append(np.nan)
            Gbp.append(np.nan)
            Gbp_err.append(np.nan)
            sourceID.append(np.nan)
            ruwe.append(np.nan)
        else:
            t_f = t_o[fltr]
            idx = np.argmin(t_f['angDist'])
            ra_j2000.append(t_f['RAJ2000'][idx])
            ra_j2000_err.append(t_f['e_RAJ2000'][idx])
            dec_j2000.append(t_f['DEJ2000'][idx])
            dec_j2000_err.append(t_f['e_DEJ2000'][idx])
            ra_j2016.append(t_f['RAdeg'][idx])
            ra_j2016_err.append(t_f['e_RAdeg'][idx])
            dec_j2016.append(t_f['DEdeg'][idx])
            dec_j2016_err.append(t_f['e_DEdeg'][idx])
            pma.append(t_f['pmRA'][idx])
            pma_err.append(t_f['e_pmRA'][idx])
            pmd.append(t_f['pmDE'][idx])
            pmd_err.append(t_f['e_pmDE'][idx])
            plx.append(t_f['Plx'][idx])
            plx_err.append(t_f['e_Plx'][idx])
            rv.append(t_f['RV'][idx])
            rv_err.append(t_f['e_RV'][idx])
            G.append(t_f['Gmag'][idx])
            G_err.append(t_f['e_Gmag'][idx])
            Grp.append(t_f['RPmag'][idx])
            Grp_err.append(t_f['e_RPmag'][idx])
            Gbp.append(t_f['BPmag'][idx])
            Gbp_err.append(t_f['e_BPmag'][idx])
            sourceID.append(t_f['DR3Name'][idx])
            ruwe.append(t_f['RUWE'][idx])
    t_f = t.copy()
    add_colnames = ['ra_gaia_J2000', 'ra_gaia_J2000_err', 'dec_gaia_J2000', 'dec_gaia_J2000_err', 
                    'ra_gaia_J2016', 'ra_gaia_J2016_err', 'dec_gaia_J2016', 'dec_gaia_J2016_err',
                    'pma', 'pma_err', 'pmd', 'pmd_err', 'plx', 'plx_err', 'rv', 'rv_err', 
                    'G', 'G_err', 'Grp', 'Grp_err', 'Gbp', 'Gbp_err', 'RUWE', 'sourceID']
    t_f.add_columns([ra_j2000, ra_j2000_err, dec_j2000, dec_j2000_err, 
                     ra_j2016, ra_j2016_err, dec_j2016, dec_j2016_err, 
                     pma, pma_err, pmd, pmd_err, plx, plx_err, rv, rv_err, 
                     G, G_err, Grp, Grp_err, Gbp, Gbp_err, ruwe, sourceID], 
                   names=add_colnames)
    t_f['plx'][t_f['plx'] < 0] = 0

    for cn in add_colnames[4:]:
        if hasattr(t_f[cn].data, 'mask'):
            t_f[cn][t_f[cn].data.mask] = 0
    return t_f


def xmatch_2mass_psc(t, radius, colRA, colDec):
    '''
    Cross match the table with 2MASS point source catalog.
    
    Parameters
    ----------
    t : Astropy Table
        The table of targets.
    radius : float
        The cross match radius, units: arcsec.
    colRA : string
        The column name of the RA.
    colDec : string
        The column name of the Dec.
    vizier_code : string
        The vizieR code of the 2MASS PSC table.
        
    Returns
    -------
    t_f : Astropy Table
        The table of cross matched results.
    '''
    assert XMatch is not None, 'The astroquery is not installed!'

    for cn in ['RAJ2000', 'DEJ2000', 'Jmag', 'Hmag', 'Kmag']:
        assert cn not in t.colnames, 'The input table has {} as a column!'.format(cn)
    
    t_o = XMatch.query(cat1=t,
                       cat2='vizier:II/246/out',
                       max_distance=radius*u.arcsec, colRA1=colRA, colDec1=colDec)
    ra_j2000 = []
    dec_j2000 = []
    jmag = []
    hmag = []
    kmag = []
    sourceID = []
    for loop in range(len(t)):
        fltr = np.isclose(t_o[colRA], t[colRA][loop]) & np.isclose(t_o[colDec], t[colDec][loop])
        if np.sum(fltr) == 0:
            ra_j2000.append(np.nan)
            dec_j2000.append(np.nan)
            jmag.append(np.nan)
            hmag.append(np.nan)
            kmag.append(np.nan)
            sourceID.append(np.nan)
        else:
            t_f = t_o[fltr]
            idx = np.argmin(t_f['angDist'])
            ra_j2000.append(t_f['RAJ2000'][idx])
            dec_j2000.append(t_f['DEJ2000'][idx])
            jmag.append(t_f['Jmag'][idx])
            hmag.append(t_f['Hmag'][idx])
            kmag.append(t_f['Kmag'][idx])
            sourceID.append(t_f['2MASS'][idx])
    t_f = t.copy()
    t_f.add_columns([ra_j2000, dec_j2000, jmag, hmag, kmag, sourceID], 
                   names=['RAJ2000', 'DECJ2000', 'Jmag', 'Hmag', 'Kmag', 'sourceID'])
    return t_f

    
def search_simbad(name):
    '''
    Find the target name from Simbad.
    
    Parameters
    ----------
    name : str
        Name of the target.
    
    Returns
    -------
    ra_hms, dec_hms  : str
        Coordinates, (HH:MM:SS, DD:MM:SS).
    pma, pmd : str
        Proper motion, units: arcsec / year
    '''
    assert Simbad is not None, 'The astroquery is not installed!'

    result_table = Simbad.query_object(name)
    
    if result_table is None:
        raise Exception('The target ({}) is not found!'.format(name))
    
    ra = result_table['RA'][0]
    dec = result_table['DEC'][0]
    pma = result_table['PMRA'][0] * 1e-3
    pmd = result_table['PMDEC'][0] * 1e-3
    c = read_coordinate(ra, dec)
    ra_hms, dec_dms = get_coord_colon(c)
    return ra_hms, dec_dms, pma, pmd
    

def get_dit(k, res='med', pol='split', tel='UT', wide=False):
    '''
    Get the optimal DIT for a given target magnitude.

    Parameters
    ----------
    k : float
        The magnitude of the target.
    res : string (default: 'med')
        The resolution of the instrument, either 'med' or 'low'.
    pol : string (default: 'split')
        The polarization mode, either 'split' or 'combined'.
    tel : string (default: 'UT')
        The telescope, either 'UT' or 'AT'.
    wide : bool (default: False)
        Use the G-wide mode if True.

    Returns
    -------
    dit : float
        The optimal DIT.
    '''
    opt_med = {
        1: 5.5,
        3: 7.0,
        10: 8.0,
        30: 11,
    }
    
    opt_low = {
        0.3: 7.5,
        1: 9.0,
        3: 10.5,
        10: 11.5,
        30: 15.0,
    }

    if res == 'med':
        opt_dict = opt_med
        k_bright_limit = 4
    elif res == 'low':
        opt_dict = opt_low
        k_bright_limit = 7
    else:
        raise ValueError('The res should be either med or low!')

    if pol == 'split':
        pass
    elif pol == 'combined':
        k = k - 0.8
    else:
        raise ValueError('The pol should be either split or combined!')
    
    if tel == 'UT':
        pass
    elif tel == 'AT':
        k = k + 2.5
    else:
        raise ValueError('The tel should be either UT or AT!')
    
    if wide:
        k = k + 1
    
    if k < k_bright_limit:
        raise ValueError(f'The target is too bright for {res}, {pol}, {tel}!')

    dit = np.array(list(opt_dict.keys()))
    mags = np.array(list(opt_dict.values()))

    idx = bisect.bisect(mags, k)

    if idx == len(mags):
        idx -= 1

    if dit[idx] == 0.3:
        return 0.3
    else:
        return int(dit[idx])