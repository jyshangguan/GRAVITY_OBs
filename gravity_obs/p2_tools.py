import numpy as np
from datetime import datetime

try:
    import p2api
except ImportError:
    p2api = None

now = datetime.now()
dt_string = now.strftime("%Y-%m-%dT%H:%M:%S")

# sanitizer for names -> replace spaces with underscores
sanitize_name = lambda s: f"{s}".replace(' ', '_')

all = ['check_item_exist', 'create_OB', 'ob_add_target', 'ob_add_description', 
       'p2api_GRAVITY']


def get_dit_med(ksc, tel='UT', verbose=True):
    '''
    Find the proper dit and ndit for MEDIUM (SPLIT).

    Parameters
    ----------
    ksc : float
        The Ks magnitude of the source.
    tel : string
        The telescope name, 'UT' or 'AT'.
    verbose : bool
        Print the warning if True.
    
    Returns
    -------
    dit, ndit : float
        The DIT and NDIT.
    '''
    if tel == 'AT':
        ksc = ksc + 3

    if ksc < 6:
        dit, ndit = 1, 120
        
        if verbose:
            print(f'The source may be too bright ({ksc}) for MEDIUM!')

    elif (ksc >= 6) & (ksc < 7.5):
        dit, ndit = 3, 40
    elif (ksc >= 7.5) & (ksc < 8.5):
        dit, ndit = 10, 12
    else:
        dit, ndit = 30, 4
    
    return dit, ndit


def get_dit_low(ksc, tel='UT', verbose=True):
    '''
    Find the proper dit and ndit for LOW (SPLIT).

    Parameters
    ----------
    ksc : float
        The Ks magnitude of the source.
    tel : string
        The telescope name, 'UT' or 'AT'.
    verbose : bool
        Print the warning if True.
    
    Returns
    -------
    dit, ndit : float
        The DIT and NDIT.
    '''
    if tel == 'AT':
        ksc = ksc + 3

    if ksc < 8.5:
        dit, ndit = 0.3, 120
        
        if verbose:
            print(f'The source may be too bright ({ksc}) for MEDIUM!')
    
    elif (ksc >= 8.5) & (ksc < 10):
        dit, ndit = 1, 120

    elif (ksc >= 10) & (ksc < 11):
        dit, ndit = 3, 40
    elif (ksc >= 11) & (ksc < 12.0):
        dit, ndit = 10, 36
    else:
        dit, ndit = 30, 12
    
    return dit, ndit



def check_item_exist(name, itemType, runContainerId, api):
    '''
    Check whether folder is in the container.
    
    Parameters
    ----------
    name : string
        The name of the item.
    itemType : string
        The type of the item, 'OB' or 'Folder'.
    runContainerId : int
        The ID of the container.
    api : p2api
    
    Returns
    -------
    it : dict 
        The information of the iterm if the item exists.  False, otherwise.
    '''
    items, _ = api.getItems(runContainerId)
    
    for it in items:
        if (it['itemType'] == itemType) & (it['name'] == name):
            return it
        
    return False


def create_OB(ob_name, folder_name, runContainerId, api):
    '''
    Create an OB or replace an OB in a folder.
    
    Parameters
    ----------
    ob_name : string
        The OB name.
    folder_name : string
        The folder name.
    runContainerId : int
        The ID of the container.
    api : p2api
    
    Returns
    -------
    ob : dict
        The OB information.
    '''
    folder_info = check_item_exist(folder_name, 'Folder', runContainerId, api)

    if folder_info:
        folderId = folder_info['containerId']
    else:
        folder, folderVersion = api.createFolder(runContainerId, folder_name)
        folderId = folder['containerId']
    
    ob_info = check_item_exist(ob_name, 'OB', folderId, api)
    if ob_info:
        obId = ob_info['obId']
        ob, obVersion = api.getOB(obId)
        api.deleteOB(obId, obVersion)
    ob, obVersion = api.createOB(folderId, ob_name)
    return ob, obVersion
    
    
def ob_add_target(ob, name, ra, dec, pma, pmd):
    '''
    Add the target information.
    
    Parameters
    ----------
    ob : dict
        The OB information.
    name : string
        The target name.
    ra : string
        The R.A. in hour angle, HH:MM:SS.SSS.
    dec : string
        The Decl. in degree, DD:MM:SS.SSS.
    pma : float
        The proper motion of R.A., in milliarcsec.
    pmd : float
        The proper motion of Decl., in milliarcsec.
    parallax : float
        The parallax, in milliarcsec.
    
    Returns
    -------
    ob : dict
        The OB information.
    '''
    ob['target']['name'] = sanitize_name(name)
    ob['target']['ra'] = ra
    ob['target']['dec'] = dec
    ob['target']['properMotionRa'] = np.round(pma * 1e-3, 6)
    ob['target']['properMotionDec'] = np.round(pmd * 1e-3, 6)
    return ob


def ob_add_description(ob, name, userComments):
    '''
    Parameters
    ----------
    ob : dict
        The OB information.
    name : string
        The observing description name.
    userComments : string
        The user comments.
    
    Returns
    -------
    ob : dict
        The OB information.
    '''
    ob['obsDescription']['name'] = name
    ob['obsDescription']['userComments'] = userComments
    return ob
    

class p2api_GRAVITY(object):
    '''
    Create OB on P2 for GRAVITY.
    '''
    def __init__(self, prog_id, username, password, root_container_id=None, no_warning=False):
        '''
        Initiate the API.
        
        Parameters
        ----------
        prog_id : string 
            Program ID, e.g. "109.23CR.001".
        username : string
            User name.
        password : string
            Password.
        '''
        assert p2api is not None, 'Cannot find p2api!'

        api = p2api.ApiConnection('production', username, password)
        runList = api.getRuns()[0]
        pidList = [r['progId'] for r in runList]
        
        if prog_id in pidList:
            nidx = pidList.index(prog_id)
        else:
            raise ValueError('Cannot find {0} in {1}!'.format(prog_id, pidList))
        
        self.api = api
        self.prog_id = prog_id
        self._run = runList[nidx]
        self._runContainerId = runList[nidx]['containerId']
        self.rootDict = dict(Root=(runList[nidx], 0))
        if root_container_id is None:
            self._rootContainterId = self._runContainerId
        else:
            self._rootContainterId = _rootContainterId
        
        self.update_content()
        
        
    def add_GRAVITY_dual_offaxis_acq(self, name, folder_name=None, 
                                    ft_mode='AUTO', met_mode='ON', sc_pick = 'T', 
                                    sc_name='Name', sc_kmag=0, sc_hmag=0, 
                                    sc_d=0,  sc_vis=1, sobj_x=0, sobj_y=0,
                                    ft_name='Name', ft_kmag=0, ft_hmag=0, 
                                    ft_d=0, ft_vis=1, ft_plx=0,
                                    spec_res='MED',  sc_pol='OUT', 
                                    ao_mode='LGS_VIS', ngs_source='FTS', ngs_name='Name',
                                    ngs_alpha='00:00:00.000', ngs_delta='00:00:00.000', 
                                    ngs_plx=0, ngs_pma=0, ngs_pmd=0, ngs_epoch=2000, ngs_mag=0, 
                                    lgs_source='SCIENCE', lgs_name='Name',
                                    lgs_alpha='00:00:00.000', lgs_delta='00:00:00.000', 
                                    lgs_plx=0, lgs_pma=0, lgs_pmd=0, lgs_epoch=2000, 
                                    baseline=['astrometric'], 
                                    vltitype=['snapshot']):
        '''
        Add acquisition template: GRAVITY_dual_acq

        Parameters
        ----------
        name : string
            The OB name.
        folder_name : string
            The folder name.
        ft_mode : string
            The fringe tracker mode (SEQ.FT.MODE), 'AUTO', 1, 2, 7 or 9.
        met_mode : string
            The metrology mode (SEQ.MET.MODE), 'ON', 'FAINT', or 'OFF'.
        picksc : string
            Whether to pick the SC source manually (SEQ.SC.PICK), 'T', 'A', or 'F'.
        ft_name : string
            The fringe tracker name (TEL.TARG.NAME).
        ft_kmag : float
            The Ks magnitude of the fringe tracker (TEL.TARG.KMAG).
        ft_hmag : float
            The H magnitude of the fringe tracker (TEL.TARG.HMAG).
        ft_d : float
            The diameter of the fringe tracker source (TEL.TARG.DIAMETER), in mas.
        ft_vis : float
            The visibility of the fringe tracker source (TEL.TARG.VIS).
        sc_name : string
            The science object name (SEQ.INS.SOBJ.NAME).
        sc_kmag : float
            The Ks magnitude of the science object (SEQ.INS.SOBJ.MAG.K).
        sc_d : float
            The diameter of the science object (SEQ.INS.SOBJ.DIAMETER), in mas.
        sc_vis : float
            The visibility of the science object (SEQ.INS.SOBJ.VIS).
        sobj_x : float
            The x offset of the sc from the ft (SEQ.INS.SOBJ.X), in mas.
        sobj_y : float
            The y offset of the sc from the ft (SEQ.INS.SOBJ.Y), in mas.
        ft_plx : float
            The parallax of the fringe tracker (TEL.TARG.PARALLAX), in arcsec.
        spec_res : string
            The spectral resolution (INS.SPEC.RES), 'LOW', 'MED', or 'HIGH'.
        sc_pol : string
            The polarization of the science object (INS.SPEC.POL), 'IN' or 'OUT'.
        ao_type : string
            The adaptive optics type (COU.AO.TYPE), 'NGS_VIS', 'NGS_IR', 'LGS_VIS', 'LGS_IR'.
        ngs_source : string
            The guide star source (COU.NGS.SOURCE), 'FTS', 'SCIENCE', or 'SETUPFILE'.
        ngs_alpha : string
            The R.A. of the guide star (COU.NGS.ALPHA), in HH:MM:SS.SSS, 
            only if ngs_source is 'SETUPFILE'.
        ngs_delta : string
            The Decl. of the guide star (COU.NGS.DELTA), in DD:MM:SS.SSS, 
            only if ngs_source is 'SETUPFILE'.
        ngs_plx : float
            The parallax of the guide star (COU.NGS.PARALLAX), in arcsec, 
            only if ngs_source is 'SETUPFILE'.
        ngs_pma : float
            The proper motion of R.A. of the guide star (COU.NGS.PMA), in arcsec/yr, 
            only if ngs_source is 'SETUPFILE'.
        ngs_pmd : float
            The proper motion of Decl. of the guide star (COU.NGS.PMD), in arcsec/yr,
            only if ngs_source is 'SETUPFILE'.
        ngs_epoch : float
            The epoch of the guide star coordinate (COU.NGS.EPOCH), in year, better only use 2000.
        ngs_mag : float
            The magnitude of the guide star (COU.NGS.MAG) in the optical or IR depending on 
            the ngs_type.
        lgs_source : string
            The laser guide star source (COU.LGS.SOURCE), 'SCIENCE', 'NGS', 'FTS', or 'SETUPFILE'.
        lgs_alpha : string
            The R.A. of the laser guide star (COU.LGS.ALPHA), in HH:MM:SS.SSS,
            only if lgs_source is 'SETUPFILE'.
        lgs_delta : string
            The Decl. of the laser guide star (COU.LGS.DELTA), in DD:MM:SS.SSS,
            only if lgs_source is 'SETUPFILE'.
        lgs_plx : float
            The parallax of the laser guide star (COU.LGS.PARALLAX), in arcsec,
            only if lgs_source is 'SETUPFILE'.
        lgs_pma : float
            The proper motion of R.A. of the laser guide star (COU.LGS.PMA), in arcsec/yr,
            only if lgs_source is 'SETUPFILE'.      
        lgs_pmd : float
            The proper motion of Decl. of the laser guide star (COU.LGS.PMD), in arcsec/yr,
            only if lgs_source is 'SETUPFILE'.
        lgs_epoch : float
            The epoch of the laser guide star coordinate (COU.LGS.EPOCH), in year, better only use 2000..
        baseline : list
            The baseline type, 'small', 'large', or 'UTs'.
        vltitype : list
            The VLTI type, 'snapshot', 'imaging', 'time-series', or 'astrometry'.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
        
        obid = ob['obId']
        try:
            acqTpl, acqTplVersion = api.createTemplate(obid, 'GRAVITY_dual_offaxis_acq')
        except p2api.P2Error as e:
            raise p2api.P2Error(e)

        sc_name  = sanitize_name(sc_name)
        ngs_name = sanitize_name(ngs_name)
        lgs_name = sanitize_name(lgs_name)

        pdict = {
            'SEQ.FT.MODE': ft_mode,
            'SEQ.MET.MODE': met_mode,
            'SEQ.PICKSC': sc_pick,
            'SEQ.INS.SOBJ.NAME': sc_name,
            'SEQ.INS.SOBJ.MAG.K': sc_kmag,
            'SEQ.INS.SOBJ.MAG.H': sc_hmag,
            'SEQ.INS.SOBJ.DIAMETER': sc_d,
            'SEQ.INS.SOBJ.VIS': sc_vis,
            'SEQ.INS.SOBJ.X': sobj_x,
            'SEQ.INS.SOBJ.Y': sobj_y,
            'TEL.TARG.NAME': ft_name,
            'TEL.TARG.PARALLAX': ft_plx,
            'TEL.TARG.MAG.K': ft_kmag,
            'TEL.TARG.MAG.H': ft_hmag,
            'TEL.TARG.VIS': ft_vis,
            'TEL.TARG.DIAMETER': ft_d,
            'INS.SPEC.RES': spec_res,
            'INS.SPEC.POL': sc_pol,
            'COU.AO.TYPE': ao_mode,
            'COU.NGS.SOURCE': ngs_source,
            'COU.NGS.NAME': ngs_name,
            'COU.NGS.ALPHA': ngs_alpha,
            'COU.NGS.DELTA': ngs_delta,
            'COU.NGS.PARALLAX': ngs_plx,
            'COU.NGS.PMA': ngs_pma,
            'COU.NGS.PMD': ngs_pmd,
            'COU.NGS.EPOCH': ngs_epoch,
            'COU.NGS.MAG': ngs_mag,
            'COU.LGS.SOURCE': lgs_source,
            'COU.LGS.NAME': lgs_name,
            'COU.LGS.ALPHA': lgs_alpha,
            'COU.LGS.DELTA': lgs_delta,
            'COU.LGS.PARALLAX': lgs_plx,
            'COU.LGS.PMA': lgs_pma,
            'COU.LGS.PMD': lgs_pmd,
            'COU.LGS.EPOCH': lgs_epoch,
            'ISS.BASELINE': baseline,
            'ISS.VLTITYPE': vltitype
        }
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        

    def add_GRAVITY_dual_onaxis_acq(self, name, folder_name=None, 
                                    ft_mode='AUTO', met_mode='ON', sc_pick = 'F', 
                                    sc_name='Name', sc_kmag=0, sc_hmag=0, 
                                    sc_d=0,  sc_vis=1, sobj_x=0, sobj_y=0,
                                    ft_name='Name',ft_kmag=0, ft_hmag=0, 
                                    ft_d=0, ft_vis=1, ft_plx=0,
                                    spec_res='MED', sc_pol='OUT', 
                                    ao_mode='LGS_VIS', ngs_source='FTS', ngs_name='Name',
                                    ngs_alpha='00:00:00.000', ngs_delta='00:00:00.000', 
                                    ngs_plx=0, ngs_pma=0, ngs_pmd=0, ngs_epoch=2000, ngs_mag=0, 
                                    lgs_source='SCIENCE', lgs_name='Name',
                                    lgs_alpha='00:00:00.000', lgs_delta='00:00:00.000', 
                                    lgs_plx=0, lgs_pma=0, lgs_pmd=0, lgs_epoch=2000,
                                    baseline=['astrometric'], 
                                    vltitype=['snapshot']):
        '''
        Add acquisition template: GRAVITY_dual_acq

        Parameters
        ----------
        name : string
            The OB name.
        folder_name : string
            The folder name.
        ft_mode : string
            The fringe tracker mode (COU.FT.MODE), 'AUTO', 1, 2, 7 or 9.
        met_mode : string
            The metrology mode (COU.MET.MODE)   , 'ON', 'FAINT', or 'OFF'.
        sc_pick : string
            Whether to pick the SC source manually (SEQ.PICKSC), 'T', 'A', or 'F'.
        ft_name : string
            The fringe tracker name (TEL.TARG.NAME).
        ft_kmag : float
            The Ks magnitude of the fringe tracker (TEL.TARG.MAG.K).
        ft_hmag : float
            The H magnitude of the fringe tracker (TEL.TARG.MAG.H).
        ft_d : float
            The diameter of the fringe tracker source (TEL.TARG.DIAMETER), in mas.
        ft_plx : float
            The parallax of the fringe tracker (TEL.TARG.PARALLAX), in arcsec.
        ft_vis : float
            The visibility of the fringe tracker source (TEL.TARG.VIS).
        sc_name : string
            The science object name (SEQ.INS.SOBJ.NAME).
        sc_kmag : float
            The Ks magnitude of the science object (SEQ.INS.SOBJ.MAG.K).
        sc_hmag : float
            The H magnitude of the science object (SEQ.INS.SOBJ.MAG.H).
        sc_d : float
            The diameter of the science object (SEQ.INS.SOBJ.DIAMETER), in mas.
        sc_vis : float
            The visibility of the science object (SEQ.INS.SOBJ.VIS).
        sobj_x : float
            The x offset of the sc from the ft (SEQ.INS.SOBJ.X), in mas.
        sobj_y : float
            The y offset of the sc from the ft (SEQ.INS.SOBJ.Y), in mas.
        spec_res : string
            The spectral resolution (INS.SPEC.RES), 'LOW', 'MED', or 'HIGH'.
        sc_pol : string
            The polarization of the science object (INS.SPEC.POL), 'IN' or 'OUT'.
        ao_type : string
            The adaptive optics type (COU.AO.TYPE), 'ADAPT_OPT', 'ADAPT_OPT_TCCD', or 'IR_AO_OFFAXIS'.
        ngs_source : string
            The guide star source (COU.NGS.SOURCE), 'FTS', 'SCIENCE', or 'SETUPFILE'.
        ngs_alpha : string
            The R.A. of the guide star (COU.NGS.ALPHA), in HH:MM:SS.SSS,
            only if ngs_source is 'SETUPFILE'.
        ngs_delta : string
            The Decl. of the guide star (COU.NGS.DELTA), in DD:MM:SS.SSS,
            only if ngs_source is 'SETUPFILE'.
        ngs_plx : float
            The parallax of the guide star (COU.NGS.PARALLAX), in arcsec,
            only if ngs_source is 'SETUPFILE'.
        ngs_pma : float
            The proper motion of R.A. of the guide star (COU.NGS.PMA), in arcsec/yr,
            only if ngs_source is 'SETUPFILE'.
        ngs_pmd : float
            The proper motion of Decl. of the guide star (COU.NGS.PMD), in arcsec/yr,
            only if ngs_source is 'SETUPFILE'.
        ngs_epoch : float
            The epoch of the guide star coordinate (COU.NGS.EPOCH), in year, better only use 2000.
        ngs_mag : float
            The magnitude of the guide star in the optical or IR depending on
            the ao_type.
        lgs_source : string
            The laser guide star source (COU.LGS.SOURCE), 'SCIENCE', 'NGS', 'FTS', or 'SETUPFILE'.
        lgs_alpha : string
            The R.A. of the laser guide star (COU.LGS.ALPHA), in HH:MM:SS.SSS,
            only if lgs_source is 'SETUPFILE'.  
        lgs_delta : string
            The Decl. of the laser guide star (COU.LGS.DELTA), in DD:MM:SS.SSS,
            only if lgs_source is 'SETUPFILE'.
        lgs_plx : float
            The parallax of the laser guide star (COU.LGS.PARALLAX), in arcsec,
            only if lgs_source is 'SETUPFILE'.
        lgs_pma : float
            The proper motion of R.A. of the laser guide star (COU.LGS.PMA), in arcsec/yr,
            only if lgs_source is 'SETUPFILE'.
        lgs_pmd : float
            The proper motion of Decl. of the laser guide star (COU.LGS.PMD), in arcsec/yr,
            only if lgs_source is 'SETUPFILE'.                  
        lgs_epoch : float
            The epoch of the laser guide star coordinate (COU.LGS.EPOCH), in year            
        baseline : list
            The baseline type, 'small', 'large', or 'UTs'.
        vltitype : list
            The VLTI type, 'snapshot', 'imaging', 'time-series', or 'astrometry'.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
        
        obid = ob['obId']
        try:
            acqTpl, acqTplVersion = api.createTemplate(obid, 'GRAVITY_dual_onaxis_acq')
        except p2api.P2Error as e:
            raise p2api.P2Error(e)

        sc_name  = sanitize_name(sc_name)
        ngs_name = sanitize_name(ngs_name)
        lgs_name = sanitize_name(lgs_name)

        pdict = {
            'SEQ.FT.MODE': ft_mode,
            'SEQ.MET.MODE': met_mode,
            'SEQ.PICKSC': sc_pick,
            'SEQ.INS.SOBJ.NAME': sc_name,
            'SEQ.INS.SOBJ.MAG.K': sc_kmag,
            'SEQ.INS.SOBJ.MAG.H': sc_hmag,
            'SEQ.INS.SOBJ.DIAMETER': sc_d,
            'SEQ.INS.SOBJ.VIS': sc_vis,
            'SEQ.INS.SOBJ.X': sobj_x,
            'SEQ.INS.SOBJ.Y': sobj_y,
            'TEL.TARG.NAME': ft_name,
            'TEL.TARG.PARALLAX': ft_plx,
            'TEL.TARG.MAG.K': ft_kmag,
            'TEL.TARG.MAG.H': ft_hmag,
            'TEL.TARG.VIS': ft_vis,
            'TEL.TARG.DIAMETER': ft_d,
            'INS.SPEC.RES': spec_res,
            'INS.SPEC.POL': sc_pol,
            'COU.AO.TYPE': ao_mode,
            'COU.NGS.SOURCE': ngs_source,
            'COU.NGS.NAME': ngs_name,
            'COU.NGS.ALPHA': ngs_alpha,
            'COU.NGS.DELTA': ngs_delta,
            'COU.NGS.PARALLAX': ngs_plx,
            'COU.NGS.PMA': ngs_pma,
            'COU.NGS.PMD': ngs_pmd,
            'COU.NGS.EPOCH': ngs_epoch,
            'COU.NGS.MAG': ngs_mag,
            'COU.LGS.GSSOURCE': lgs_source,
            'COU.LGS.NAME': lgs_name,
            'COU.LGS.ALPHA': lgs_alpha,
            'COU.LGS.DELTA': lgs_delta,
            'COU.LGS.PARALLAX': lgs_plx,
            'COU.LGS.PMA': lgs_pma,
            'COU.LGS.PMD': lgs_pmd,
            'COU.LGS.EPOCH': lgs_epoch,
            'ISS.BASELINE': baseline,
            'ISS.VLTITYPE': vltitype
        }
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
     

    def add_GRAVITY_dual_obs_calibrator(self, name, folder_name=None, dit=0.3, ndit_obj=32, 
                                        ndit_sky=32, hwpoff=[0], obsseq='O S', sky_x=2000, 
                                        sky_y=2000):
        '''
        Add acquisition template: GRAVITY_single_acq

        Parameters
        ----------
        name : string
            The OB name.
        folder_name : string
            The folder name.
        dit : float
            The DIT.
        ndit_obj : int
            The NDIT for the object.
        ndit_sky : int
            The NDIT for the sky.
        hwpoff : list
            The HWP offset, in degree.
        obsseq : string
            The observation sequence, e.g. 'O O S'.
        sky_x : float
            The x offset of the sky from the object, in mas.
        sky_y : float
            The y offset of the sky from the object, in mas.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
        
        obid = ob['obId']
        try:
            acqTpl, acqTplVersion = api.createTemplate(obid, 'GRAVITY_dual_obs_calibrator')
        except p2api.P2Error as e:
            raise p2api.P2Error(e)
        
        #pars = acqTpl['parameters']
        #pList = [dit, ndit_obj, ndit_sky, hwpoff, obsseq, sky_x, sky_y]
        #pdict = {}
        #for loop, pv in enumerate(pList):
        #    pdict[pars[loop]['name']] = pv

        pdict = {
            'DET2.DIT': dit,
            'DET2.NDIT.OBJECT': ndit_obj,
            'DET2.NDIT.SKY': ndit_sky,
            'SEQ.SKY.X': sky_x, 
            'SEQ.SKY.Y': sky_y,
            'SEQ.HWPOFF': hwpoff,
            'SEQ.OBSSEQ': obsseq,
        }

        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        
        
    def add_GRAVITY_dual_obs_exp(self, name, folder_name=None, dit=0.3, ndit_obj=32, 
                                 ndit_sky=32, hwpoff=[0], obsseq='O S', reloff_x=[0], 
                                 reloff_y=[0], sky_x=2000, sky_y=2000):
        '''
        Add acquisition template: GRAVITY_single_acq

        Parameters
        ----------
        name : string
            The OB name.
        folder_name : string
            The folder name.
        dit : float
            The DIT.
        ndit_obj : int
            The NDIT for the object.
        ndit_sky : int
            The NDIT for the sky.
        hwpoff : list
            The HWP offset, in degree.
        obsseq : string
            The observation sequence, e.g. 'O O S'.
        reloff_x : list
            The x offset of the object from the current position, in mas.
        reloff_y : list
            The y offset of the object from the current position, in mas.
        sky_x : float
            The x offset of the sky from the object, in mas.
        sky_y : float
            The y offset of the sky from the object, in mas.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
        
        obid = ob['obId']
        try:
            acqTpl, acqTplVersion = api.createTemplate(obid, 'GRAVITY_dual_obs_exp')
        except p2api.P2Error as e:
            raise p2api.P2Error(e)
        
        pdict = {
            'DET2.DIT': dit,
            'DET2.NDIT.OBJECT': ndit_obj,
            'DET2.NDIT.SKY': ndit_sky,
            'SEQ.SKY.X': sky_x, 
            'SEQ.SKY.Y': sky_y,
            'SEQ.HWPOFF': hwpoff,
            'SEQ.OBSSEQ': obsseq,
            'SEQ.RELOFF.X': reloff_x,
            'SEQ.RELOFF.Y': reloff_y
        }
        
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        
        
    def add_GRAVITY_dual_obs_swap(self, name, folder_name=None, ft_mode='AUTO'):
        '''
        Add acquisition template: GRAVITY_dual_obs_swap

        Parameters
        ----------
        name : string
            The OB name.
        folder_name : string
            The folder name.
        ft_mode : string
            The fringe tracker mode, 'AUTO', 1, 2, 7 or 9.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
        
        obid = ob['obId']
        try:
            acqTpl, acqTplVersion = api.createTemplate(obid, 'GRAVITY_dual_obs_swap')
        except p2api.P2Error as e:
            raise p2api.P2Error(e)
            
        pars = acqTpl['parameters']
        pList = [ft_mode]
        pdict = {}
        for loop, pv in enumerate(pList):
            pdict[pars[loop]['name']] = pv
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        
        
    def add_GRAVITY_dual_wide_acq(self, name, folder_name=None, 
                                  ft_mode='AUTO', met_mode='ON',
                                  sc_name='Name', sc_kmag=0, sc_hmag=0, 
                                  sc_d=0, sc_plx = 0, sc_vis=1,
                                  ft_name='Name',  ft_epoch=2000,
                                  ft_alpha=None, ft_delta=None, ft_plx=0, ft_pma=0, ft_pmd=0, 
                                  ft_kmag=0, ft_hmag=0, ft_d=0, ft_vis=1, 
                                  spec_res='MED', sc_pol='OUT', 
                                  ao_mode='LGS_VIS', ngs_source='FTS', ngs_name='Name',
                                  ngs_alpha='00:00:00.000', ngs_delta='00:00:00.000', 
                                  ngs_plx=0, ngs_pma=0, ngs_pmd=0, ngs_epoch=2000, ngs_mag=0, 
                                  lgs_source='FTS', lgs_name='Name',
                                  lgs_alpha='00:00:00.000', lgs_delta='00:00:00.000', 
                                  lgs_plx=0, lgs_pma=0, lgs_pmd=0, lgs_epoch=2000,
                                  baseline=['astrometric'], 
                                  vltitype=['snapshot']
                                  ):
        '''
        Add acquisition template: GRAVITY_dual_wide_acq

        Parameters
        ----------
        name : string
            The OB name.
        folder_name : string
            The folder name.
        ft_mode : string
            The fringe tracker mode (SEQ.FT.MODE), 'AUTO', 1, 2, 7 or 9.
        met_mode : string
            The metrology mode (SEQ.MT.MODE), 'ON', 'FAINT', or 'OFF'.
        sc_name : string
            The science object name (TEL.TARG.NAME).
        sc_kmag : float
            The Ks magnitude of the science object (TEL.TARG.MAG.K).
        sc_hmag : float
            The H magnitude of the science object (TEL.TARG.MAG.H).
        sc_d : float
            The diameter of the science object (TEL.TARG.DIAMETER), in mas.
        sc_vis : float
            The visibility of the science object (TEL.TARG.VIS).
        sc_plx : float
            The parallax of the science object (TEL.TARG.PARALLAX), in arcsec
        ft_name : string
            The fringe tracker name (COU.FITS.NAME).
        ft_kmag : float
            The Ks magnitude of the fringe tracker (COU.FITS.MAG.K).
        ft_hmag : float
            The H magnitude of the fringe tracker (COU.FITS.MAG.H).
        ft_d : float
            The diameter of the fringe tracker source (COU.FITS.DIAMETER), in mas.
        ft_vis : float
            The visibility of the fringe tracker source (COU.FITS.VIS).
        ft_epoch : float
            The epoch of the FT source coordinate (COU.FITS.EPOCH), in year, better only use 2000.
        ft_alpha : string
            The R.A. of the fringe tracker (COU.FITS.ALPHA), in HH:MM:SS.SSS.
        ft_delta : string
            The Decl. of the fringe tracker (COU.FITS.DELTA), in DD:MM:SS.SSS.
        ft_plx : float
            The parallax of the fringe tracker (COU.FITS.PARALLAX), in arcsec.
        ft_pma : float
            The proper motion of R.A. of the fringe tracker (COU.FITS.PMA), in arcsec/yr.
        ft_pmd : float
            The proper motion of Decl. of the fringe tracker (COU.FITS.PMD), in arcsec/yr.
        spec_res : string
            The spectral resolution (INS.SPEC.RES), 'LOW', 'MED', or 'HIGH'.
        sc_pol : string
            The polarization of the science object (INS.SPEC.POL), 'IN' or 'OUT'.
        ngs_source : string
            The guide star source (COU.NGS.SOURCE), 'FTS', 'SCIENCE', or 'SETUPFILE'.
        ngs_alpha : string
            The R.A. of the guide star (COU.NGS.ALPHA), in HH:MM:SS.SSS,
            only if ngs_source is 'SETUPFILE'.
        ngs_delta : string
            The Decl. of the guide star (COU.NGS.DELTA), in DD:MM:SS.SSS,
            only if ngs_source is 'SETUPFILE'.
        ngs_plx : float
            The parallax of the guide star (COU.NGS.PARALLAX), in arcsec,
            only if ngs_source is 'SETUPFILE'.
        ngs_pma : float
            The proper motion of R.A. of the guide star (COU.NGS.PMA), in arcsec/yr,
            only if ngs_source is 'SETUPFILE'.
        ngs_pmd : float
            The proper motion of Decl. of the guide star (COU.NGS.PMD), in arcsec/yr,
            only if ngs_source is 'SETUPFILE'.
        ngs_epoch : float
            The epoch of the guide star coordinate (COU.NGS.EPOCH), in year, better only use 2000.
        ngs_mag : float
            The magnitude of the guide star (COU.NGS.MAG) in the optical or IR depending on
            the ngs_type.
        ao_type : string
            The adaptive optics type (COU.AO.TYPE), 'ADAPT_OPT', 'ADAPT_OPT_TCCD', or 'IR_AO_OFFAXIS'.
        lgs_source : string
            The laser guide star source (COU.LGS.SOURCE), 'SCIENCE', 'NGS', 'FTS', or 'SETUPFILE'.
        lgs_alpha : string
            The R.A. of the laser guide star (COU.LGS.ALPHA), in HH:MM:SS.SSS,
            only if lgs_source is 'SETUPFILE                
        lgs_delta : string
            The Decl. of the laser guide star (COU.LGS.DELTA), in DD:MM:SS.SSS,
            only if lgs_source is 'SETUPFILE'.
        lgs_plx : float
            The parallax of the laser guide star (COU.LGS.PARALLAX), in arcsec,
            only if lgs_source is 'SETUPFILE'.
        lgs_pma : float
            The proper motion of R.A. of the laser guide star (COU.LGS.PMA), in arcsec/yr,
            only if lgs_source is 'SETUPFILE'.
        lgs_pmd : float
            The proper motion of Decl. of the laser guide star (COU.LGS.PMD), in arcsec/yr,
            only if lgs_source is 'SETUPFILE'.                  
        lgs_epoch : float
            The epoch of the laser guide star coordinate (COU.LGS.EPOCH), in year, better only use 2000.  
        baseline : list
            The baseline type, 'small', 'large', or 'UTs'.
        vltitype : list
            The VLTI type, 'snapshot', 'imaging', 'time-series', or 'astrometry'.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
        
        obid = ob['obId']
        try:
            acqTpl, acqTplVersion = api.createTemplate(obid, 'GRAVITY_dual_wide_acq')
        except p2api.P2Error as e:
            raise p2api.P2Error(e)
        
        pars = acqTpl['parameters']

        sc_name = sanitize_name(sc_name)
        ft_name = sanitize_name(ft_name)
        ngs_name = sanitize_name(ngs_name)
        lgs_name = sanitize_name(lgs_name)

        pdict = {
            'SEQ.FT.MODE': ft_mode,
            'SEQ.MET.MODE': met_mode,
            'TEL.TARG.NAME': sc_name,
            'TEL.TARG.PARALLAX': sc_plx,
            'TEL.TARG.MAG.K': sc_kmag,
            'TEL.TARG.MAG.H': sc_hmag,
            'TEL.TARG.VIS': sc_vis,
            'TEL.TARG.DIAMETER': sc_d,
            'INS.SPEC.RES': spec_res,
            'INS.SPEC.POL': sc_pol,
            'COU.FTS.NAME': ft_name,
            'COU.FTS.ALPHA': ft_alpha,
            'COU.FTS.DELTA': ft_delta,
            'COU.FTS.PARALLAX': ft_plx,
            'COU.FTS.PMA': ft_pma,
            'COU.FTS.PMD': ft_pmd,
            'COU.FTS.EPOCH': ft_epoch,
            'COU.FTS.MAG.K': ft_kmag,
            'COU.FTS.MAG.H': ft_hmag,
            'COU.FTS.VIS': ft_vis,
            'COU.FTS.DIAMETER': ft_d,
            'COU.AO.TYPE': ao_mode,
            'COU.NGS.SOURCE': ngs_source,
            'COU.NGS.NAME': ngs_name,
            'COU.NGS.ALPHA': ngs_alpha,
            'COU.NGS.DELTA': ngs_delta,
            'COU.NGS.PARALLAX': ngs_plx,
            'COU.NGS.PMA': ngs_pma,
            'COU.NGS.PMD': ngs_pmd,
            'COU.NGS.EPOCH': ngs_epoch,
            'COU.NGS.MAG': ngs_mag,
            'COU.LGS.SOURCE': lgs_source,
            'COU.LGS.NAME': lgs_name,
            'COU.LGS.ALPHA': lgs_alpha,
            'COU.LGS.DELTA': lgs_delta,
            'COU.LGS.PARALLAX': lgs_plx,
            'COU.LGS.PMA': lgs_pma,
            'COU.LGS.PMD': lgs_pmd,
            'COU.LGS.EPOCH': lgs_epoch,
            'ISS.BASELINE': baseline,
            'ISS.VLTITYPE': vltitype,
        }
        
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        
        
    def add_GRAVITY_single_onaxis_acq(self, name, folder_name=None, ft_mode='AUTO', met_mode='ON', 
                                      sc_name='Name', sc_kmag=0, sc_hmag=0, sc_d=0, sc_vis=1, plx=0, 
                                      spec_res='MED', sc_pol='OUT', ao_mode='LGS_VIS', 
                                      ngs_source='SCIENCE', ngs_name='Name',
                                      ngs_alpha='00:00:00.000', ngs_delta='00:00:00.000', 
                                      ngs_plx=0, ngs_pma=0, ngs_pmd=0, ngs_epoch=2000, ngs_mag=0, 
                                      lgs_source='SCIENCE', lgs_name='Name',
                                      lgs_alpha='00:00:00.000', lgs_delta='00:00:00.000', 
                                      lgs_plx=0, lgs_pma=0, lgs_pmd=0, lgs_epoch=2000,
                                      baseline=['astrometric'], vltitype=['snapshot']):
        '''
        Add acquisition template: GRAVITY_single_acq

        Parameters
        ----------
        name : string
            The OB name.
        folder_name : string
            The folder name.
        ft_mode : string
            The fringe tracker mode (SEQ.FT.MODE), 'AUTO', 1, 2, 7 or 9. 
        met_mode : string
            The metrology mode (SEQ.MET.MODE), 'ON', 'FAINT', or 'OFF'.
        sc_name : string
            The science object name (TEL.TARG.NAME).
        sc_kmag : float
            The Ks magnitude of the science object (TEL.TARG.MAG.K).
        sc_hmag : float
            The H magnitude of the science object (TEL.TARG.MAG.H).
        sc_d : float
            The diameter of the science object, in mas (TEL.TARG.DIAMETER).
        sc_vis : float
            The visibility of the science object (TEL.TARG.VIS).
        plx : float
            The parallax of the science object, in arcsec (TEL.TARG.PARALLAX).
        spec_res : string
            The spectral resolution (INS.SPEC.RES), 'LOW', 'MED', or 'HIGH'.
        sc_pol : string
            The polarization of the science object (INS.SPEC.POL), 'IN' or 'OUT'.
        ao_mode : string
            The AO guide star mode (COU.AO.TYPE), 'NGS_VIS', 'NGS_IR', 'LGS_VIS', 'LGS_IR'.
        ngs_source : string
            The guide star source (COU.NGS.SOURCE), 'SCIENCE', or 'SETUPFILE'.
        ngs_alpha : string
            The R.A. of the guide star (COU.NGS.ALPHA), in HH:MM:SS.SSS,
            only if ngs_source is 'SETUPFILE'.
        ngs_delta : string
            The Decl. of the guide star (COU.NGS.DELTA), in DD:MM:SS.SSS,
            only if ngs_source is 'SETUPFILE'.
        ngs_plx : float
            The parallax of the guide star (COU.NGS.PARALLAX), in arcsec,
            only if ngs_source is 'SETUPFILE'.
        ngs_pma : float
            The proper motion of R.A. of the guide star (COU.NGS.PMA), in arcsec/yr,
            only if ngs_source is 'SETUPFILE'.
        ngs_pmd : float
            The proper motion of Decl. of the guide star (COU.NGS.PMD), in arcsec/yr,
            only if ngs_source is 'SETUPFILE'.
        ngs_epoch : float
            The epoch of the guide star coordinate , in year, better only use 2000.
        ngs_mag : float
            The magnitude of the guide star in the optical or IR (COU.NGS.MAG) depending on
            the ao_type.
        lgs_source : string
            The LGS source (COU.LGS.SOURCE), 'SCIENCE', 'NGS', 'FTS', or 'SETUPFILE'
        lgs_alpha : string
            The R.A. of the LGS (COU.LGS.ALPHA), in HH:MM:SS.SSS,
            only if lgs_source is 'SETUPFILE'.
        lgs_delta : string
            The Decl. of the LGS (COU.LGS.DELTA), in DD:MM:SS.SSS,
            only if lgs_source is 'SETUPFILE'.
        lgs_plx : float
            The parallax of the LGS (COU.LGS.PARALLAX), in arcsec,
            only if lgs_source is 'SETUPFILE'.
        lgs_pma : float
            The proper motion of R.A. of the LGS (COU.LGS.PMA), in arcsec/yr,
            only if lgs_source is 'SETUPFILE'.
        lgs_pmd : float
            The proper motion of Decl. of the LGS (COU.LGS.PMD), in arcsec/yr,
            only if lgs_source is 'SETUPFILE'.
        lgs_epoch : float
            The epoch of the LGS coordinate, in year, better only use 2000.
            only if lgs_source is 'SETUPFILE'.
        baseline : list
            The baseline type (ISS.BASELINE), 'small', 'large', or 'UTs'.
        vltitype : list
            The VLTI type (ISS.VLTITYPE), 'snapshot', 'imaging', 'time-series', or 'astrometry'.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
        
        obid = ob['obId']
        try:
            acqTpl, acqTplVersion = api.createTemplate(obid, 'GRAVITY_single_onaxis_acq')
        except p2api.P2Error as e:
            raise p2api.P2Error(e)
        
        sc_name = sanitize_name(sc_name)
        ngs_name = sanitize_name(ngs_name)
        lgs_name = sanitize_name(lgs_name)

        pdict = {
            'SEQ.FT.MODE': ft_mode,
            'SEQ.MET.MODE': met_mode,
            'TEL.TARG.NAME': sc_name,
            'TEL.TARG.MAG.K': sc_kmag,
            'TEL.TARG.MAG.H': sc_hmag,
            'TEL.TARG.DIAMETER': sc_d,
            'TEL.TARG.VIS': sc_vis,
            'TEL.TARG.PARALLAX': plx,
            'INS.SPEC.RES': spec_res,
            'INS.SPEC.POL': sc_pol,
            'COU.AO.TYPE': ao_mode,
            'COU.NGS.SOURCE': ngs_source,
            'COU.NGS.NAME': ngs_name,
            'COU.NGS.ALPHA': ngs_alpha,
            'COU.NGS.DELTA': ngs_delta,
            'COU.NGS.PARALLAX': ngs_plx,
            'COU.NGS.PMA': ngs_pma,
            'COU.NGS.PMD': ngs_pmd,
            'COU.NGS.EPOCH': ngs_epoch,
            'COU.NGS.MAG': ngs_mag,
            'COU.LGS.SOURCE': lgs_source,
            'COU.LGS.NAME': lgs_name,
            'COU.LGS.ALPHA': lgs_alpha,
            'COU.LGS.DELTA': lgs_delta,
            'COU.LGS.PARALLAX': lgs_plx,
            'COU.LGS.PMA': lgs_pma,
            'COU.LGS.PMD': lgs_pmd,
            'COU.LGS.EPOCH': lgs_epoch,
            'ISS.BASELINE': baseline,
            'ISS.VLTITYPE': vltitype
        }
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        
    
    def add_GRAVITY_single_offaxis_acq(self, name, folder_name=None, ft_mode='AUTO', met_mode='OFF', 
                                      sc_name='Name', sc_kmag=0, sc_hmag=0, sc_d=0, sc_vis=1, plx=0, 
                                      spec_res='MED', ft_pol='OUT', ao_mode='LGS_VIS', 
                                      ngs_source='SCIENCE', ngs_name='Name',
                                      ngs_alpha='00:00:00.000', ngs_delta='00:00:00.000', 
                                      ngs_plx=0, ngs_pma=0, ngs_pmd=0, ngs_epoch=2000, ngs_mag=0, 
                                      lgs_source='SCIENCE', lgs_name='Name',
                                      lgs_alpha='00:00:00.000', lgs_delta='00:00:00.000', 
                                      lgs_plx=0, lgs_pma=0, lgs_pmd=0, lgs_epoch=2000, 
                                      baseline=['astrometric'], vltitype=['snapshot']):
        '''
        Add acquisition template: GRAVITY_single_acq

        Parameters
        ----------
        name : string
            The OB name.
        folder_name : string
            The folder name.
        ft_mode : string
            The fringe tracker mode (SEQ.FT.MODE, 'AUTO', 1, 2, 7 or 9.
        met_mode : string
            The metrology mode (SEQ.MET.MODE), 'ON', 'FAINT', or 'OFF'.
        sc_name : string
            The science object name (TEL.TARG.NAME).
        sc_kmag : float
            The Ks magnitude of the science object (TEL.TARG.MAG.K).
        sc_hmag : float
            The H magnitude of the science object (TEL.TARG.MAG.H).
        sc_d : float
            The diameter of the science object (TEL.TARG.DIAMETER), in mas.
        sc_vis : float
            The visibility of the science object (TEL.TARG.VIS).
        plx : float
            The parallax of the science object (TEL.TARG.PARALLAX) in arcsec.
        ft_pol : string
            The polarization of the fringe tracker (INS.FT.POL), 'IN' or 'OUT'.
        ao_type : string
            The adaptive optics type (COU.AO.TYPE), 'NGS_VIS', 'NGS_IR', 'LGS_VIS', 'LGS_IR'.
        ngs_source : string
            The guide star source (COU.NGS.SOURCE), 'SCIENCE', or 'SETUPFILE'.
        ngs_alpha : string
            The R.A. of the guide star (COU.NGS.ALPHA), in HH:MM:SS.SSS,
            only if ngs_source is 'SETUPFILE'.
        ngs_delta : string
            The Decl. of the guide star (COU.NGS.DELTA), in DD:MM:SS.SSS,
            only if ngs_source is 'SETUPFILE'.
        ngs_plx : float
            The parallax of the guide star (COU.NGS.PARALLAX), in arcsec,
            only if ngs_source is 'SETUPFILE'.
        ngs_pma : float
            The proper motion of R.A. of the guide star (COU.NGS.PMA), in arcsec/yr,
            only if ngs_source is 'SETUPFILE'.
        ngs_pmd : float
            The proper motion of Decl. of the guide star (COU.NGS.PMD), in arcsec/yr,
            only if ngs_source is 'SETUPFILE'.
        ngs_epoch : float
            The epoch of the guide star coordinate (COU.NGS.EPOCH), in year, better only use 2000.
        ngs_mag : float
            The magnitude of the guide star in the optical or IR depending on
            the ao_type.
        lgs_source : string
            The LGS source (COU.LGS.SOURCE), 'SCIENCE', 'NGS', 'FTS', or 'SETUPFILE'
        lgs_alpha : string 
            The R.A. of the LGS (COU.LGS.ALPHA), in HH:MM:SS.SSS,
            only if lgs_source is 'SETUPFILE'.
        lgs_delta : string
            The Decl. of the LGS (COU.LGS.DELTA), in DD:MM:SS.SSS,
            only if lgs_source is 'SETUPFILE'. 
        lgs_plx : float
            The parallax of the LGS (COU.LGS.PARALLAX), in arcsec,
            only if lgs_source is 'SETUPFILE'.
        lgs_pma : float
            The proper motion of R.A. of the LGS (COU.LGS.PMA), in arcsec/yr,
            only if lgs_source is 'SETUPFILE'.
        lgs_pmd : float
            The proper motion of Decl. of the LGS (COU.LGS.PMD), in arcsec/yr,
            only if lgs_source is 'SETUPFILE'.
        lgs_epoch : float
            The epoch of the LGS coordinate (COU.LGS.EPOCH), in year, better only use 2000.
            only if lgs_source is 'SETUPFILE'.
        baseline : list
            The baseline type, 'small', 'large', or 'UTs'.
        vltitype : list
            The VLTI type, 'snapshot', 'imaging', 'time-series', or 'astrometry'.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
        
        obid = ob['obId']
        try:
            acqTpl, acqTplVersion = api.createTemplate(obid, 'GRAVITY_single_offaxis_acq')
        except p2api.P2Error as e:
            raise p2api.P2Error(e)
        
        sc_name = sanitize_name(sc_name)
        ngs_name = sanitize_name(ngs_name)  
        lgs_name = sanitize_name(lgs_name)

        pdict = {
            'SEQ.FT.MODE': ft_mode,
            'SEQ.MET.MODE': met_mode,
            'TEL.TARG.NAME': sc_name,
            'TEL.TARG.MAG.K': sc_kmag,
            'TEL.TARG.MAG.H': sc_hmag,
            'TEL.TARG.DIAMETER': sc_d,
            'TEL.TARG.VIS': sc_vis,
            'TEL.TARG.PARALLAX': plx,
            'INS.SPEC.RES': spec_res,
            'INS.FT.POL': ft_pol,
            'COU.AO.TYPE': ao_mode,
            'COU.NGS.SOURCE': ngs_source,
            'COU.NGS.NAME': ngs_name,
            'COU.NGS.ALPHA': ngs_alpha,
            'COU.NGS.DELTA': ngs_delta,
            'COU.NGS.PARALLAX': ngs_plx,
            'COU.NGS.PMA': ngs_pma,
            'COU.NGS.PMD': ngs_pmd,
            'COU.NGS.EPOCH': ngs_epoch,
            'COU.NGS.MAG': ngs_mag,
            'COU.LGS.SOURCE': lgs_source,
            'COU.LGS.NAME': lgs_name,
            'COU.LGS.ALPHA': lgs_alpha,
            'COU.LGS.DELTA': lgs_delta,
            'COU.LGS.PARALLAX': lgs_plx,
            'COU.LGS.PMA': lgs_pma,
            'COU.LGS.PMD': lgs_pmd,
            'COU.LGS.EPOCH': lgs_epoch,
            'ISS.BASELINE': baseline,
            'ISS.VLTITYPE': vltitype
        }
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])


    def add_GRAVITY_single_obs_calibrator(self, name, folder_name=None, dit=0.3, ndit_obj=32, 
                                          ndit_sky=32, hwpoff=[0], obsseq='O S', sky_x=2000, 
                                          sky_y=2000):
        '''
        Add acquisition template: GRAVITY_single_acq

        Parameters
        ----------
        name : string
            The OB name.
        folder_name : string
            The folder name.
        dit : float
            The DIT.
        ndit_obj : int
            The NDIT for the object.
        ndit_sky : int
            The NDIT for the sky.
        hwpoff : list
            The HWP offset, in degree.
        obsseq : string
            The observation sequence, e.g. 'O O S'.
        sky_x : float
            The x offset of the sky from the object, in mas.
        sky_y : float
            The y offset of the sky from the object, in mas.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
        
        obid = ob['obId']
        try:
            acqTpl, acqTplVersion = api.createTemplate(obid, 'GRAVITY_single_obs_calibrator')
        except p2api.P2Error as e:
            raise p2api.P2Error(e)
        
        pars = acqTpl['parameters']
        pList = [dit, ndit_obj, ndit_sky, hwpoff, obsseq, sky_x, sky_y]
        pdict = {}
        for loop, pv in enumerate(pList):
            pdict[pars[loop]['name']] = pv
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        
        
    def add_GRAVITY_single_obs_exp(self, name, folder_name=None, dit=0.3, ndit_obj=32, 
                                   ndit_sky=32, hwpoff=[0], obsseq='O S', sky_x=2000, 
                                   sky_y=2000):
        '''
        Add acquisition template: GRAVITY_single_acq

        Parameters
        ----------
        name : string
            The OB name.
        folder_name : string
            The folder name.
        dit : float
            The DIT.
        ndit_obj : int
            The NDIT for the object.
        ndit_sky : int
            The NDIT for the sky.
        hwpoff : list
            The HWP offset, in degree.
        obsseq : string
            The observation sequence, e.g. 'O O S'.
        sky_x : float
            The x offset of the sky from the object, in mas.
        sky_y : float
            The y offset of the sky from the object, in mas.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
        
        obid = ob['obId']
        try:
            acqTpl, acqTplVersion = api.createTemplate(obid, 'GRAVITY_single_obs_exp')
        except p2api.P2Error as e:
            raise p2api.P2Error(e)
        
        pars = acqTpl['parameters']
        
        pdict = {
            'DET2.DIT': dit,
            'DET2.NDIT.OBJECT': ndit_obj,
            'DET2.NDIT.SKY': ndit_sky,
            'SEQ.SKY.X': sky_x,
            'SEQ.SKY.Y': sky_y,
            'SEQ.HWPOFF': hwpoff,
            'SEQ.OBSSEQ': obsseq
        }
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        
        
    def create_folder(self, name, container_id=None, overwrite=False):
        '''
        Create folder.
        
        Parameters
        ----------
        name : string
            Folder name.
        runContainerId (optional) : string
            The run container ID.  If not provided, create the folder directly under the run.
        overwrite : bool
            Overwrite the existing OB if True.
        '''
        api = self.api
        fdict = self.folderDict
        
        if container_id is None:
            container_id = self._rootContainterId
        
        if name in fdict:
            if overwrite:
                folder, folderVersion = fdict[name]
                self.delete_folder(folder['containerId'], force=True)
            else:
                raise Exception('The folder ({}) exists!'.format(name))
                
        fdict[name] = api.createFolder(container_id, name)
        return fdict[name]
    
        
    def create_OB(self, name, folder_name=None, overwrite=False):
        '''
        Create an OB.
        
        Parameters
        ----------
        name : string
            OB name.
        folder_name : string 
            Folder name.
        overwrite : bool
            Overwrite the existing OB if True.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        containerId = folder['containerId']
        
        odict = folder.get('OBs', None)
        if odict is None:
            folder['OBs'] = {}
            odict = folder['OBs']
            
        if name in odict:
            if overwrite:
                ob, obVersion = odict[name]
                api.deleteOB(ob['obId'], obVersion)
            else:
                raise Exception('The OB ({}) exists!'.format(name))
        
        odict[name] = api.createOB(containerId, name)
        return odict[name]
    
    
    def create_rootFolder(self, name, overwrite=False):
        '''
        Create a root folder and work on it.

        Parameters
        ----------
        name : string
            Folder name.
        overwrite : bool
            Overwrite the existing OB if True.
        '''
        api = self.api
        containerId = self._runContainerId
        
        items = api.getItems(containerId)[0]
        for it in items:
            if (it['itemType'] == 'Folder') & (it['name'] == name):
                if overwrite:
                    self.delete_folder(it['containerId'], force=True)
                else:
                    raise Exception('The folder ({}) already exists!'.format(name))
        
        fd, fdV = api.createFolder(containerId, name)
        self.rootDict['Root'] = [fd, fdV]
        self.set_rootContainterId(fd['containerId'])
        return fd, fdV
    
    
    def delete_folder(self, folder_id, force=False):
        '''
        Delete the folder.
        
        Parameters 
        ----------
        folder_id : int 
            Folder ID.
        force : bool
            Empty the folder and delete it if True.
        '''
        api = self.api
        
        if force:
            # Delete all of the content first
            items = api.getItems(folder_id)[0]
            for it in items:
                if it['itemType'] == 'OB':
                    ob, obV = api.getOB(it['obId'])
                    api.deleteOB(ob['obId'], obV)
                elif it['itemType'] == 'Folder':
                    fd, fdV = api.getContainer(it['containerId'])
                    self.delete_folder(fd['containerId'], force=force)
                else:
                    raise KeyError('Cannot recognize this type ({})!'.format(it['itemType']))
            fd, fdV = api.getContainer(folder_id)
            api.deleteContainer(folder_id, fdV)
        else:
            fd, fdV = api.getContainer(folder_id)
            api.deleteContainer(folder_id, fdV)

    
    def get_folder(self, folder_name):
        '''
        Get the folder information.
        
        Parameters
        ----------
        folder_name : string 
            Folder name.
        '''
        if folder_name is None:
            folder_name = 'Root'
            folder, folderVersion = self.rootDict[folder_name]  # Root folder 
            #folderVersion = None
        else:
            assert folder_name in self.folderDict, 'Cannot find the foler ({})!'.format(folder_name)
            folder, folderVersion = self.folderDict[folder_name]
            
        return folder, folderVersion
    
        
    def save_OB(self, name, folder_name=None):
        '''
        Save an OB.
        
        Parameters
        ----------
        name : string
            OB name.
        folder_name : string 
            Folder name.
        '''
        api = self.api
        folder, folderVersion = self.get_folder(folder_name)
        odict = folder.get('OBs', None)
        if odict is None:
            raise Exception('The folder ({}) does not contain any OB!'.format(folder_name))
        
        assert name in odict, 'Cannot find the OB in {}!'.format(folder_name)
        ob, obVersion = odict[name]
            
        api.saveOB(ob, obVersion)
        odict[name] = api.getOB(ob['obId'])
        return odict[name]
    
    
    def set_rootFolder(self, name):
        '''
        Set a root folder and work on it.

        Parameters
        ----------
        name : string
            Folder name.
        '''
        api = self.api
        containerId = self._runContainerId
        
        items = api.getItems(containerId)[0]
        fd = None
        for it in items:
            if (it['itemType'] == 'Folder') & (it['name'] == name):
                fd, fdV = api.getContainer(it['containerId'])
                self.rootDict['Root'] = [fd, fdV]
        
        if fd is None:
            raise Exception('Cannot find a folder named {}!'.format(name))
        else:
            self.set_rootContainterId(fd['containerId'])
        
    
    def set_rootContainterId(self, root_container_id):
        '''
        Set the rootContainterId.
        
        Parameters
        ----------
        root_container_id : int 
            ID of the root container.
        '''
        self._rootContainterId = root_container_id
        self.update_content()
    
    
    def update_content(self, no_warning=False):
        '''
        Update the content of this run.

        Parameters
        ----------
        no_warning : bool
            Do not raise warning if True.
        '''
        api = self.api
        
        # Put in the existing folders and OBs. I do not care the duplication at this point.
        self.folderDict = {}
        items = api.getItems(self._rootContainterId)[0]
        for it in items:
            name = it['name']
            if it['itemType'] == 'OB':
                if name in self.rootDict:
                    if not no_warning:
                        raise Warning('OB "{}" already exists in Root!'.format(name))
                self.rootDict[name] = api.getOB(it['obId'])
            elif it['itemType'] == 'Folder':
                if name in self.folderDict:
                    if not no_warning:
                        raise Warning('Folder "{}" already exists in Root!'.format(name))
                
                self.folderDict[name] = api.getContainer(it['containerId'])
                folder, folderVersion = self.folderDict[it['name']]
                folder['OBs'] = {}
                
                items_in_folder = api.getItems(folder['containerId'])[0]
                for it_in_folder in items_in_folder:
                    if it_in_folder['itemType'] == 'OB':
                        folder['OBs'][it_in_folder['name']] = api.getOB(it_in_folder['obId'])
            else:
                raise KeyError('Cannot recognize this type ({})!'.format(it['itemType']))


