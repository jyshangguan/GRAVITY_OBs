import numpy as np
from datetime import datetime
import p2api

now = datetime.now()
dt_string = now.strftime("%Y-%m-%dT%H:%M:%S")

all = ['check_item_exist', 'create_OB', 'ob_add_target', 'ob_add_description', 
       'p2api_GRAVITY']


def get_dit_med(ksc, tel='UT', verbose=True):
    '''
    Find the proper dit and ndit for MEDIUM.
    '''
    if tel == 'AT':
        ksc = ksc + 3

    if ksc < 6:
        dit, ndit = 0.3, 320
        
        if verbose:
            print(f'The source may be too bright ({ksc}) for MEDIUM!')

    elif (ksc >= 6) & (ksc < 7):
        dit, ndit = 1, 120
    elif (ksc >= 7) & (ksc < 8.5):
        dit, ndit = 3, 40
    elif (ksc >= 8.5) & (ksc < 9.5):
        dit, ndit = 10, 12
    else:
        dit, ndit = 30, 4
    
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
    ob['target']['name'] = name
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
        
        
    def add_GRAVITY_dual_offaxis_acq(self, name, folder_name=None, ft_mode='AUTO', met_mode='ON', picksc='A', 
                                     ft_name='Name', ft_kmag=0, ft_hmag=0, ft_d=0, ft_vis=1, 
                                     sc_name='Name', sc_kmag=0, sc_d=0, sc_vis=1, sobj_x=0, sobj_y=0,
                                     ft_plx=0, spec_res='MED', ft_pol='OUT', sc_pol='OUT', ag_type='ADAPT_OPT', 
                                     gs_source='FT', gs_alpha='00:00:00.00', gs_delta='00:00:00.00', gs_plx=0, 
                                     gs_pma=0, gs_pmd=0, gs_epoch=2000, gs_mag=0, baseline=['astrometric'], 
                                     vltitype=['snapshot']):
        '''
        Add acquisition template: GRAVITY_dual_acq
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
        
        pdict = {
            'SEQ.FT.MODE': ft_mode,
            'SEQ.MET.MODE': met_mode,
            'SEQ.PICKSC': picksc,
            'SEQ.FT.ROBJ.NAME': ft_name,
            'SEQ.FT.ROBJ.MAG': ft_kmag,
            'SEQ.FT.ROBJ.HMAG': ft_hmag,
            'SEQ.FT.ROBJ.DIAMETER': ft_d,
            'SEQ.FT.ROBJ.VIS': ft_vis,
            'SEQ.INS.SOBJ.NAME': sc_name,
            'SEQ.INS.SOBJ.MAG': sc_kmag,
            'SEQ.INS.SOBJ.DIAMETER': sc_d,
            'SEQ.INS.SOBJ.VIS': sc_vis,
            'SEQ.INS.SOBJ.X': sobj_x,
            'SEQ.INS.SOBJ.Y': sobj_y,
            'TEL.TARG.PARALLAX': ft_plx,
            'INS.SPEC.RES': spec_res,
            'INS.FT.POL': ft_pol,
            'INS.SPEC.POL': sc_pol,
            'COU.AG.TYPE': ag_type,
            'COU.AG.GSSOURCE': gs_source,
            'COU.AG.ALPHA': gs_alpha,
            'COU.AG.DELTA': gs_delta,
            'COU.AG.PARALLAX': gs_plx,
            'COU.AG.PMA': gs_pma,
            'COU.AG.PMD': gs_pmd,
            'COU.AG.EPOCH': gs_epoch,
            'COU.GS.MAG': gs_mag,
            'ISS.BASELINE': baseline,
            'ISS.VLTITYPE': vltitype
        }
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        

    def add_GRAVITY_dual_onaxis_acq(self, name, folder_name=None, ft_mode='AUTO', met_mode='ON', 
                                    ft_name='Name', ft_kmag=0, ft_hmag=0, ft_d=0, ft_vis=1, 
                                    sc_name='Name', sc_kmag=0, sc_d=0, sc_vis=1, sobj_x=0, sobj_y=0,
                                    ft_plx=0, spec_res='MED', ft_pol='OUT', sc_pol='OUT', ag_type='ADAPT_OPT', 
                                    gs_source='FT', gs_alpha='00:00:00.00', gs_delta='00:00:00.00', gs_plx=0, 
                                    gs_pma=0, gs_pmd=0, gs_epoch=2000, gs_mag=0, baseline=['astrometric'], 
                                    vltitype=['snapshot']):
        '''
        Add acquisition template: GRAVITY_dual_acq
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
        
        pdict = {
            'SEQ.FT.MODE': ft_mode,
            'SEQ.MET.MODE': met_mode,
            'SEQ.FT.ROBJ.NAME': ft_name,
            'SEQ.FT.ROBJ.MAG': ft_kmag,
            'SEQ.FT.ROBJ.HMAG': ft_hmag,
            'SEQ.FT.ROBJ.DIAMETER': ft_d,
            'SEQ.FT.ROBJ.VIS': ft_vis,
            'SEQ.INS.SOBJ.NAME': sc_name,
            'SEQ.INS.SOBJ.MAG': sc_kmag,
            'SEQ.INS.SOBJ.DIAMETER': sc_d,
            'SEQ.INS.SOBJ.VIS': sc_vis,
            'SEQ.INS.SOBJ.X': sobj_x,
            'SEQ.INS.SOBJ.Y': sobj_y,
            'TEL.TARG.PARALLAX': ft_plx,
            'INS.SPEC.RES': spec_res,
            'INS.FT.POL': ft_pol,
            'INS.SPEC.POL': sc_pol,
            'COU.AG.TYPE': ag_type,
            'COU.AG.GSSOURCE': gs_source,
            'COU.AG.ALPHA': gs_alpha,
            'COU.AG.DELTA': gs_delta,
            'COU.AG.PARALLAX': gs_plx,
            'COU.AG.PMA': gs_pma,
            'COU.AG.PMD': gs_pmd,
            'COU.AG.EPOCH': gs_epoch,
            'COU.GS.MAG': gs_mag,
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
        
        
    def add_GRAVITY_dual_wide_acq(self, name, folder_name=None, ft_mode='AUTO', met_mode='ON',
                                  ft_name='Name', ft_mag=0, ft_d=0, ft_vis=1, ft_epoch=2000,
                                  sc_name='Name', sc_mag=0, 
                                  sc_d=0, sc_vis=1, sc_hmag=0, ft_hmag=0, ft_alpha=None, 
                                  ft_delta=None, ft_plx=0, ft_pma=0, ft_pmd=0, 
                                  sc_plx=0, spec_res='MED', ft_pol='IN', sc_pol='IN', 
                                  gssource='SCIENCE', ag_alpha='00:00:00.000', ag_delta='00:00:00.000',
                                  ag_plx=0, gs_mag=0, ag_pma=0, ag_pmd=0, ag_epoch=2000, ag_type='ADAPT_OPT', 
                                  baseline=['astrometric'], vltitype=['snapshot']):
        '''
        Add acquisition template: GRAVITY_dual_wide_acq
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
        
        pdict = {
            'SEQ.FT.MODE': ft_mode,
            'SEQ.MET.MODE': met_mode,
            'SEQ.FT.ROBJ.NAME': ft_name,
            'SEQ.FT.ROBJ.MAG': ft_mag,
            'SEQ.FT.ROBJ.HMAG': ft_hmag,
            'SEQ.FT.ROBJ.DIAMETER': ft_d,
            'SEQ.FT.ROBJ.VIS': ft_vis,
            'SEQ.FT.ROBJ.ALPHA': ft_alpha,
            'SEQ.FT.ROBJ.DELTA': ft_delta,
            'SEQ.FT.ROBJ.PARALLAX': ft_plx,
            'SEQ.FT.ROBJ.PMA': ft_pma,
            'SEQ.FT.ROBJ.PMD': ft_pmd,
            'SEQ.FT.ROBJ.EPOCH': ft_epoch,
            'SEQ.INS.SOBJ.NAME': sc_name,
            'SEQ.INS.SOBJ.MAG': sc_mag,
            'SEQ.INS.SOBJ.HMAG': sc_hmag,
            'SEQ.INS.SOBJ.DIAMETER': sc_d,
            'SEQ.INS.SOBJ.VIS': sc_vis,
            'TEL.TARG.PARALLAX': sc_plx,
            'INS.SPEC.RES': spec_res,
            'INS.FT.POL': ft_pol,
            'INS.SPEC.POL': sc_pol,
            'COU.AG.TYPE': ag_type,
            'COU.AG.GSSOURCE': gssource,
            'COU.AG.ALPHA': ag_alpha,
            'COU.AG.DELTA': ag_delta,
            'COU.AG.PARALLAX': ag_plx,
            'COU.AG.PMA': ag_pma,
            'COU.AG.PMD': ag_pmd,
            'COU.AG.EPOCH': ag_epoch,
            'COU.GS.MAG': gs_mag,
            'ISS.BASELINE': baseline,
            'ISS.VLTITYPE': vltitype
        }
        
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        
        
    def add_GRAVITY_single_onaxis_acq(self, name, folder_name=None, ft_mode='AUTO', met_mode='ON', 
                                      sc_name='Name', sc_kmag=0, sc_hmag=0, sc_d=0, sc_vis=1, plx=0, 
                                      spec_res='MED', ft_pol='OUT', sc_pol='OUT', ag_type='ADAPT_OPT', 
                                      gs_source='SCIENCE', gs_alpha='00:00:00.000', gs_delta='00:00:00.000', 
                                      gs_plx=0, gs_pma=0, gs_pmd=0, gs_epoch=2000, gs_mag=0, 
                                      baseline=['astrometric'], vltitype=['snapshot']):
        '''
        Add acquisition template: GRAVITY_single_acq
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
        
        pars = acqTpl['parameters']
        
        pdict = {
            'SEQ.FT.MODE': ft_mode,
            'SEQ.MET.MODE': met_mode,
            'SEQ.INS.SOBJ.NAME': sc_name,
            'SEQ.INS.SOBJ.MAG': sc_kmag,
            'SEQ.INS.SOBJ.HMAG': sc_hmag,
            'SEQ.INS.SOBJ.DIAMETER': sc_d,
            'SEQ.INS.SOBJ.VIS': sc_vis,
            'TEL.TARG.PARALLAX': plx,
            'INS.SPEC.RES': spec_res,
            'INS.FT.POL': ft_pol,
            'INS.SPEC.POL': sc_pol,
            'COU.AG.TYPE': ag_type,
            'COU.AG.GSSOURCE': gs_source,
            'COU.AG.ALPHA': gs_alpha,
            'COU.AG.DELTA': gs_delta,
            'COU.AG.PARALLAX': gs_plx,
            'COU.AG.PMA': gs_pma,
            'COU.AG.PMD': gs_pmd,
            'COU.AG.EPOCH': gs_epoch,
            'COU.GS.MAG': gs_mag,
            'ISS.BASELINE': baseline,
            'ISS.VLTITYPE': vltitype
        }
        acqTpl, acqTplVersion  = api.setTemplateParams(obid, acqTpl, pdict, acqTplVersion)
        
        # update the OB version
        odict[name] = api.getOB(ob['obId'])
        
    
    def add_GRAVITY_single_offaxis_acq(self, name, folder_name=None, ft_mode='AUTO', met_mode='OFF', 
                                       sc_name='Name', sc_kmag=0, sc_hmag=0, sc_d=0, sc_vis=1, plx=0, 
                                       ft_pol='OUT', ag_type='ADAPT_OPT', gs_source='SCIENCE', 
                                       gs_alpha='00:00:00.000', gs_delta='00:00:00.000', 
                                       gs_plx=0, gs_pma=0, gs_pmd=0, gs_epoch=2000, gs_mag=0, 
                                       baseline=['astrometric'], vltitype=['snapshot']):
        '''
        Add acquisition template: GRAVITY_single_acq
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
        
        pars = acqTpl['parameters']
        
        pdict = {
            'SEQ.FT.MODE': ft_mode,
            'SEQ.MET.MODE': met_mode,
            'SEQ.INS.SOBJ.NAME': sc_name,
            'SEQ.INS.SOBJ.MAG': sc_kmag,
            'SEQ.INS.SOBJ.HMAG': sc_hmag,
            'SEQ.INS.SOBJ.DIAMETER': sc_d,
            'SEQ.INS.SOBJ.VIS': sc_vis,
            'TEL.TARG.PARALLAX': plx,
            'INS.FT.POL': ft_pol,
            'COU.AG.TYPE': ag_type,
            'COU.AG.GSSOURCE': gs_source,
            'COU.AG.ALPHA': gs_alpha,
            'COU.AG.DELTA': gs_delta,
            'COU.AG.PARALLAX': gs_plx,
            'COU.AG.PMA': gs_pma,
            'COU.AG.PMD': gs_pmd,
            'COU.AG.EPOCH': gs_epoch,
            'COU.GS.MAG': gs_mag,
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
    
        
