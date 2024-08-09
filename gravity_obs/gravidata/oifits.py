import logging
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.utils.decorators import lazyproperty
from .gravi_utils import N_BASELINE, N_TRIANGLE, N_TELESCOPE
from .gravi_utils import baseline_names, telescope_names, triangle_names
from .gravi_utils import t2b_matrix, lambda_met
from .astro_utils import phase_model, matrix_opd
from .astro_utils import grid_search, gdelay_astrometry, compute_gdelay


class GraviFits(object):
    '''
    The general class to read and plot GRAVITY OIFITS data.
    '''
    def __init__(self, filename, ignore_flag=False):
        '''
        Parameters
        ----------
        filename : str
            The name of the OIFITS file.
        ignore_flag : bool, optional
            If True, the FLAG and REJECTION_FLAG will be ignored. Default is False.
        '''
        self._filename = filename
        header = fits.getheader(filename, 0)
        self._header = header
        self._ignore_flag = ignore_flag

        self._arcfile = header.get('ARCFILE', None)
        if self._arcfile is not None:
            self._arctime = self._arcfile.split('.')[1]
        else:
            self._arctime = None

        self._object = header.get('OBJECT', None)

        # Instrument mode
        self._res = header.get('ESO INS SPEC RES', None)
        self._pol = header.get('ESO INS POLA MODE', None)
        if self._pol == 'SPLIT':
            self._pol_list = [1, 2]
        elif self._pol == 'COMBINED':
            self._pol_list = [0]
        else:
            self._pol_list = None
        self._npol = len(self._pol_list)

        # Science target
        self._sobj_x = header.get('ESO INS SOBJ X', None)
        self._sobj_y = header.get('ESO INS SOBJ Y', None)
        self._sobj_offx = header.get('ESO INS SOBJ OFFX', None)
        self._sobj_offy = header.get('ESO INS SOBJ OFFY', None)
        self._swap = header.get('ESO INS SOBJ SWAP', None) == 'YES'
        self._dit = header.get('ESO DET2 SEQ1 DIT', None)
        self._ndit = header.get('ESO DET2 NDIT', None)

        telescop = header.get('TELESCOP', None)
        if 'U1234' in telescop:
            self.set_telescope('UT')
        elif 'A1234' in telescop:
            self.set_telescope('AT')
        else:
            self.set_telescope('GV')

    
    def copy(self):
        '''
        Copy the current AstroFits object.
        '''
        return deepcopy(self)
    

    def get_extver(self, fiber='SC', polarization=None):
        '''
        Get the extver of the OIFITS HDU. The first digit is the fiber type 
        (1 or 2), and the second digit is the polarization (0, 1, or 2).

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        extver : int
            The extver of the OIFITS HDU.
        '''
        assert fiber in ['SC', 'FT'], 'fiber must be SC or FT.'

        if polarization is None:
            if self._pol == 'SPLIT':
                polarization = 1
            else:
                polarization = 0
        assert polarization in [0, 1, 2], 'polarization must be 0, 1, or 2.'

        fiber_code = {'SC': 1, 'FT': 2}
        extver = int(f'{fiber_code[fiber]}{polarization}')
        return extver


    def get_vis(self, field, fiber='SC', polarization=None):
        '''
        Get data from the OI_VIS extension of the OIFITS HDU.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        data : masked array
            The data of the OI_VIS HDU. The shape is (NDIT, N_BASELINE, N_CHANNEL).
        '''
        extver = self.get_extver(fiber, polarization)
        data = fits.getdata(self._filename, 'OI_VIS', extver=extver)

        # Get the data
        field_data = data[field]
        if len(field_data.shape) == 2:
            nchn = field_data.shape[1]

            # Get the flag
            if self._ignore_flag:
                flag = None
            else:
                try:
                    flag = data['FLAG']
                    try:
                        rejflag  = data['REJECTION_FLAG'][:, np.newaxis]
                    except KeyError:
                        rejflag = np.zeros_like(flag)
                    flag = flag | ((rejflag & 19) > 0)
                except KeyError:
                    flag = None

        elif len(field_data.shape) < 3:
            # Supporting data, so flag is not needed
            flag = None  #np.zeros_like(field_data, dtype=bool)
            nchn = 1

        else:
            raise ValueError(f'The data has an incorrect shape (data[field].shape)!')

        field_data = np.ma.array(field_data, mask=flag).reshape(-1, N_BASELINE, nchn)
        return field_data
    

    def get_wavelength(self, fiber='SC', units='micron'):
        '''
        Get the wavelength of the OIFITS HDU.
        '''
        assert units in ['micron', 'm'], 'units must be micron or m.'

        extver = self.get_extver(fiber)
        wave = fits.getdata(self._filename, 'OI_WAVELENGTH', extver=extver)['EFF_WAVE']

        if units == 'micron':
            wave = wave * 1e6

        return wave
    

    def set_telescope(self, telescope):
        '''
        Set the telescope input of GRAVITY. The telescope, baseline, and 
        triangle names are adjusted accordingly.

        Parameters
        ----------
        telescope : str
            The telescope. Either UT, AT, or GV.
        '''
        assert telescope in ['UT', 'AT', 'GV'], 'telescope must be UT, AT, or GV.'
        self._telescope = telescope_names[telescope]
        self._baseline = baseline_names[telescope]
        self._triangle = triangle_names[telescope]


class AstroFits(GraviFits):
    '''
    A class to read and plot GRAVITY ASTROREDUCED data.
    '''
    def __init__(self, filename, ignore_flag=False, per_exp=False, normalize=False):
        '''
        Parameters
        ----------
        filename : str
            The path to the OIFITS file.
        ignore_flag : bool, optional [False]
            Use the FLAG to mask the data if False..
        per_exp : bool, optional [False]
            Use visibility data per exposure time if True.
        normalize : bool, optional [False]
            Normalize the visibility data with F1F2, if True.
        '''
        super().__init__(filename, ignore_flag)

        self._wave = self.get_wavelength(units='micron')

        # Set the uv coordinates
        self._uvcoord_m = self.get_uvcoord_vis(units='m')
        self._uvcoord_permas = self.get_uvcoord_vis(units='permas')
        self._uvcoord_Mlambda = self.get_uvcoord_vis(units='Mlambda')

        # Visibility data
        for p in self._pol_list:
            visdata = self.get_visdata(polarization=p, per_exp=per_exp, normalize=normalize)
            extver = self.get_extver(fiber='SC', polarization=p)
            setattr(self, f'_visdata_{extver}', visdata)


    def chi2_phase(self, ra, dec, polarization=None):
        '''
        Calculate the chi2 to search for source offset with the phase method.
        '''
        gamma = []
        gooddata = []

        if polarization is None:
            pols = self._pol_list
        else:
            pols = [polarization]

        for p in pols:
            u, v = self._uvcoord_Mlambda
            phase = phase_model(ra, dec, u, v)
            model = np.exp(1j * phase)
            visdata = getattr(self, f'_visdata_1{p}')
            gamma.append(np.conj(model) * visdata)
            gooddata.append(visdata.mask == False)

        gamma = np.ma.sum(np.concatenate(gamma), axis=0) / np.sum(np.concatenate(gooddata), axis=0)
        chi2 = np.ma.sum(gamma.imag**2)
        chi2_baseline = np.ma.sum(gamma.imag**2, axis=1)
        return chi2, chi2_baseline


    def correct_visdata(self, 
                        polarization : int = None,
                        met_jump : list = None, 
                        opdzp : list = None):
        '''
        Correct the metrology phase jump.

        Parameters
        ----------
        met_jump : list
            Number of fringe jumps to be added, telescope quantity, [UT4, UT3, UT2, UT1].
        opdzp : list
            OPD zeropoint, baseline quantity, [UT43, UT42, UT41, UT32, UT31, UT21].
        '''
        assert ~((met_jump is None) & (opdzp is None)), 'met_jump and opdzp cannot be None at the same time!'
        extver = self.get_extver(fiber='SC', polarization=polarization)
        visdata = getattr(self, f'_visdata_{extver}')

        if met_jump is not None:
            corr_tel = np.array(met_jump)[:, np.newaxis] * 2*np.pi * (1 - lambda_met / self._wave)[np.newaxis, :]
            opdDisp_corr = np.dot(t2b_matrix, corr_tel)
            visdata *= np.exp(1j * opdDisp_corr)

        if opdzp is not None:
            v_opd = np.exp(2j * np.pi * np.array(opdzp)[:, np.newaxis] / self._wave[np.newaxis, :])
            visdata *= np.conj(v_opd)[np.newaxis, :, :]


    def diff_visphi(self, 
                    oi : 'AstroFits', 
                    polarization : int = None,
                    average : bool = False):
        '''
        Calculate the difference of the VISPHI between the VISDATA of 
        the current and input AstroFits file.
        '''
        extver = self.get_extver(fiber="SC", polarization=polarization)
        vd1 = getattr(self, f'_visdata_{extver}')
        vd2 = getattr(oi, f'_visdata_{extver}')

        if average:
            vd1 = np.mean(vd1, axis=0, keepdims=True)
            vd2 = np.mean(vd2, axis=0, keepdims=True)

        visphi = np.angle(vd1 * np.conj(vd2))
        return visphi


    def get_opdDisp(self, polarization=None):
        '''
        Get the OPD_DISP from the OIFITS file.
        '''
        return self.get_vis('OPD_DISP', fiber='SC', polarization=polarization)
    

    def get_opdMetCorr(self, polarization=None):
        '''
        Get the OPD_MET_CORR by combining OPD_MET_FC_CORR and OPD_MET_TELFC_MCORR from the OIFITS file.
        '''
        opdMetFcCorr = self.get_vis('OPD_MET_FC_CORR', fiber='SC', polarization=polarization)
        opdMetTelFcMCorr = self.get_vis('OPD_MET_TELFC_MCORR', fiber='SC', polarization=polarization)
        opdMetCorr = (opdMetFcCorr + opdMetTelFcMCorr)

        return opdMetCorr


    def get_phaseref(self, polarization=None):
        '''
        Get the PHASE_REF from the OIFITS file.
        '''
        return self.get_vis('PHASE_REF', fiber='SC', polarization=polarization)


    def get_opdSep(self):
        '''
        Get the OPD due to the separation vector from FT to SC.
        '''
        opdSep = np.pi / 180 / 3600 / 1e3 * (self._uvcoord_m[0] * self._sobj_x + self._uvcoord_m[1] * self._sobj_y)
        return opdSep


    def get_visdata(self, polarization=None, per_exp=False, normalize=False):
        '''
        Get the phase referenced VISDATA of the OIFITS HDU.

        Parameters
        ----------
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        visdata : masked arrays
            The VISDATA of the OIFITS HDU. The shape is (NDIT, NBASELINE, NCHANNEL).
        '''
        assert ~(per_exp & normalize), 'per_exp and normalize cannot be True at the same time.'

        visdata = self.get_vis('VISDATA', fiber='SC', polarization=polarization)
        phaseref= self.get_phaseref(polarization=polarization)
        opdDisp = self.get_opdDisp(polarization=polarization)
        opdMetCorr = self.get_opdMetCorr(polarization=polarization)
        opdSep = self.get_opdSep()

        visdata = visdata * np.exp(1j * (phaseref + 2*np.pi / self._wave * (opdSep - opdDisp - opdMetCorr) * 1e6))

        if per_exp:
            visdata /= self._dit

        if normalize:
            f1f2 = self.get_vis('F1F2', fiber='SC', polarization=polarization)
            visdata /= np.sqrt(f1f2)

        return visdata


    def get_uvcoord_vis(self, polarization=None, units='m'):
        '''
        Get the u and v coordinates of the baselines.

        Parameters
        ----------
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv coordinates. Either Mlambda, permas, or m.
            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.
        
        Returns
        -------
        ucoord, vcoord : arrays
            The uv coordinate of the baselines, (NDIT, NBASELINE, NCHANNEL).
        '''
        assert units in ['Mlambda', 'permas', 'm'], 'units must be Mlambda, per mass, or m.'

        ucoord = self.get_vis('UCOORD', fiber='SC', polarization=polarization)
        vcoord = self.get_vis('VCOORD', fiber='SC', polarization=polarization)
        
        if units != 'm':
            wave = self.get_wavelength(units='micron')
            ucoord = ucoord / wave[np.newaxis, np.newaxis, :]
            vcoord = vcoord / wave[np.newaxis, np.newaxis, :]
            
            if units == 'permas':
                ucoord = ucoord * np.pi / 180. / 3600 * 1e3
                vcoord = vcoord * np.pi / 180. / 3600 * 1e3

        return ucoord, vcoord


    def get_uvdist(self, units='Mlambda'):
        '''
        Get the uv distance of the baselines.

        Parameters
        ----------
        units : str, optional
            The units of the uv distance. Either Mlambda, permas, or m.
            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.
        
        Returns
        -------
        uvdist : array
            The uv distance of the baselines, (NDIT, NBASELINE, NCHANNEL).
        '''
        assert units in ['Mlambda', 'permas', 'm'], 'units must be Mlambda, per mass, or m.'

        ucoord, vcoord = getattr(self, f'_uvcoord_{units}')
        uvdist = np.sqrt(ucoord**2 + vcoord**2)

        return uvdist


    def grid_search_phase(self, polarization=None, plot=False, **kwargs):
        '''
        Perform a grid search to find the best RA and Dec offsets.

        WARNING
        -------
        This method seems to be less robust than the cal_offset_opd() method for 
        individual files.
        '''
        res = grid_search(self.chi2_phase, 
                          chi2_func_args=dict(polarization=polarization), 
                          plot=plot, **kwargs)
        
        if plot:
            if self._swap:
                ra_total = self._sobj_x - res['ra_best']
                dec_total = self._sobj_y - res['dec_best']
            else:
                ra_total = self._sobj_x + res['ra_best']
                dec_total = self._sobj_y + res['dec_best']

            ax1, ax2 = res['axs']
            ax1.text(0.05, 0.95, f'Swap: {self._swap}', fontsize=12, 
                     transform=ax1.transAxes, va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            text = '\n'.join([f'SOBJ_XY: ({self._sobj_x:.2f}, {self._sobj_y:.2f})', 
                    f'Total: ({ra_total:.2f}, {dec_total:.2f})'])
            ax2.text(0.05, 0.95, text, fontsize=12, transform=ax2.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        return res


    def plot_visphi(self, polarization=None, average=False, ax=None, plain=False):
        '''
        Plot the angle of the VISDATA.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        extver = self.get_extver(fiber='SC', polarization=polarization)
        visdata = getattr(self, f'_visdata_{extver}')
        ruv = self.get_uvdist(units='Mlambda')

        if average:
            visdata = np.mean(visdata, axis=0, keepdims=True)
            ruv = np.mean(ruv, axis=0, keepdims=True)
        
        for dit in range(visdata.shape[0]):
            for bsl in range(visdata.shape[1]):
                ax.plot(ruv[dit, bsl, :], np.angle(visdata[dit, bsl, :], deg=True), color=f'C{bsl}')
        
        if not plain:
            ax.set_ylim([-180, 180])
            ax.set_xlabel('uv distance (Mlambda)', fontsize=18)
            ax.set_ylabel(r'VISPHI ($^\circ$)', fontsize=18)
            ax.minorticks_on()
        return ax
    

    def plot_visamp(self, polarization=None, average=False, ax=None, plain=False):
        '''
        Plot the amplitude of the VISDATA
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        extver = self.get_extver(fiber='SC', polarization=polarization)
        visdata = getattr(self, f'_visdata_{extver}')
        ruv = self.get_uvdist(units='Mlambda')

        if average:
            visdata = np.mean(visdata, axis=0, keepdims=True)
            ruv = np.mean(ruv, axis=0, keepdims=True)
        
        for dit in range(visdata.shape[0]):
            for bsl in range(visdata.shape[1]):
                ax.plot(ruv[dit, bsl, :], np.absolute(visdata[dit, bsl, :]), color=f'C{bsl}')
        
        if not plain:
            ax.set_ylim([0, None])
            ax.set_xlabel('uv distance (Mlambda)', fontsize=18)
            ax.set_ylabel(r'VISAMP', fontsize=18)
            ax.minorticks_on()
        return ax
        

class SciVisFits(GraviFits):
    '''
    '''
    # Revised
    def __init__(self, filename, ignore_flag=False, normalize=True):
        super().__init__(filename, ignore_flag)

        self._wave = self.get_wavelength(units='micron')

        # Set the uv coordinates
        self._uvcoord_m = self.get_uvcoord_vis(units='m')
        self._uvcoord_permas = self.get_uvcoord_vis(units='permas')
        self._uvcoord_Mlambda = self.get_uvcoord_vis(units='Mlambda')

        # Visibility data
        for p in self._pol_list:
            extver = self.get_extver(fiber='SC', polarization=p)

            visdata = self.get_visdata(fiber='SC', polarization=p, 
                                       normalize=normalize)
            setattr(self, f'_visdata_{extver}', visdata)

            visamp = self.get_vis('VISAMP', fiber='SC', polarization=p)
            setattr(self, f'_visamp_{extver}', visamp)

            visphi = self.get_vis('VISPHI', fiber='SC', polarization=p)
            setattr(self, f'_visphi_{extver}', visphi)
            
            extver = self.get_extver(fiber='FT', polarization=p)

            visdata = self.get_visdata(fiber='FT', polarization=p, 
                                       normalize=normalize)
            setattr(self, f'_visdata_{extver}', visdata)

            visamp = self.get_vis('VISAMP', fiber='FT', polarization=p)
            setattr(self, f'_visamp_{extver}', visdata)

            visphi = self.get_vis('VISPHI', fiber='FT', polarization=p)
            setattr(self, f'_visphi_{extver}', visphi)


    def chi2_phase(
            self, 
            ra : float, 
            dec : float, 
            polarization=None):
        '''
        Calculate the chi2 to search for source offset with the phase method.
        '''
        gamma = []
        gooddata = []

        if polarization is None:
            pols = self._pol_list
        else:
            pols = [polarization]

        for p in pols:
            u, v = self._uvcoord_Mlambda
            phase = phase_model(ra, dec, u, v)
            model = np.exp(1j * phase)
            visdata = getattr(self, f'_visdata_1{p}')
            gamma.append(np.conj(model) * visdata)
            gooddata.append(visdata.mask == False)

        gamma = np.ma.sum(np.concatenate(gamma), axis=0) / np.sum(np.concatenate(gooddata), axis=0)
        chi2 = np.ma.sum(gamma.imag**2)
        chi2_baseline = np.ma.sum(gamma.imag**2, axis=1)
        return chi2, chi2_baseline


    def correct_visdata(
            self, 
            polarization : int = None,
            met_jump : list = None, 
            opdzp : list = None):
        '''
        Correct the metrology phase jump.

        Parameters
        ----------
        met_jump : list
            Number of fringe jumps to be added, telescope quantity, [UT4, UT3, UT2, UT1].
        opdzp : list
            OPD zeropoint, baseline quantity, [UT43, UT42, UT41, UT32, UT31, UT21].
        '''
        assert ~((met_jump is None) & (opdzp is None)), 'met_jump and opdzp cannot be None at the same time!'
        extver = self.get_extver(fiber='SC', polarization=polarization)
        visdata = getattr(self, f'_visdata_{extver}')

        if met_jump is not None:
            corr_tel = np.array(met_jump)[:, np.newaxis] * 2*np.pi * (1 - lambda_met / self._wave)[np.newaxis, :]
            opdDisp_corr = np.dot(t2b_matrix, corr_tel)
            visdata *= np.exp(1j * opdDisp_corr)
            setattr(self, f'_visphi_{extver}', np.angle(visdata, deg=True))

        if opdzp is not None:
            v_opd = np.exp(2j * np.pi * np.array(opdzp)[:, np.newaxis] / self._wave[np.newaxis, :])
            visdata *= np.conj(v_opd)[np.newaxis, :, :]
            setattr(self, f'_visphi_{extver}', np.angle(visdata, deg=True))


    def diff_visphi(self, 
                    oi : 'AstroFits', 
                    polarization : int = None,
                    average : bool = False):
        '''
        Calculate the difference of the VISPHI between the VISDATA of 
        the current and input AstroFits file.
        '''
        extver = self.get_extver(fiber="SC", polarization=polarization)
        vd1 = getattr(self, f'_visdata_{extver}')
        vd2 = getattr(oi, f'_visdata_{extver}')

        if average:
            vd1 = np.mean(vd1, axis=0, keepdims=True)
            vd2 = np.mean(vd2, axis=0, keepdims=True)

        visphi = np.angle(vd1 * np.conj(vd2))
        return visphi


    # Revised
    def get_visdata(
            self, 
            fiber : str = 'SC', 
            polarization : int = None, 
            per_exp : bool = False, 
            normalize : bool = True):
        '''
        Get the VISDATA of the SCIVIS data.

        Parameters
        ----------
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        visdata : masked arrays
            The VISDATA of the OIFITS HDU. The shape is (NDIT, NBASELINE, NCHANNEL).
        '''
        assert ~(per_exp & normalize), 'per_exp and normalize cannot be True at the same time.'

        if normalize:
            visamp = self.get_vis('VISAMP', fiber=fiber, polarization=polarization)
            visphi = self.get_vis('VISPHI', fiber=fiber, polarization=polarization)
            visdata = visamp * np.exp(1j * np.deg2rad(visphi))
            return visdata

        visdata = self.get_vis('VISDATA', fiber=fiber, polarization=polarization)

        if per_exp:
            if visdata.shape[0] == 1:
                visdata /= self._dit * self._ndit
            elif visdata.shape[0] == self._ndit:
                visdata /= self._dit
            else:
                raise ValueError('Unclear how to calculate the exposure time!')

        return visdata


    def get_uvcoord_vis(
            self, 
            fiber : str = 'SC', 
            polarization : int = None, 
            units : str = 'm'):
        '''
        Get the u and v coordinates of the baselines.

        Parameters
        ----------
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv coordinates. Either Mlambda, permas, or m.
            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.
        
        Returns
        -------
        ucoord, vcoord : arrays
            The uv coordinate of the baselines, (NDIT, NBASELINE, NCHANNEL).
        '''
        assert units in ['Mlambda', 'permas', 'm'], 'units must be Mlambda, per mass, or m.'

        ucoord = self.get_vis('UCOORD', fiber=fiber, polarization=polarization)
        vcoord = self.get_vis('VCOORD', fiber=fiber, polarization=polarization)
        
        if units != 'm':
            wave = self.get_wavelength(units='micron')
            ucoord = ucoord / wave[np.newaxis, np.newaxis, :]
            vcoord = vcoord / wave[np.newaxis, np.newaxis, :]
            
            if units == 'permas':
                ucoord = ucoord * np.pi / 180. / 3600 * 1e3
                vcoord = vcoord * np.pi / 180. / 3600 * 1e3

        return ucoord, vcoord


    def get_uvdist(
            self, 
            units : str = 'Mlambda'):
        '''
        Get the uv distance of the baselines.

        Parameters
        ----------
        units : str, optional
            The units of the uv distance. Either Mlambda, permas, or m.
            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.
        
        Returns
        -------
        uvdist : array
            The uv distance of the baselines, (NDIT, NBASELINE, NCHANNEL).
        '''
        assert units in ['Mlambda', 'permas', 'm'], 'units must be Mlambda, per mass, or m.'

        ucoord, vcoord = getattr(self, f'_uvcoord_{units}')
        uvdist = np.sqrt(ucoord**2 + vcoord**2)

        return uvdist


    def grid_search_phase(
            self, 
            polarization=None, 
            plot=False, 
            **kwargs):
        '''
        Perform a grid search to find the best RA and Dec offsets.

        WARNING
        -------
        This method seems to be less robust than the cal_offset_opd() method for 
        individual files.
        '''
        res = grid_search(self.chi2_phase, 
                          chi2_func_args=dict(polarization=polarization), 
                          plot=plot, **kwargs)
        
        if plot:
            if self._swap:
                ra_total = self._sobj_x - res['ra_best']
                dec_total = self._sobj_y - res['dec_best']
            else:
                ra_total = self._sobj_x + res['ra_best']
                dec_total = self._sobj_y + res['dec_best']

            ax1, ax2 = res['axs']
            ax1.text(0.05, 0.95, f'Swap: {self._swap}', fontsize=12, 
                     transform=ax1.transAxes, va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            text = '\n'.join([f'SOBJ_XY: ({self._sobj_x:.2f}, {self._sobj_y:.2f})', 
                    f'Measured: ({ra_total:.2f}, {dec_total:.2f})'])
            ax2.text(0.05, 0.95, text, fontsize=12, transform=ax2.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        return res


    # Revised
    def plot_visphi(self, fiber='SC', polarization=None, use_visdata=False, ax=None, plain=False):
        '''
        Plot the angle of the VISDATA.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        extver = self.get_extver(fiber=fiber, polarization=polarization)

        if use_visdata:
            visdata = getattr(self, f'_visdata_{extver}')
            visphi = np.angle(visdata[dit, bsl, :], deg=True)
        else:
            visphi = getattr(self, f'_visphi_{extver}')

        ruv = self.get_uvdist(units='Mlambda')

        for dit in range(visphi.shape[0]):
            for bsl in range(visphi.shape[1]):
                ax.plot(ruv[dit, bsl, :], visphi[dit, bsl, :], color=f'C{bsl}')
        
        if not plain:
            ax.set_ylim([-180, 180])
            ax.set_xlabel('uv distance (Mlambda)', fontsize=18)
            ax.set_ylabel(r'VISPHI ($^\circ$)', fontsize=18)
            ax.minorticks_on()
        return ax
    

    # Revised
    def plot_visamp(self, fiber='SC', polarization=None, use_visdata=False, ax=None, plain=False):
        '''
        Plot the amplitude of the VISDATA
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        extver = self.get_extver(fiber=fiber, polarization=polarization)

        if use_visdata:
            visdata = getattr(self, f'_visdata_{extver}')
            visamp = np.absolute(visdata[dit, bsl, :])
        else:
            visamp = getattr(self, f'_visamp_{extver}')

        ruv = self.get_uvdist(units='Mlambda')
        
        for dit in range(visamp.shape[0]):
            for bsl in range(visamp.shape[1]):
                ax.plot(ruv[dit, bsl, :], visamp[dit, bsl, :], color=f'C{bsl}')
        
        if not plain:
            ax.set_ylim([0, None])
            ax.set_xlabel('uv distance (Mlambda)', fontsize=18)
            ax.set_ylabel(r'VISAMP', fontsize=18)
            ax.minorticks_on()
        return ax



class GraviList(object):
    '''
    A list of GRAVITY OIFITS object.
    '''
    def __init__(self, name='GraviList', log_name=None, verbose=True):

        self._name = name

        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(logging.INFO)  # Set the logging level
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Avoid adding handlers multiple times
        if not self._logger.hasHandlers():
            # Create a file handler
            file_handler = logging.FileHandler(f'{log_name}.log', mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

            # Create a console handler
            if verbose:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self._logger.addHandler(console_handler)


    def __add__(self, other : 'GraviList'):
        '''
        Add two AstroList objects.
        '''
        self._datalist = self._datalist + other._datalist
    

    def __getitem__(self, index):
        '''
        Parameters
        ----------
        index : int or list
        '''
        if isinstance(index, list):
            return [self._datalist[i] for i in index]
        elif isinstance(index, int):
            return self._datalist[index]


    def __setitem__(self, index, value):
        self._datalist[index] = value


    def __len__(self):
        return len(self._datalist)


    def __iter__(self):
        return iter(self._datalist)
    

    def __repr__(self):
        return f'GraviList with {len(self._datalist)} files'
    

    def __str__(self):
        return f'GraviList with {len(self._datalist)} files'


    def append(self, oi):
        '''
        Append an AstroFits object to the list.
        '''
        self._datalist.append(oi)

    
    def copy(self):
        '''
        Copy the current AstroList object.
        '''
        return deepcopy(self)


class AstroList(GraviList):
    '''
    A list of AstroFits objects.
    '''
    def __init__(
            self, 
            files : list, 
            ignore_flag : bool = False, 
            per_exp : bool = False, 
            normalize : bool = False,
            verbose=True,
            log_name=None) -> None:
        '''
        Parameters
        ----------

        '''
        super().__init__(name='AstroList', log_name=log_name, verbose=verbose)
        
        self._datalist = []
        self._index_unswap = []
        self._index_swap = []
        for i, f in enumerate(files):
            self._logger.info(f'Processing {f}')

            oi = AstroFits(f, ignore_flag, per_exp, normalize)
            self._datalist.append(oi)

            if oi._swap:
                self._index_swap.append(i)
            else:
                self._index_unswap.append(i)

        self._pol_list = self._datalist[0]._pol_list
        self._sobj_x = self._datalist[0]._sobj_x
        self._sobj_y = self._datalist[0]._sobj_y


    def chi2_phase(self, ra, dec):
        '''
        Parameters
        ----------
        ra : float
            Right ascension offset in milliarcsec.
        dec : float
            Declination offset in milliarcsec.
        '''
        gamma1 = []
        gooddata1 = []
        for oi in self[self._index_unswap]:
            for p in self._pol_list:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(ra, dec, u, v)
                model = np.exp(1j * phase)
                visdata = getattr(oi, f'_visdata_1{p}')
                gamma1.append(np.conj(model) * visdata)
                gooddata1.append(visdata.mask == False)
        
        gamma2 = []
        gooddata2 = []
        for oi in self[self._index_swap]:
            for p in self._pol_list:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(ra, dec, u, v)
                model = np.exp(1j * phase)
                visdata = getattr(oi, f'_visdata_1{p}')
                gamma2.append(model * visdata)
                gooddata2.append(visdata.mask == False)

        gamma1 = np.ma.sum(np.concatenate(gamma1), axis=0) / np.sum(np.concatenate(gooddata1), axis=0)
        gamma2 = np.ma.sum(np.concatenate(gamma2), axis=0) / np.sum(np.concatenate(gooddata2), axis=0)
        gamma_swap = (np.conj(gamma1) * gamma2)**0.5  # Important not to use np.sqrt() here!
        chi2 = np.ma.sum(gamma_swap.imag**2)
        chi2_baseline = np.ma.sum(gamma_swap.imag**2, axis=1)
    
        return chi2, chi2_baseline


    def compute_metzp(
            self, 
            ra : float,
            dec : float,
            closure=True,
            closure_lim : float = 1.2,
            max_width : float = 2000,
            plot=False, 
            axs=None, 
            verbose=True,
            pdf=None):
        '''
        Calculate the metrology zero point
        '''
        visdata1 = []
        visdata2 = []
        for p in self._pol_list:
            # Unswapped data
            vd = []
            for oi in self[self._index_unswap]:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(-ra, -dec, u, v)
                vd.append(getattr(oi, f'_visdata_1{p}') * np.exp(1j * phase))
            visdata1.append(vd)
    
            vd = []
            for oi in self[self._index_swap]:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(ra, dec, u, v)
                vd.append(getattr(oi, f'_visdata_1{p}') * np.exp(1j * phase))
            visdata2.append(vd)

        visdata = 0.5 * (np.mean(visdata1, axis=(1, 2)) + np.mean(visdata2, axis=(1, 2)))
        phi0 = np.angle(visdata)
    
        npol = len(self._pol_list)
        wave = oi._wave
        baseline_name = oi._baseline

        # Prepare deriving the metrology zeropoint; [NPOL, NBASELINE]
        self._opdzp = compute_gdelay(visdata, wave, max_width=max_width, 
                                     closure=closure, closure_lim=closure_lim, 
                                     verbose=verbose, logger=self._logger)
        self._fczp = np.array([self._opdzp[:, 2], 
                               self._opdzp[:, 4], 
                               self._opdzp[:, 5], 
                               np.zeros(npol)])

        if plot:
            if axs is None:
                fig, axs = plt.subplots(6, npol, figsize=(8*npol, 12), sharex=True)
                fig.suptitle(f'Metrology zero point in phase', fontsize=14)
                fig.subplots_adjust(hspace=0.02)
                axo = fig.add_subplot(111, frameon=False) # The out axis
                axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18, labelpad=20)
                axo.set_ylabel(r'$\phi_0$ (rad)', fontsize=18, labelpad=50)

            else:
                assert len(axs) == 6, 'The number of axes must be 6!'

            for i, p in enumerate(self._pol_list):
                for bsl, ax in enumerate(axs[:, i]):
                    ax.plot(wave, phi0[i, bsl, :], color=f'C{bsl}')
                    ax.plot(wave, np.angle(matrix_opd(self._opdzp[i, bsl], wave)), color='gray', alpha=0.5, 
                            label='OPD model')
                    ax.text(0.95, 0.9, f'{self._opdzp[i, bsl]:.2f} $\mu$m', fontsize=14, color='k', 
                            transform=ax.transAxes, va='top', ha='right', 
                            bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))

                    ax.minorticks_on()
                    ax.text(0.02, 0.9, baseline_name[bsl], transform=ax.transAxes, fontsize=14, 
                            va='top', ha='left', color=f'C{bsl}', fontweight='bold',
                            bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))
            ax.legend(loc='lower left', fontsize=14)

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)


    def correct_metzp(self):
        '''
        Correct the metrology zero points.
        '''
        for oi in self:
            for i, p in enumerate(self._pol_list):
                oi.correct_visdata(polarization=p, opdzp=self._opdzp[i, :])


    def correct_met_jump(
            self, 
            index : int, 
            met_jump : list):
        '''
        Correct the metrology phase jump.
        '''
        for p in self._pol_list:
            self._datalist[index].correct_visdata(polarization=p, met_jump=met_jump)


    def grid_search_phase(self, plot=True, **kwargs):
        '''
        Perform a grid search to find the best RA and Dec offsets.
        '''
        res= grid_search(self.chi2_phase, plot=plot, **kwargs)

        if plot:
            ra_total = self._sobj_x + res['ra_best']
            dec_total = self._sobj_y + res['dec_best']
            ax = res['axs'][1]
            text = '\n'.join([f'SOBJ_XY: ({self._sobj_x:.2f}, {self._sobj_y:.2f})', 
                    f'Total: ({ra_total:.2f}, {dec_total:.2f})'])
            ax.text(0.05, 0.95, text, fontsize=12, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        return res


    def gdelay_astrometry(self, max_width=2000, plot=False, axs=None, pdf=None):
        '''
        Perform the group delay astrometry.
        '''
        pair_index = np.array(np.meshgrid(self._index_unswap, self._index_swap)).T.reshape(-1, 2)

        offsetList = []
        for loop, (i, j) in enumerate(pair_index):
            offset = []

            if plot:
                npol = len(self._pol_list)
                if axs is None:
                    fig, axs_use = plt.subplots(1, npol, figsize=(6*npol, 6), sharex=True, sharey=True)
                    fig.suptitle(f'GD astrometry: {self[int(i)]._arcfile} and {self[int(j)]._arcfile}', fontsize=14)
                    fig.subplots_adjust(wspace=0.02)
                    axo = fig.add_subplot(111, frameon=False)
                    axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                    axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                    axo.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24, labelpad=20)
                    axo.set_ylabel(r'VISPHI ($^\circ$)', fontsize=24, labelpad=32)
                else:
                    axs_use = axs[loop, :]
            else:
                axs_use = None

            for loop, p in enumerate(self._pol_list):
                if axs_use is None:
                    ax = None
                else:
                    ax = np.atleast_1d(axs_use)[loop]
                    ax.axhline(y=0, ls='--', color='k')

                offset.append(gdelay_astrometry(self[int(i)], self[int(j)], polarization=p, 
                                                average=True, max_width=max_width, plot=plot, 
                                                ax=ax))

            if ax is not None:
                ax.minorticks_on()
                ax.set_ylim([-np.pi, np.pi])
                ax.legend(loc='best', ncols=3, handlelength=1, columnspacing=1, fontsize=14)
            
            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)

            offsetList.append(offset)

        offsetList = np.array(offsetList).swapaxes(1, 2)  #[NDIT, XY, POLARIZATION]

        return offsetList


    def run_swap_astrometry(self, met_jump_dict=None, plot=True, report_name=None, verbose=True):
        '''
        The main function to measure astrometry and metrology zero point from 
        the swap data.
        '''
        assert len(self._index_unswap) > 0, 'There is no unswap data in the list!'
        assert len(self._index_swap) > 0, 'There is no swap data in the list!'

        if report_name is not None:
            from matplotlib.backends.backend_pdf import PdfPages

            pdf = PdfPages(report_name)
        else:
            pdf = None


        # Correct the metrology phase jump
        if met_jump_dict is not None:
            self._logger.info('Correct the metrology phase jump')
            for i, met_jump in met_jump_dict.items():
                self.correct_met_jump(i, met_jump)


        # Plot data
        if plot:
            self._logger.info('Plotting the data')
            for oi in self:
                fig, axs = plt.subplots(oi._npol, 2, figsize=(14, 7*oi._npol))
                axs = np.array(axs).reshape(-1, 2)
                fig.suptitle(f'Original data: {oi._arcfile}', fontsize=18)

                for i, p in enumerate(self._pol_list):
                    oi.plot_visamp(polarization=p, average=True, ax=axs[i, 0])
                    oi.plot_visphi(polarization=p, average=True, ax=axs[i, 1])

                if pdf is not None:
                    pdf.savefig(fig)
                    plt.close(fig)
        

        # Search for astrometry using the phase method
        self._logger.info('Grid search for astrometry solution')
        res = self.grid_search_phase(plot=False)


        # Measure metrology zero point
        self._logger.info('Measure metrology zero point')
        self.compute_metzp(res['ra_best'], res['dec_best'], plot=plot, pdf=pdf)


        # Correct the metrology zero point
        self._logger.info('Correct the metrology zeropoint')
        self.correct_metzp()


        # Plot the phase to evaluate the data correction
        self._logger.info('Plotting the metrology zeropoint corrected data')
        for oi in self:
            fig, axs = plt.subplots(1, oi._npol, figsize=(7*oi._npol, 7))
            axs = np.atleast_1d(axs)
            fig.suptitle(f'METZP corrected: {oi._arcfile}', fontsize=18)

            for i, p in enumerate(self._pol_list):
                oi.plot_visphi(polarization=p, average=True, ax=axs[i])

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)


        # Plot the chi2 map, grid search solution, and group delay astrometry solutions
        self._logger.info('Final grid search for astrometry solution')

        if plot:
            fig, axs = plt.subplots(1, 3, figsize=(21, 7))
            fig.suptitle('Astrometry results', fontsize=18)

            axs_grid = axs[:2]
        else:
            axs_grid = None

        res = self.grid_search_phase(plot=plot, axs=axs_grid)
        self._ra_best = res['ra_best']
        self._dec_best = res['dec_best']
        self._sobj_x_fit = self._sobj_x + res['ra_best']
        self._sobj_y_fit = self._sobj_y + res['dec_best']

        
        # Calculate the group delay astrometry
        self._logger.info('Calculate the group delay astrometry')
        offsets_gd = self.gdelay_astrometry(plot=plot, pdf=pdf)

        if plot:
            ax = axs[2]
            colors = ['C0', 'C2']
            markers = ['+', 'x']
            for p in range(offsets_gd.shape[-1]):
                for i in range(offsets_gd.shape[0]):
                    if i == 0:
                        label = f'GD P{self._pol_list[p]}'
                    else:
                        label = None

                    ax.plot(offsets_gd[i, 0, p], offsets_gd[i, 1, p], color=colors[p], ls='none', 
                            marker=markers[p], ms=6, lw=2, label=label)

            ax.plot(res['ra_best_zoom'], res['dec_best_zoom'], marker='x', ms=15, color='C3')
            ax.legend(fontsize=16, loc='upper right', handlelength=1)
        
            # Expand the panel by 2 times for the current axis and make the aspect ratio equal
            ax = axs[2]
            xlim =  ax.get_xlim()
            ylim =  ax.get_ylim()
        
            alim = np.max([np.abs(np.diff(xlim)), np.abs(np.diff(ylim))]) * 0.8
            xcent = np.mean(xlim)
            ycent = np.mean(ylim)
            ax.set_xlim([xcent - alim, xcent + alim])
            ax.set_ylim([ycent - alim, ycent + alim])
            ax.set_aspect('equal')
            ax.set_title('Zoom in astrometry', fontsize=16)
            ax.minorticks_on()
            ax.set_xlabel(r'$\Delta$RA (mas)', fontsize=18)

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)

        if pdf is not None:
            pdf.close()
        
        self._logger.info('OPD_MET_ZERO_FC: ')
        for i, p in enumerate(self._pol_list):
            self._logger.info(', '.join([f'{v:.2f}' for v in self._fczp[:, i]]))

        self._logger.info("Pipeline completed!")


class SciVisList(GraviList):
    '''
    A list of SciVisFits objects.
    '''
    def __init__(
            self, 
            files : list, 
            ignore_flag : bool = False, 
            normalize : bool = False,
            verbose=True,
            log_name=None) -> None:
        '''
        Parameters
        ----------

        '''
        super().__init__(name='SciVisList', log_name=log_name, verbose=verbose)
        
        self._datalist = []
        self._index_unswap = []
        self._index_swap = []
        for i, f in enumerate(files):
            self._logger.info(f'Processing {f}')

            oi = SciVisFits(f, ignore_flag=ignore_flag, normalize=normalize)
            self._datalist.append(oi)

            if oi._swap:
                self._index_swap.append(i)
            else:
                self._index_unswap.append(i)

        self._pol_list = self._datalist[0]._pol_list
        self._sobj_x = self._datalist[0]._sobj_x
        self._sobj_y = self._datalist[0]._sobj_y


    def chi2_phase(self, ra, dec):
        '''
        Parameters
        ----------
        ra : float
            Right ascension offset in milliarcsec.
        dec : float
            Declination offset in milliarcsec.
        '''
        gamma1 = []
        gooddata1 = []
        for oi in self[self._index_unswap]:
            for p in self._pol_list:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(ra, dec, u, v)
                model = np.exp(1j * phase)
                visdata = getattr(oi, f'_visdata_1{p}')
                gamma1.append(np.conj(model) * visdata)
                gooddata1.append(visdata.mask == False)
        
        gamma2 = []
        gooddata2 = []
        for oi in self[self._index_swap]:
            for p in self._pol_list:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(ra, dec, u, v)
                model = np.exp(1j * phase)
                visdata = getattr(oi, f'_visdata_1{p}')
                gamma2.append(model * visdata)
                gooddata2.append(visdata.mask == False)

        gamma1 = np.ma.sum(np.concatenate(gamma1), axis=0) / np.sum(np.concatenate(gooddata1), axis=0)
        gamma2 = np.ma.sum(np.concatenate(gamma2), axis=0) / np.sum(np.concatenate(gooddata2), axis=0)
        gamma_swap = (np.conj(gamma1) * gamma2)**0.5  # Important not to use np.sqrt() here!
        chi2 = np.ma.sum(gamma_swap.imag**2)
        chi2_baseline = np.ma.sum(gamma_swap.imag**2, axis=1)
    
        return chi2, chi2_baseline


    def compute_metzp(
            self, 
            ra : float,
            dec : float,
            closure=True,
            closure_lim : float = 1.2,
            max_width : float = 2000,
            plot=False, 
            axs=None, 
            verbose=True,
            pdf=None):
        '''
        Calculate the metrology zero point
        '''
        visdata1 = []
        visdata2 = []
        for p in self._pol_list:
            # Unswapped data
            vd = []
            for oi in self[self._index_unswap]:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(-ra, -dec, u, v)
                vd.append(getattr(oi, f'_visdata_1{p}') * np.exp(1j * phase))
            visdata1.append(vd)
    
            vd = []
            for oi in self[self._index_swap]:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(ra, dec, u, v)
                vd.append(getattr(oi, f'_visdata_1{p}') * np.exp(1j * phase))
            visdata2.append(vd)

        visdata = 0.5 * (np.mean(visdata1, axis=(1, 2)) + np.mean(visdata2, axis=(1, 2)))
        phi0 = np.angle(visdata)
    
        npol = len(self._pol_list)
        wave = oi._wave
        baseline_name = oi._baseline

        # Prepare deriving the metrology zeropoint; [NPOL, NBASELINE]
        self._opdzp = compute_gdelay(visdata, wave, max_width=max_width, 
                                     closure=closure, closure_lim=closure_lim, 
                                     verbose=verbose, logger=self._logger)
        self._fczp = np.array([self._opdzp[:, 2], 
                               self._opdzp[:, 4], 
                               self._opdzp[:, 5], 
                               np.zeros(npol)])

        if plot:
            if axs is None:
                fig, axs = plt.subplots(6, npol, figsize=(8*npol, 12), sharex=True)
                fig.suptitle(f'Metrology zero point in phase', fontsize=14)
                fig.subplots_adjust(hspace=0.02)
                axo = fig.add_subplot(111, frameon=False) # The out axis
                axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18, labelpad=20)
                axo.set_ylabel(r'$\phi_0$ (rad)', fontsize=18, labelpad=50)

            else:
                assert len(axs) == 6, 'The number of axes must be 6!'

            for i, p in enumerate(self._pol_list):
                for bsl, ax in enumerate(axs[:, i]):
                    ax.plot(wave, phi0[i, bsl, :], color=f'C{bsl}')
                    ax.plot(wave, np.angle(matrix_opd(self._opdzp[i, bsl], wave)), color='gray', alpha=0.5, 
                            label='OPD model')
                    ax.text(0.95, 0.9, f'{self._opdzp[i, bsl]:.2f} $\mu$m', fontsize=14, color='k', 
                            transform=ax.transAxes, va='top', ha='right', 
                            bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))

                    ax.minorticks_on()
                    ax.text(0.02, 0.9, baseline_name[bsl], transform=ax.transAxes, fontsize=14, 
                            va='top', ha='left', color=f'C{bsl}', fontweight='bold',
                            bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))
            ax.legend(loc='lower left', fontsize=14)

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)


    def correct_metzp(self):
        '''
        Correct the metrology zero points.
        '''
        for oi in self:
            for i, p in enumerate(self._pol_list):
                oi.correct_visdata(polarization=p, opdzp=self._opdzp[i, :])


    def correct_met_jump(
            self, 
            index : int, 
            met_jump : list):
        '''
        Correct the metrology phase jump.
        '''
        for p in self._pol_list:
            self._datalist[index].correct_visdata(polarization=p, met_jump=met_jump)


    def grid_search_phase(self, plot=True, **kwargs):
        '''
        Perform a grid search to find the best RA and Dec offsets.
        '''
        res= grid_search(self.chi2_phase, plot=plot, **kwargs)

        if plot:
            ra_total = self._sobj_x + res['ra_best']
            dec_total = self._sobj_y + res['dec_best']
            ax = res['axs'][1]
            text = '\n'.join([f'SOBJ_XY: ({self._sobj_x:.2f}, {self._sobj_y:.2f})', 
                    f'Total: ({ra_total:.2f}, {dec_total:.2f})'])
            ax.text(0.05, 0.95, text, fontsize=12, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        return res


    def gdelay_astrometry(self, max_width=2000, plot=False, axs=None, pdf=None):
        '''
        Perform the group delay astrometry.
        '''
        pair_index = np.array(np.meshgrid(self._index_unswap, self._index_swap)).T.reshape(-1, 2)

        offsetList = []
        for loop, (i, j) in enumerate(pair_index):
            offset = []

            if plot:
                npol = len(self._pol_list)
                if axs is None:
                    fig, axs_use = plt.subplots(1, npol, figsize=(6*npol, 6), sharex=True, sharey=True)
                    fig.suptitle(f'GD astrometry: {self[int(i)]._arcfile} and {self[int(j)]._arcfile}', fontsize=14)
                    fig.subplots_adjust(wspace=0.02)
                    axo = fig.add_subplot(111, frameon=False)
                    axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                    axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                    axo.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24, labelpad=20)
                    axo.set_ylabel(r'VISPHI ($^\circ$)', fontsize=24, labelpad=32)
                else:
                    axs_use = axs[loop, :]
            else:
                axs_use = None

            for loop, p in enumerate(self._pol_list):
                if axs_use is None:
                    ax = None
                else:
                    ax = np.atleast_1d(axs_use)[loop]
                    ax.axhline(y=0, ls='--', color='k')

                offset.append(gdelay_astrometry(self[int(i)], self[int(j)], polarization=p, 
                                                average=True, max_width=max_width, plot=plot, 
                                                ax=ax))

            if ax is not None:
                ax.minorticks_on()
                ax.set_ylim([-np.pi, np.pi])
                ax.legend(loc='best', ncols=3, handlelength=1, columnspacing=1, fontsize=14)
            
            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)

            offsetList.append(offset)

        offsetList = np.array(offsetList).swapaxes(1, 2)  #[NDIT, XY, POLARIZATION]

        return offsetList


    def run_swap_astrometry(self, met_jump_dict=None, plot=True, report_name=None, verbose=True):
        '''
        The main function to measure astrometry and metrology zero point from 
        the swap data.
        '''
        assert len(self._index_unswap) > 0, 'There is no unswap data in the list!'
        assert len(self._index_swap) > 0, 'There is no swap data in the list!'

        if report_name is not None:
            from matplotlib.backends.backend_pdf import PdfPages

            pdf = PdfPages(report_name)
        else:
            pdf = None


        # Correct the metrology phase jump
        if met_jump_dict is not None:
            self._logger.info('Correct the metrology phase jump')
            for i, met_jump in met_jump_dict.items():
                self.correct_met_jump(i, met_jump)


        # Plot data
        if plot:
            self._logger.info('Plotting the data')
            for oi in self:
                fig, axs = plt.subplots(oi._npol, 2, figsize=(14, 7*oi._npol))
                axs = np.array(axs).reshape(-1, 2)
                fig.suptitle(f'Original data: {oi._arcfile}', fontsize=18)

                for i, p in enumerate(self._pol_list):
                    oi.plot_visamp(polarization=p, ax=axs[i, 0])
                    oi.plot_visphi(polarization=p, ax=axs[i, 1])

                if pdf is not None:
                    pdf.savefig(fig)
                    plt.close(fig)
        

        # Search for astrometry using the phase method
        self._logger.info('Grid search for astrometry solution')
        res = self.grid_search_phase(plot=False)


        # Measure metrology zero point
        self._logger.info('Measure metrology zero point')
        self.compute_metzp(res['ra_best'], res['dec_best'], plot=plot, pdf=pdf)


        # Correct the metrology zero point
        self._logger.info('Correct the metrology zeropoint')
        self.correct_metzp()


        # Plot the phase to evaluate the data correction
        self._logger.info('Plotting the metrology zeropoint corrected data')
        for oi in self:
            fig, axs = plt.subplots(1, oi._npol, figsize=(7*oi._npol, 7))
            axs = np.atleast_1d(axs)
            fig.suptitle(f'METZP corrected: {oi._arcfile}', fontsize=18)

            for i, p in enumerate(self._pol_list):
                oi.plot_visphi(polarization=p, ax=axs[i])
                axs[i].text(0.05, 0.95, f'P{p}', fontsize=16, 
                            transform=axs[i].transAxes, va='top', ha='left',
                            bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)


        # Plot the chi2 map, grid search solution, and group delay astrometry solutions
        self._logger.info('Final grid search for astrometry solution')

        if plot:
            fig, axs = plt.subplots(1, 3, figsize=(21, 7))
            fig.suptitle('Astrometry results', fontsize=18)

            axs_grid = axs[:2]
        else:
            axs_grid = None

        res = self.grid_search_phase(plot=plot, axs=axs_grid)
        self._ra_best = res['ra_best']
        self._dec_best = res['dec_best']
        self._sobj_x_fit = self._sobj_x + res['ra_best']
        self._sobj_y_fit = self._sobj_y + res['dec_best']

        
        # Calculate the group delay astrometry
        self._logger.info('Calculate the group delay astrometry')
        offsets_gd = self.gdelay_astrometry(plot=plot, pdf=pdf)

        if plot:
            ax = axs[2]
            colors = ['C0', 'C2']
            markers = ['+', 'x']
            for p in range(offsets_gd.shape[-1]):
                for i in range(offsets_gd.shape[0]):
                    if i == 0:
                        label = f'GD P{self._pol_list[p]}'
                    else:
                        label = None

                    ax.plot(offsets_gd[i, 0, p], offsets_gd[i, 1, p], color=colors[p], ls='none', 
                            marker=markers[p], ms=6, lw=2, label=label)

            ax.plot(res['ra_best_zoom'], res['dec_best_zoom'], marker='x', ms=15, color='C3')
            ax.legend(fontsize=16, loc='upper right', handlelength=1)
        
            # Expand the panel by 2 times for the current axis and make the aspect ratio equal
            ax = axs[2]
            xlim =  ax.get_xlim()
            ylim =  ax.get_ylim()
        
            alim = np.max([np.abs(np.diff(xlim)), np.abs(np.diff(ylim))]) * 0.8
            xcent = np.mean(xlim)
            ycent = np.mean(ylim)
            ax.set_xlim([xcent - alim, xcent + alim])
            ax.set_ylim([ycent - alim, ycent + alim])
            ax.set_aspect('equal')
            ax.set_title('Zoom in astrometry', fontsize=16)
            ax.minorticks_on()
            ax.set_xlabel(r'$\Delta$RA (mas)', fontsize=18)

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)

        if pdf is not None:
            pdf.close()
        
        self._logger.info('OPD_MET_ZERO_FC: ')
        for i, p in enumerate(self._pol_list):
            self._logger.info(', '.join([f'{v:.2f}' for v in self._fczp[:, i]]))

        self._logger.info("Pipeline completed!")




#class SciVisFits_old(object):
#    '''
#    A class to read and plot GRAVITY SCIVIS data.
#    '''
#    def __init__(self, filename):
#        '''
#        Parameters
#        ----------
#        filename : str
#            The name of the OIFITS file.
#        '''
#        self._filename = filename
#        self._hdul = fits.open(filename)
#        header = self._hdul[0].header
#        self._header = header
#
#        self._arcfile = header.get('ARCFILE', None)
#        if self._arcfile is not None:
#            self._arctime = self._arcfile.split('.')[1]
#        else:
#            self._arctime = None
#
#        self._object = header.get('OBJECT', None)
#
#        # Instrument mode
#        self._pol = header.get('ESO INS POLA MODE', None)
#        self._res = header.get('ESO INS SPEC RES', None)
#
#        # Science target
#        self._sobj_x = header.get('ESO INS SOBJ X', None)
#        self._sobj_y = header.get('ESO INS SOBJ Y', None)
#        self._sobj_offx = header.get('ESO INS SOBJ OFFX', None)
#        self._sobj_offy = header.get('ESO INS SOBJ OFFY', None)
#        self._swap = header.get('ESO INS SOBJ SWAP', None) == 'YES'
#        self._dit = header.get('ESO DET2 SEQ1 DIT', None)
#        self._ndit = header.get('ESO DET2 NDIT', None)
#
#        telescop = header.get('TELESCOP', None)
#        if 'U1234' in telescop:
#            self.set_telescope('UT')
#        elif 'A1234' in telescop:
#            self.set_telescope('AT')
#        else:
#            self.set_telescope('GV')
#
#    
#    def get_extver(self, fiber='SC', polarization=None):
#        '''
#        Get the extver of the OIFITS HDU. The first digit is the fiber type 
#        (1 or 2), and the second digit is the polarization (0, 1, or 2).
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        
#        Returns
#        -------
#        extver : int
#            The extver of the OIFITS HDU.
#        '''
#        assert fiber in ['SC', 'FT'], 'fiber must be SC or FT.'
#
#        if polarization is None:
#            if self._pol == 'SPLIT':
#                polarization = 1
#            else:
#                polarization = 0
#        assert polarization in [0, 1, 2], 'polarization must be 0, 1, or 2.'
#
#        fiber_code = {'SC': 1, 'FT': 2}
#        extver = int(f'{fiber_code[fiber]}{polarization}')
#        return extver
#
#
#    def get_t3phi(self, fiber='SC', polarization=None):
#        '''
#        Get the T3PHI and T3PHIERR of the OIFITS HDU.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        
#        Returns
#        -------
#        t3phi, t3phierr : masked arrays
#            The T3PHI and T3PHIERR of the OIFITS HDU. The shape is (N_TRIANGLE, N_CHANNEL).
#        '''
#        extver = self.get_extver(fiber, polarization)
#        flag = self._hdul['OI_T3', extver].data['FLAG']
#        t3phi = np.ma.array(self._hdul['OI_T3', extver].data['T3PHI'], mask=flag)
#        t3phierr = np.ma.array(self._hdul['OI_T3', extver].data['T3PHIERR'], mask=flag)
#        t3phi = np.reshape(t3phi, (-1, N_TRIANGLE, t3phi.shape[1]))
#        t3phierr = np.reshape(t3phierr, (-1, N_TRIANGLE, t3phierr.shape[1]))
#        return t3phi, t3phierr
#    
#    
#    def get_t3_uvcoord(self, fiber='SC', polarization=None, units='Mlambda'):
#        '''
#        Get the uv coordinates of the triangles. It returns two uv coordinates. 
#        The third uv coordinate can be calculated as -u1-u2 and -v1-v2.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the uv coordinates. Either Mlambda, per mas, or m.
#        
#        Returns
#        -------
#        u1coord, v1coord, u2coord, v2coord : arrays
#            The uv coordinate of the triangles. 
#            If units=m, the shape is (N_TRIANGLE, ), otherwise (N_TRIANGLE, N_CHANNEL).
#        '''
#        assert units in ['Mlambda', 'per mas', 'm'], 'units must be Mlambda, per mas, or m.'
#
#        extver = self.get_extver(fiber, polarization)
#        u1coord = self._hdul['OI_T3', extver].data['U1COORD'].reshape(-1, N_TRIANGLE)
#        v1coord = self._hdul['OI_T3', extver].data['V1COORD'].reshape(-1, N_TRIANGLE)
#        u2coord = self._hdul['OI_T3', extver].data['U2COORD'].reshape(-1, N_TRIANGLE)
#        v2coord = self._hdul['OI_T3', extver].data['V2COORD'].reshape(-1, N_TRIANGLE)
#        
#        if units != 'm':
#            wave = self.get_wavelength(fiber, polarization, units='micron')
#            u1coord = u1coord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
#            v1coord = v1coord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
#            u2coord = u2coord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
#            v2coord = v2coord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
#            
#            if units == 'per mas':
#                u1coord = u1coord * np.pi / 180. / 3600 * 1e3
#                v1coord = v1coord * np.pi / 180. / 3600 * 1e3
#                u2coord = u2coord * np.pi / 180. / 3600 * 1e3
#                v2coord = v2coord * np.pi / 180. / 3600 * 1e3
#        
#        return u1coord, v1coord, u2coord, v2coord
#
#
#    def get_uvdist(self, fiber='SC', polarization=None, units='Mlambda'):
#        '''
#        Get the uv distance of the baselines.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the uv coordinates. Either Mlambda, per mas, or m.
#
#        Returns
#        -------
#        uvdist : array
#            The uv distance of the baselines. If units=m, the shape is (N_BASELINE, ), 
#            otherwise (N_BASELINE, N_CHANNEL).
#        '''
#        u, v = self.get_vis_uvcoord(fiber, polarization, units=units)
#        uvdist = np.sqrt(u**2 + v**2)
#        return uvdist
#    
#
#    def get_uvmax(self, fiber='SC', polarization=None, units='Mlambda'):
#        '''
#        Get the maximum uv distance of the triangles.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the uv coordinates. Either Mlambda, per mas, or m.
#
#        Returns
#        -------
#        uvmax : array
#            The maximum uv distance of the triangles. If units=m, the shape is (N_TRIANGLE, ), 
#            otherwise (N_TRIANGLE, N_CHANNEL).
#        '''
#        u1, v1, u2, v2 = self.get_t3_uvcoord(fiber, polarization, units=units)
#        u3 = -u1 - u2
#        v3 = -v1 - v2
#        uvmax = np.max([np.sqrt(u1**2 + v1**2), 
#                        np.sqrt(u2**2 + v2**2), 
#                        np.sqrt(u3**2 + v3**2)], axis=0)
#        return uvmax
#
#
#    def get_vis_uvcoord(self, fiber='SC', polarization=None, units='m'):
#        '''
#        Get the u and v coordinates of the baselines.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the uv coordinates. Either Mlambda, per mas, or m.
#        
#        Returns
#        -------
#        ucoord, vcoord : arrays
#            The uv coordinate of the baselines. 
#            If units=m, the shape is (N_BASELINE, ), otherwise (N_BASELINE, N_CHANNEL).
#        '''
#        assert units in ['Mlambda', 'per mas', 'm'], 'units must be Mlambda, per mass, or m.'
#
#        extver = self.get_extver(fiber, polarization)
#        ucoord = self._hdul['OI_VIS', extver].data['UCOORD'].reshape(-1, N_BASELINE)
#        vcoord = self._hdul['OI_VIS', extver].data['VCOORD'].reshape(-1, N_BASELINE)
#        
#        if units != 'm':
#            wave = self.get_wavelength(fiber, polarization, units='micron')
#            ucoord = ucoord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
#            vcoord = vcoord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
#            
#            if units == 'per mas':
#                ucoord = ucoord * np.pi / 180. / 3600 * 1e3
#                vcoord = vcoord * np.pi / 180. / 3600 * 1e3
#        
#        return ucoord, vcoord
#    
#
#    def get_visamp(self, fiber='SC', polarization=None, fromdata=False):
#        '''
#        Get the VISAMP and VISAMPERR of the OIFITS HDU.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for
#            COMBINED and SPLIT, respectively.
#        
#        Returns
#        -------
#        visamp, visamperr : masked arrays
#            The VISAMP and VISAMPERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
#        '''
#        extver = self.get_extver(fiber, polarization)
#
#        if hasattr(self, f'_visamp_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_visamp_{extver}'), self.__getattribute__(f'_visamperr_{extver}')
#
#        flag = self._hdul['OI_VIS', extver].data['FLAG']
#        visamp = np.ma.array(self._hdul['OI_VIS', extver].data['VISAMP'], mask=flag)
#        visamperr = np.ma.array(self._hdul['OI_VIS', extver].data['VISAMPERR'], mask=flag)
#        visamp = np.reshape(visamp, (-1, N_BASELINE, visamp.shape[1]))
#        visamperr = np.reshape(visamperr, (-1, N_BASELINE, visamperr.shape[1]))
#
#        self.__setattr__(f'_visamp_{extver}', visamp)
#        self.__setattr__(f'_visamperr_{extver}', visamperr)
#
#        return visamp, visamperr
#    
#
#    def get_visdata(self, fiber='SC', polarization=None, fromdata=False, per_exp=False):
#        '''
#        Get the VISDATA and VISERR of the OIFITS HDU.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for
#            COMBINED and SPLIT, respectively.
#        
#        Returns
#        -------
#        visdata, viserr : masked arrays
#            The VISDATA and VISERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
#        '''
#        extver = self.get_extver(fiber, polarization)
#
#        if hasattr(self, f'_visdata_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_visdata_{extver}'), self.__getattribute__(f'_viserr_{extver}')
#
#        flag = self._hdul['OI_VIS', extver].data['FLAG']
#        visdata = np.ma.array(self._hdul['OI_VIS', extver].data['VISDATA'], mask=flag)
#        viserr = np.ma.array(self._hdul['OI_VIS', extver].data['VISERR'], mask=flag)
#        visdata = np.reshape(visdata, (-1, N_BASELINE, visdata.shape[1]))
#        viserr = np.reshape(viserr, (-1, N_BASELINE, viserr.shape[1]))
#
#        if per_exp:
#            visdata /= self._dit * self._ndit
#            viserr /= self._dit * self._ndit
#
#        self.__setattr__(f'_visdata_{extver}', visdata)
#        self.__setattr__(f'_viserr_{extver}', viserr)
#
#        return visdata, viserr
#    
#
#    def get_visphi(self, fiber='SC', polarization=None, fromdata=False):
#        '''
#        Get the VISPHI and VISPHIERR of the OIFITS HDU.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for
#            COMBINED and SPLIT, respectively.
#        
#        Returns
#        -------
#        visphi, visphierr : masked arrays
#            The VISPHI and VISPHIERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
#        '''
#        extver = self.get_extver(fiber, polarization)
#
#        if hasattr(self, f'_visphi_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_visphi_{extver}'), self.__getattribute__(f'_visphierr_{extver}')
#
#        flag = self._hdul['OI_VIS', extver].data['FLAG']
#        visphi = np.ma.array(self._hdul['OI_VIS', extver].data['VISPHI'], mask=flag)
#        visphierr = np.ma.array(self._hdul['OI_VIS', extver].data['VISPHIERR'], mask=flag)
#        visphi = np.reshape(visphi, (-1, N_BASELINE, visphi.shape[1]))
#        visphierr = np.reshape(visphierr, (-1, N_BASELINE, visphierr.shape[1]))
#
#        self.__setattr__(f'_visphi_{extver}', visphi)
#        self.__setattr__(f'_visphierr_{extver}', visphierr)
#
#        return visphi, visphierr
#
#
#    def get_vis2data(self, fiber='SC', polarization=None, fromdata=False):
#        '''
#        Get the VIS2DATA and VIS2ERR of the OIFITS HDU.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for
#            COMBINED and SPLIT, respectively.
#
#        Returns
#        -------
#        vis2data, vis2err : masked arrays
#            The VIS2DATA and VIS2ERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
#        '''
#        extver = self.get_extver(fiber, polarization)
#
#        if hasattr(self, f'_vis2data_{extver}') & hasattr(self, f'_vis2err_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_vis2data_{extver}'), self.__getattribute__(f'_vis2err_{extver}')
#
#        flag = self._hdul['OI_VIS', extver].data['FLAG']
#        vis2data = np.ma.array(self._hdul['OI_VIS2', extver].data['VIS2DATA'], mask=flag)
#        vis2err = np.ma.array(self._hdul['OI_VIS2', extver].data['VIS2ERR'], mask=flag)
#        vis2data = np.reshape(vis2data, (-1, N_BASELINE, vis2data.shape[1]))
#        vis2err = np.reshape(vis2err, (-1, N_BASELINE, vis2err.shape[1]))
#
#        self.__setattr__(f'_vis2data_{extver}', vis2data)
#        self.__setattr__(f'_vis2err_{extver}', vis2err)
#
#        return vis2data, vis2err
#
#
#    def get_wavelength(self, fiber='SC', polarization=None, units='micron'):
#        '''
#        Get the wavelength of the OIFITS HDU.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the wavelength. Either micron or m.
#        
#        Returns
#        -------
#        wave : array
#            The wavelength of the OIFITS HDU. If units=m, the shape is (N_CHANNEL, ), 
#            otherwise (N_CHANNEL, ).
#        '''
#        assert units in ['micron', 'm'], 'units must be um or m.'
#
#        extver = self.get_extver(fiber, polarization)
#        wave = self._hdul['OI_WAVELENGTH', extver].data['EFF_WAVE']
#
#        if units == 'micron':
#            wave = wave * 1e6
#
#        return wave
#
#
#    def plot_t3phi_ruv(self, fiber='SC', polarization=None, units='Mlambda', 
#                       show_average=False, ax=None, plain=False, legend_kwargs=None, 
#                       **kwargs):
#        '''
#        Plot the T3PHI as a function of the uv distance.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the uv distance. Either Mlambda or per mas.
#        show_average : bool, optional
#            If True, the average T3PHI will be shown. Default is False.
#        ax : matplotlib axis, optional
#            The axis to plot the T3PHI as a function of the uv distance. If None, 
#            a new figure and axis will be created.
#        plain : bool, optional
#            If True, the axis labels and legend will not be plotted.
#        legend_kwargs : dict, optional
#            The keyword arguments of the legend.
#        kwargs : dict, optional
#            The keyword arguments of the errorbar plot.
#
#        Returns
#        -------
#        ax : matplotlib axis
#            The axis of the T3PHI as a function of the uv distance.
#        '''
#        assert units in ['Mlambda', 'per mas'], 'The units must be Mlambda or per mas.'
#
#        if ax is None:
#            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#
#        uvdist = self.get_uvmax(fiber, polarization, units=units)
#        t3phi, t3phierr = self.get_t3phi(fiber, polarization)
#
#        if 'alpha' not in kwargs:
#            kwargs['alpha'] = 0.5
#        
#        if 'marker' not in kwargs:
#            kwargs['marker'] = 'o'
#        
#        if 'ecolor' not in kwargs:
#            kwargs['ecolor'] = 'gray'
#
#        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
#            kwargs['ls'] = '-'
#
#        for trl in range(N_TRIANGLE):
#            kwargs_use = kwargs.copy()
#
#            if 'color' not in kwargs:
#                kwargs_use['color'] = f'C{trl}'
#
#            for dit in range(t3phi.shape[0]):
#
#                if 'label' not in kwargs:
#                    if dit == 0:
#                        kwargs_use['label'] = f'{self._triangle[trl]}'
#                    else:
#                        kwargs_use['label'] = None
#
#                ax.errorbar(
#                    uvdist[dit, trl], t3phi[dit, trl, :], yerr=t3phierr[dit, trl, :], 
#                    **kwargs_use)
#
#        if show_average:
#            t3ave = np.nanmean(t3phi[0, :, :], axis=-1)
#            t3std = np.nanstd(t3phi[0, :, :], axis=-1)
#            t3phi_text = '\n'.join([fr'{self._triangle[ii]}: {t3ave[ii]:.1f} +/- {t3std[ii]:.1f}$^\circ$' 
#                                    for ii in range(N_TRIANGLE)])
#            ax.text(0.45, 0.95, t3phi_text, fontsize=14, transform=ax.transAxes, ha='left', va='top', 
#                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
#
#        if not plain:
#            if units == 'Mlambda':
#                ax.set_xlabel(r'$uv$ distance (M$\lambda$)', fontsize=24)
#            else:
#                ax.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24)
#            ax.set_ylabel(r'T3PHI ($^\circ$)', fontsize=24)
#            ax.minorticks_on()
#
#            if legend_kwargs is None:
#                legend_kwargs = {}
#
#            if 'fontsize' not in legend_kwargs:
#                legend_kwargs['fontsize'] = 14
#
#            if 'loc' not in legend_kwargs:
#                legend_kwargs['loc'] = 'upper left'
#
#            if 'ncols' not in legend_kwargs:
#                legend_kwargs['ncols'] = 2
#
#            ax.legend(**legend_kwargs)
#        return ax
#    
#
#    def plot_visamp_uvdist(self, fiber='SC', polarization=None, units='Mlambda',
#                           show_average=False, ax=None, plain=False, legend_kwargs=None, 
#                           **kwargs):
#        '''
#        Plot the VISAMP as a function of the uv distance.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the uv distance. Either Mlambda or per mas.
#        show_average : bool, optional
#            If True, the average VISAMP will be shown. Default is False.
#        ax : matplotlib axis, optional
#            The axis to plot the VISAMP as a function of the uv distance. If None, 
#            a new figure and axis will be created.
#        plain : bool, optional
#            If True, the axis labels and legend will not be plotted.
#        legend_kwargs : dict, optional
#            The keyword arguments of the legend.
#        kwargs : dict, optional
#            The keyword arguments of the errorbar plot.
#                
#        Returns
#        -------
#        ax : matplotlib axis
#            The axis of the VISAMP as a function of the uv distance.
#        '''
#        assert units in ['Mlambda', 'per mas'], 'The units must be Mlambda or per mas.'
#
#        if ax is None:
#            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#
#        uvdist = self.get_uvdist(fiber, polarization, units=units)
#        visamp, visamperr = self.get_visamp(fiber, polarization)
#
#        if 'alpha' not in kwargs:
#            kwargs['alpha'] = 0.5
#
#        if 'marker' not in kwargs:
#            kwargs['marker'] = 'o'
#
#        if 'ecolor' not in kwargs:
#            kwargs['ecolor'] = 'gray'
#
#        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
#            kwargs['ls'] = '-'
#            
#        for bsl in range(N_BASELINE):
#            kwargs_use = kwargs.copy()
#
#            if 'color' not in kwargs:
#                kwargs_use['color'] = f'C{bsl}'
#
#            for dit in range(visamp.shape[0]):
#
#                if 'label' not in kwargs:
#                    if dit == 0:
#                        kwargs_use['label'] = f'{self._baseline[bsl]}'
#                    else:
#                        kwargs_use['label'] = None
#
#                ax.errorbar(
#                    uvdist[dit, bsl], visamp[dit, bsl, :], yerr=visamperr[dit, bsl, :], 
#                    **kwargs_use)
#                
#        if show_average:
#            visave = np.nanmean(visamp[0, :, :], axis=-1)
#            visstd = np.nanstd(visamp[0, :, :], axis=-1)
#            visamp_text = '\n'.join([fr'{self._baseline[ii]}: {visave[ii]:.1f} +/- {visstd[ii]:.1f}' 
#                                    for ii in range(N_BASELINE)])
#            ax.text(0.45, 0.95, visamp_text, fontsize=14, transform=ax.transAxes, ha='left', va='top',
#                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
#            
#        if not plain:
#            if units == 'Mlambda':
#                ax.set_xlabel(r'$uv$ distance (M$\lambda$)', fontsize=24)
#            else:
#                ax.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24)
#            ax.set_ylabel('VISAMP', fontsize=24)
#            ax.minorticks_on()
#
#            if legend_kwargs is None:
#                legend_kwargs = {}
#
#            if 'fontsize' not in legend_kwargs:
#                legend_kwargs['fontsize'] = 14
#
#            if 'loc' not in legend_kwargs:
#                legend_kwargs['loc'] = 'upper left'
#
#            if 'ncols' not in legend_kwargs:
#                legend_kwargs['ncols'] = 2
#
#            ax.legend(**legend_kwargs)
#
#        return ax
#    
#
#    def plot_visdata_wavelength(self, fiber='SC', polarization=None, axs=None, 
#                                plain=False, legend_kwargs=None, **kwargs):
#        '''
#        Plot the VISDATA as a function of the wavelength.
#
#        Parameters
#        ----------
#        '''
#        if axs is None:
#            fig, axs = plt.subplots(N_BASELINE, 1, figsize=(12, 6), sharex=True, 
#                                    sharey=True)
#
#        wave = self.get_wavelength(fiber, polarization)
#        visdata, viserr = self.get_visdata(fiber, polarization)
#
#        if 'alpha' not in kwargs:
#            kwargs['alpha'] = 0.5
#
#        if 'marker' not in kwargs:
#            kwargs['marker'] = 'o'
#
#        if 'ecolor' not in kwargs:
#            kwargs['ecolor'] = 'gray'
#
#        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
#            kwargs['ls'] = '-'
#            
#        for bsl in range(N_BASELINE):
#            kwargs_use = kwargs.copy()
#
#            ax = axs[bsl]
#            for dit in range(visdata.shape[0]):
#
#                if 'label' not in kwargs:
#                    if dit == 0:
#                        kwargs_use['label'] = f'{self._baseline[bsl]}'
#                    else:
#                        kwargs_use['label'] = None
#
#                if 'color' not in kwargs:
#                    kwargs_use['color'] = f'C0'
#
#                ax.errorbar(
#                    wave, visdata[dit, bsl, :].real, yerr=viserr[dit, bsl, :].real, **kwargs_use)
#
#                if 'color' not in kwargs:
#                    kwargs_use['color'] = f'C3'
#
#                ax.errorbar(
#                    wave, visdata[dit, bsl, :].imag, yerr=viserr[dit, bsl, :].imag, **kwargs_use)
#
#        if not plain:
#            fig.subplots_adjust(hspace=0, wspace=0)
#            axo = fig.add_subplot(111, frameon=False) # The out axis
#            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
#            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
#            axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=24, labelpad=25)
#            axo.set_title('VISDATA', fontsize=24)
#            ax.minorticks_on()
#
#            if legend_kwargs is None:
#                legend_kwargs = {}
#
#            if 'fontsize' not in legend_kwargs:
#                legend_kwargs['fontsize'] = 14
#
#            if 'loc' not in legend_kwargs:
#                legend_kwargs['loc'] = 'lower right'
#            
#            if 'bbox_to_anchor' not in legend_kwargs:
#                legend_kwargs['bbox_to_anchor'] = (1, 1)
#
#            if 'ncols' not in legend_kwargs:
#                legend_kwargs['ncols'] = 2
#
#            axs[0].legend(**legend_kwargs)
#        
#        return axs
#
#
#    def plot_visphi_uvdist(self, fiber='SC', polarization=None, units='Mlambda',
#                           show_average=False, ax=None, plain=False, legend_kwargs=None, 
#                           **kwargs):
#        '''
#        Plot the VISPHI as a function of the uv distance.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the uv distance. Either Mlambda or per mas.
#        show_average : bool, optional
#            If True, the average VISPHI will be shown. Default is False.
#        ax : matplotlib axis, optional
#            The axis to plot the VISPHI as a function of the uv distance. If None, 
#            a new figure and axis will be created.
#        plain : bool, optional
#            If True, the axis labels and legend will not be plotted.
#        legend_kwargs : dict, optional
#            The keyword arguments of the legend.
#        kwargs : dict, optional
#            The keyword arguments of the errorbar plot.
#                
#        Returns
#        -------
#        ax : matplotlib axis
#            The axis of the VISPHI as a function of the uv distance.
#        '''
#        assert units in ['Mlambda', 'per mas'], 'The units must be Mlambda or per mas.'
#
#        if ax is None:
#            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#
#        uvdist = self.get_uvdist(fiber, polarization, units=units)
#        visphi, visphierr = self.get_visphi(fiber, polarization)
#
#        if 'alpha' not in kwargs:
#            kwargs['alpha'] = 0.5
#
#        if 'marker' not in kwargs:
#            kwargs['marker'] = 'o'
#
#        if 'ecolor' not in kwargs:
#            kwargs['ecolor'] = 'gray'
#
#        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
#            kwargs['ls'] = '-'
#            
#        for bsl in range(N_BASELINE):
#            kwargs_use = kwargs.copy()
#
#            if 'color' not in kwargs:
#                kwargs_use['color'] = f'C{bsl}'
#
#            for dit in range(visphi.shape[0]):
#
#                if 'label' not in kwargs:
#                    if dit == 0:
#                        kwargs_use['label'] = f'{self._baseline[bsl]}'
#                    else:
#                        kwargs_use['label'] = None
#
#                ax.errorbar(
#                    uvdist[dit, bsl], visphi[dit, bsl, :], yerr=visphierr[dit, bsl, :], 
#                    **kwargs_use)
#                
#        if show_average:
#            visave = np.nanmean(visphi[0, :, :], axis=-1)
#            visstd = np.nanstd(visphi[0, :, :], axis=-1)
#            visphi_text = '\n'.join([fr'{self._baseline[ii]}: {visave[ii]:.1f} +/- {visstd[ii]:.1f}$^\circ$' 
#                                    for ii in range(N_BASELINE)])
#            ax.text(0.45, 0.95, visphi_text, fontsize=14, transform=ax.transAxes, ha='left', va='top',
#                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
#            
#        if not plain:
#            if units == 'Mlambda':
#                ax.set_xlabel(r'$uv$ distance (M$\lambda$)', fontsize=24)
#            else:
#                ax.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24)
#            ax.set_ylabel('VISPHI ($^\circ$)', fontsize=24)
#            ax.minorticks_on()
#
#            if legend_kwargs is None:
#                legend_kwargs = {}
#
#            if 'fontsize' not in legend_kwargs:
#                legend_kwargs['fontsize'] = 14
#
#            if 'loc' not in legend_kwargs:
#                legend_kwargs['loc'] = 'upper left'
#
#            if 'ncols' not in legend_kwargs:
#                legend_kwargs['ncols'] = 2
#
#            ax.legend(**legend_kwargs)
#
#        return
#
#    
#    def plot_vis2data_uvdist(self, fiber='SC', polarization=None, units='Mlambda',
#                             show_average=False, ax=None, plain=False, legend_kwargs=None, 
#                             **kwargs):
#        '''
#        Plot the VIS2 as a function of the uv distance.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the uv distance. Either Mlambda or per mas.
#        show_average : bool, optional
#            If True, the average VIS2 will be shown. Default is False.
#        ax : matplotlib axis, optional
#            The axis to plot the VIS2 as a function of the uv distance. If None, 
#            a new figure and axis will be created.
#        plain : bool, optional
#            If True, the axis labels and legend will not be plotted.
#        legend_kwargs : dict, optional
#            The keyword arguments of the legend.
#        kwargs : dict, optional
#            The keyword arguments of the errorbar plot.
#                
#        Returns
#        -------
#        ax : matplotlib axis
#            The axis of the VIS2 as a function of the uv distance.
#        '''
#        assert units in ['Mlambda', 'per mas'], 'The units must be Mlambda or per mas.'
#
#        if ax is None:
#            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#
#        uvdist = self.get_uvdist(fiber, polarization, units=units)
#        vis2data, vis2err = self.get_vis2data(fiber, polarization)
#
#        if 'alpha' not in kwargs:
#            kwargs['alpha'] = 0.5
#
#        if 'marker' not in kwargs:
#            kwargs['marker'] = 'o'
#
#        if 'ecolor' not in kwargs:
#            kwargs['ecolor'] = 'gray'
#
#        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
#            kwargs['ls'] = '-'
#            
#        for bsl in range(N_BASELINE):
#            kwargs_use = kwargs.copy()
#
#            if 'color' not in kwargs:
#                kwargs_use['color'] = f'C{bsl}'
#
#            for dit in range(vis2data.shape[0]):
#
#                if 'label' not in kwargs:
#                    if dit == 0:
#                        kwargs_use['label'] = f'{self._baseline[bsl]}'
#                    else:
#                        kwargs_use['label'] = None
#
#                ax.errorbar(
#                    uvdist[dit, bsl], vis2data[dit, bsl, :], yerr=vis2err[dit, bsl, :], 
#                    **kwargs_use)
#        
#        if show_average:
#            visave = np.nanmean(vis2data[0, :, :], axis=-1)
#            visstd = np.nanstd(vis2data[0, :, :], axis=-1)
#            visphi_text = '\n'.join([fr'{self._baseline[ii]}: {visave[ii]:.1f} +/- {visstd[ii]:.1f}$^\circ$' 
#                                    for ii in range(N_BASELINE)])
#            ax.text(0.45, 0.95, visphi_text, fontsize=14, transform=ax.transAxes, ha='left', va='top',
#                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
#            
#        if not plain:
#            if units == 'Mlambda':
#                ax.set_xlabel(r'$uv$ distance (M$\lambda$)', fontsize=24)
#            else:
#                ax.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24)
#            ax.set_ylabel('VIS2DATA', fontsize=24)
#            ax.minorticks_on()
#
#            if legend_kwargs is None:
#                legend_kwargs = {}
#
#            if 'fontsize' not in legend_kwargs:
#                legend_kwargs['fontsize'] = 14
#
#            if 'loc' not in legend_kwargs:
#                legend_kwargs['loc'] = 'upper left'
#
#            if 'ncols' not in legend_kwargs:
#                legend_kwargs['ncols'] = 2
#
#            ax.legend(**legend_kwargs)
#        return
#    
#
#    def set_telescope(self, telescope):
#        '''
#        Set the telescope input of GRAVITY. The telescope, baseline, and 
#        triangle names are adjusted accordingly.
#
#        Parameters
#        ----------
#        telescope : str
#            The telescope. Either UT, AT, or GV.
#        '''
#        assert telescope in ['UT', 'AT', 'GV'], 'telescope must be UT, AT, or GV.'
#        self._telescope = telescope_names[telescope]
#        self._baseline = baseline_names[telescope]
#        self._triangle = triangle_names[telescope]
#
#
#class AstroFits_old(object):
#    '''
#    A class to read and plot GRAVITY ASTROREDUCED data.
#    '''
#    def __init__(self, filename):
#        '''
#        Parameters
#        ----------
#        filename : str
#            The name of the OIFITS file.
#        '''
#        self._filename = filename
#        self._hdul = fits.open(filename)
#        header = self._hdul[0].header
#        self._header = header
#
#        self._arcfile = header.get('ARCFILE', None)
#        if self._arcfile is not None:
#            self._arctime = self._arcfile.split('.')[1]
#        else:
#            self._arctime = None
#
#        self._object = header.get('OBJECT', None)
#
#        # Instrument mode
#        self._pol = header.get('ESO INS POLA MODE', None)
#        self._res = header.get('ESO INS SPEC RES', None)
#
#        # Science target
#        self._sobj_x = header.get('ESO INS SOBJ X', None)
#        self._sobj_y = header.get('ESO INS SOBJ Y', None)
#        self._sobj_offx = header.get('ESO INS SOBJ OFFX', None)
#        self._sobj_offy = header.get('ESO INS SOBJ OFFY', None)
#        self._swap = header.get('ESO INS SOBJ SWAP', None) == 'YES'
#        self._dit = header.get('ESO DET2 SEQ1 DIT', None)
#
#        telescop = header.get('TELESCOP', None)
#        if 'U1234' in telescop:
#            self.set_telescope('UT')
#        elif 'A1234' in telescop:
#            self.set_telescope('AT')
#        else:
#            self.set_telescope('GV')
#
#
#    def copy(self):
#        '''
#        Copy the current AstroFits object.
#        '''
#        return deepcopy(self)
#    
#
#    def correct_met_jump(self, fringe_tel):
#        '''
#        Correct the metrology phase jump.
#
#        Parameters
#        ----------
#        opd_tel : array
#            The correction values add to OPD_DISP, in the unit of fringe.
#            The input value is telescope based, [UT4, UT3, UT2, UT1].
#        '''
#        wave = self.get_wavelength(units='micron')
#        corr_tel = np.array(fringe_tel)[:, np.newaxis] * 2*np.pi * (1 - lambda_met / wave)[np.newaxis, :]
#        self._opdDisp_corr = np.dot(t2b_matrix, corr_tel)
#
#
#    def diff_visphi(self, 
#                    oi : 'AstroFits', 
#                    polarization : int = None, 
#                    fromdata : bool = False, 
#                    opdzp : np.array = None, 
#                    per_dit : bool = False):
#        '''
#        Calculate the difference of the VISPHI between the current and another OIFITS HDU.
#        '''
#        visref_self = self.get_visref(polarization=polarization, fromdata=fromdata, opdzp=opdzp, per_dit=per_dit)
#        visref_oi = oi.get_visref(polarization=polarization, fromdata=fromdata, opdzp=opdzp, per_dit=per_dit)
#
#        visphi = np.angle(visref_self * np.conj(visref_oi), deg=True)
#        return visphi
#
#
#    def get_extver(self, polarization=None):
#        '''
#        Get the extver of the OIFITS HDU. The first digit is the fiber type 
#        (1 or 2), and the second digit is the polarization (0, 1, or 2).
#
#        Parameters
#        ----------
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        
#        Returns
#        -------
#        extver : int
#            The extver of the OIFITS HDU.
#        '''
#        if polarization is None:
#            if self._pol == 'SPLIT':
#                polarization = 1
#            else:
#                polarization = 0
#        assert polarization in [0, 1, 2], 'polarization must be 0, 1, or 2.'
#
#        extver = int(f'1{polarization}')
#        return extver
#    
#
#    def get_f1f2(self, polarization=None, fromdata=False):
#        '''
#        Get the geometric flux (F1F2) of the OIFITS HDU.
#        '''
#        extver = self.get_extver(polarization)
#
#        if hasattr(self, f'_f1f2_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_f1f2_{extver}')
#        
#        f1f2 = self._hdul['OI_VIS', extver].data['F1F2']
#        f1f2 = np.reshape(f1f2, (-1, N_BASELINE, f1f2.shape[1]))
#
#        flag = self.get_visflag(polarization, fromdata=fromdata)
#        f1f2 = np.ma.array(f1f2, mask=flag)
#
#        self.__setattr__(f'_f1f2_{extver}', f1f2)
#
#        return f1f2
#
#
#    def get_gdelay(self, polarization=None, fromdata=False, field='GDELAY_BOOT'):
#        '''
#        Get the GDELAY_BOOT.
#        '''
#        extver = self.get_extver(polarization)
#
#        if hasattr(self, f'_gdelay_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_gdelay_{extver}')
#        
#        gdelay = self._hdul['OI_VIS', extver].data[field].reshape(-1, N_BASELINE)
#
#        self.__setattr__(f'_gdelay_{extver}', gdelay)
#
#        return gdelay
#
#
#    def get_offset(self, opdzp, polarization=None, method='leastsq', fromdata=False, 
#                   kws_opd_visref={}, plot=False, **kwargs):
#        '''
#        Get the astrometry offset.
#        '''
#        opd = self.get_opd_visref(polarization=polarization, fromdata=fromdata, 
#                                  opdzp=opdzp, **kws_opd_visref)
#        uvcoord = self.get_vis_uvcoord(polarization=polarization, units='m', 
#                                       fromdata=fromdata)
#
#        if method == 'leastsq':
#            uvcoord = np.array(uvcoord).swapaxes(0, 1)
#            offset = np.mean(solve_offset(opd, uvcoord), axis=0)
#            chi2_grid = None
#            chi2_grid_zoom = None
#
#        elif method == 'grid':
#            if 'ra_init' not in kwargs:
#                kwargs['ra_init'] = self._sobj_x
#            if 'dec_init' not in kwargs:
#                kwargs['dec_init'] = self._sobj_y
#            
#            res = grid_search_opd(opd, uvcoord, plot=plot, **kwargs)
#
#            offset = np.array([res['ra_best'], res['dec_best']])
#            chi2_grid = res['chi2_grid']
#            chi2_grid_zoom = res['chi2_grid_zoom']
#        else:
#            raise ValueError(f'The method {method} is not supported!')
#        
#        if self._swap:
#            offset *= -1
#
#        res = dict(offset=offset, chi2_grid=chi2_grid, chi2_grid_zoom=chi2_grid_zoom)
#
#        return res
#
#
#    def get_opd_visref(self, polarization=None, fromdata=False, opdzp=None, opd_lim=3000, 
#                       step=1, zoom=20, iterations=2, progress=False, plot=False):
#        '''
#        Calculate the OPD from the VISREF data.
#        '''
#        extver = self.get_extver(polarization)
#        visref = self.get_visref(polarization, fromdata=fromdata, opdzp=opdzp)
#        visphi = np.angle(visref)
#        wave = self.get_wavelength(polarization=polarization, units='micron')
#
#        if progress:
#            iterate = tqdm(range(visphi.shape[0]))
#        else:
#            iterate = range(visphi.shape[0])
#
#        opd = np.array([
#            fit_opd_closure(visphi[dit, :, :], wave, opd_lim=opd_lim, step=step, 
#                            zoom=zoom, iterations=iterations, plot=plot)
#            for dit in iterate])
#        return opd
#
#
#    def get_opdDisp(self, polarization=None, fromdata=False):
#        '''
#        Get the OPD_DISP data from the OIFITS file.
#        '''
#        extver = self.get_extver(polarization)
#        attr_name = f'_opddisp_{extver}'
#
#        if hasattr(self, attr_name) & (not fromdata):
#            return self.__getattribute__(attr_name)
#        
#        opddisp = self._hdul['OI_VIS', extver].data['OPD_DISP']
#        opddisp = np.reshape(opddisp, (-1, N_BASELINE, opddisp.shape[1]))
#
#        self.__setattr__(attr_name, opddisp)
#
#        return opddisp
#    
#
#    def get_opdMetCorr(self, polarization=None, fromdata=False):
#        '''
#        '''
#        extver = self.get_extver(polarization)
#
#        if hasattr(self, f'_opdMetCorr_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_opdMetCorr_{extver}')
#
#        opdMetFcCorr = self.get_opdMetFcCorr(polarization=polarization, fromdata=fromdata)
#        opdMetTelFcMCorr = self.get_opdMetTelFcMCorr(polarization=polarization, fromdata=fromdata)
#        opdMetCorr = (opdMetFcCorr + opdMetTelFcMCorr)[:, :, np.newaxis]
#
#        self.__setattr__(f'_opdMetCorr_{extver}', opdMetCorr)
#
#        return opdMetCorr
#
#
#    def get_opdMetTelFcMCorr(self, polarization=None, fromdata=False):
#        '''
#        '''
#        extver = self.get_extver(polarization)
#
#        if hasattr(self, f'_opdMetTelFcMCorr_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_opdMetTelFcMCorr_{extver}')
#        
#        opdMetTelFcMCorr = self._hdul['OI_VIS', extver].data['OPD_MET_TELFC_MCORR'].reshape(-1, N_BASELINE)
#
#        return opdMetTelFcMCorr
#    
#
#    def get_opdMetFcCorr(self, polarization=None, fromdata=False):
#        '''
#        '''
#        extver = self.get_extver(polarization)
#
#        if hasattr(self, f'_opdMetFcCorr_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_opdMetFcCorr_{extver}')
#        
#        opdMetFcCorr = self._hdul['OI_VIS', extver].data['OPD_MET_FC_CORR'].reshape(-1, N_BASELINE)
#
#        return opdMetFcCorr
#
#
#    def get_phaseref(self, polarization=None, fromdata=False):
#        '''
#        '''
#        extver = self.get_extver(polarization)
#
#        if hasattr(self, f'_phaseref_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_phaseref_{extver}')
#        
#        phaseref = self._hdul['OI_VIS', extver].data['PHASE_REF']
#        phaseref = np.reshape(phaseref, (-1, N_BASELINE, phaseref.shape[1]))
#
#        self.__setattr__(f'_phaseref_{extver}', phaseref)
#
#        return phaseref
#
#
#    def get_uvdist(self, fiber='SC', polarization=None, units='Mlambda', fromdata=False):
#        '''
#        Get the uv distance of the baselines.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the uv coordinates. Either Mlambda, permas, or m.
#            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.
#
#        Returns
#        -------
#        uvdist : array
#            The uv distance of the baselines. If units=m, the shape is (N_BASELINE, ), 
#            otherwise (N_BASELINE, N_CHANNEL).
#        '''
#        u, v = self.get_vis_uvcoord(polarization, units=units, fromdata=fromdata)
#        uvdist = np.sqrt(u**2 + v**2)
#        return uvdist
#
#
#    def get_vis_uvcoord(self, polarization=None, units='m', fromdata=False):
#        '''
#        Get the u and v coordinates of the baselines.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the uv coordinates. Either Mlambda, permas, or m.
#            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.
#        
#        Returns
#        -------
#        ucoord, vcoord : arrays
#            The uv coordinate of the baselines. 
#            If units=m, the shape is (N_BASELINE, ), otherwise (N_BASELINE, N_CHANNEL).
#        '''
#        assert units in ['Mlambda', 'permas', 'm'], 'units must be Mlambda, per mass, or m.'
#
#        if hasattr(self, f'_vis_ucoord_{units}') & hasattr(self, f'_vis_vcoord_{units}') & (not fromdata):
#            return self.__getattribute__(f'_vis_ucoord_{units}'), self.__getattribute__(f'_vis_vcoord_{units}')
#
#        extver = self.get_extver(polarization)
#        ucoord = self._hdul['OI_VIS', extver].data['UCOORD'].reshape(-1, N_BASELINE)
#        vcoord = self._hdul['OI_VIS', extver].data['VCOORD'].reshape(-1, N_BASELINE)
#        
#        if units != 'm':
#            wave = self.get_wavelength(polarization, units='micron')
#            ucoord = ucoord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
#            vcoord = vcoord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
#            
#            if units == 'permas':
#                ucoord = ucoord * np.pi / 180. / 3600 * 1e3
#                vcoord = vcoord * np.pi / 180. / 3600 * 1e3
#
#        setattr(self, f'_vis_ucoord_{units}', ucoord)
#        setattr(self, f'_vis_vcoord_{units}', vcoord)
#        
#        return ucoord, vcoord
#
#
#    def get_visflag(self, polarization=None, fromdata=False):
#        '''
#        '''
#        extver = self.get_extver(polarization)
#
#        if hasattr(self, f'_visflag_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_visflag_{extver}')
#        
#        flag = self._hdul['OI_VIS', extver].data['FLAG']
#        try:
#            rejflag  = self._hdul['OI_VIS', extver].data('REJECTION_FLAG')
#        except:
#            rejflag = np.zeros_like(flag)
#        flag = flag | ((rejflag & 19) > 0)
#        flag = np.reshape(flag, (-1, N_BASELINE, flag.shape[1]))
#
#        self.__setattr__(f'_visflag_{extver}', flag)
#
#        return flag
#
#
#    def get_visdata(self, polarization=None, fromdata=False):
#        '''
#        Get the VISDATA and VISERR of the OIFITS HDU.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for
#            COMBINED and SPLIT, respectively.
#        
#        Returns
#        -------
#        visdata, viserr : masked arrays
#            The VISDATA and VISERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
#        '''
#        extver = self.get_extver(polarization)
#
#        if hasattr(self, f'_visdata_{extver}') & hasattr(self, f'_viserr_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_visdata_{extver}'), self.__getattribute__(f'_viserr_{extver}')
#
#        visdata = self._hdul['OI_VIS', extver].data['VISDATA']
#        viserr = self._hdul['OI_VIS', extver].data['VISERR']
#        visdata = np.reshape(visdata, (-1, N_BASELINE, visdata.shape[1]))
#        viserr = np.reshape(viserr, (-1, N_BASELINE, viserr.shape[1]))
#        
#        flag = self.get_visflag(polarization, fromdata=fromdata)
#        visdata = np.ma.array(visdata, mask=flag)
#        viserr = np.ma.array(viserr, mask=flag)
#
#        self.__setattr__(f'_visdata_{extver}', visdata)
#        self.__setattr__(f'_viserr_{extver}', viserr)
#
#        return visdata, viserr
#
#
#    def get_visref(self, polarization=None, fromdata=False, opdzp=None, 
#                   per_dit=False, normalized=False, exoplanet=True):
#        '''
#        Get the phase referenced VISDATA (VISREF) for astrometry measurement.
#        '''
#        extver = self.get_extver(polarization)
#
#        attr_name = f'_visref_{extver}'
#
#        if hasattr(self, '_opdDisp_corr'):
#            attr_name += '_metcorr'
#        
#        if exoplanet:
#            attr_name += '_exop'
#
#        if per_dit:
#            attr_name += '_perdit'
#        
#        if normalized:
#            attr_name += '_normalized'
#
#        if opdzp is not None:
#            attr_name += f'_zpcorr'
#
#        if hasattr(self, attr_name) & (not fromdata):
#            return self.__getattribute__(attr_name)
#        
#        wave = self.get_wavelength(polarization=polarization, fromdata=fromdata, units='micron')
#        visdata, _ = self.get_visdata(polarization=polarization, fromdata=fromdata)
#        phaseref= self.get_phaseref(polarization=polarization, fromdata=fromdata)
#        opdDisp = self.get_opdDisp(polarization=polarization, fromdata=fromdata)
#        opdMetCorr = self.get_opdMetCorr(polarization=polarization, fromdata=fromdata)
#
#        if exoplanet:
#            opdSep = 0
#        else:
#            u, v = self.get_vis_uvcoord(polarization=polarization, units='m', fromdata=fromdata)
#            opdSep = np.pi / 180 / 3600 / 1e3 * (u * self._sobj_x + v * self._sobj_y)[:, :, np.newaxis]
#
#        visref = visdata * np.exp(1j * (phaseref + 2*np.pi / wave * (opdSep - opdDisp - opdMetCorr) * 1e6))
#
#        if hasattr(self, '_opdDisp_corr'):
#            visref *= np.exp(1j * self._opdDisp_corr)
#
#        if opdzp is not None:
#            v_opd = matrix_opd(opdzp, wave).T
#            visref *= np.conj(v_opd[np.newaxis, :, :])
#
#        if per_dit:
#            visref /= self._dit
#
#        if normalized:
#            f1f2 = self.get_f1f2(polarization=polarization, fromdata=fromdata)
#            visref /= np.sqrt(f1f2)
#        
#        self.__setattr__(f'_visref_{extver}', visref)
#
#        return visref
#
#
#    def get_visref_rephase(self, phi0=0, radec=None, polarization=None, 
#                           fromdata=False, per_dit=False, normalized=False, 
#                           plot=False, axs=None, plain=False):
#        '''
#        Get the re-phased VISREF data to check the astrometry measurement.
#        '''
#        visref = self.get_visref(polarization=polarization, fromdata=fromdata, 
#                                 per_dit=per_dit, normalized=normalized)
#
#        if radec is not None:
#            ra, dec = radec
#            u, v = self.get_vis_uvcoord(polarization=polarization, units='Mlambda', fromdata=fromdata)
#            phase_source = phase_model(u, v, ra, dec)
#        else:
#            phase_source = 0
#
#        visref_rephase = visref * np.exp(1j * (phase_source - phi0))
#
#        if plot:
#            uvdist = self.get_uvdist(polarization=polarization, units='permas', fromdata=fromdata)
#            x = uvdist.mean(axis=0)
#            y = visref_rephase.mean(axis=0)
#
#            if axs is None:
#                fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
#                fig.subplots_adjust(hspace=0.02)
#            else:
#                assert len(axs) == 2, 'axs must have 2 axes.'
#
#            for bsl in range(N_BASELINE):
#                ax = axs[0]
#                ax.plot(x[bsl, :], np.absolute(y[bsl, :]), color=f'C{bsl}', label=self._baseline[bsl])
#
#                if not plain:
#                    ax.legend(loc='lower left', ncols=3, fontsize=14, handlelength=1, columnspacing=1)
#                    ax.set_ylabel('VISAMP', fontsize=16)
#                    ax.set_ylim([0, 1.1])
#                    ax.minorticks_on()
#
#                ax = axs[1]
#                ax.plot(x[bsl, :], np.angle(y[bsl, :], deg=True), color=f'C{bsl}')
#                
#                if not plain:
#                    ax.set_ylabel(r'VISPHI ($^\circ$)', fontsize=16)
#                    ax.set_ylim([-180, 180])
#                    ax.set_xlabel(r'Space frequency (mas$^{-1}$)', fontsize=16)
#                    ax.minorticks_on()
#        
#        return visref_rephase
#
#
#    def get_wavelength(self, polarization=None, units='micron', fromdata=False):
#        '''
#        Get the wavelength of the OIFITS HDU.
#
#        Parameters
#        ----------
#        fiber : str
#            The fiber type. Either SC or FT.
#        polarization : int, optional
#            The polarization. If None, the polarization 0 and 1 will be used for 
#            COMBINED and SPLIT, respectively.
#        units : str, optional
#            The units of the wavelength. Either micron or m.
#        
#        Returns
#        -------
#        wave : array
#            The wavelength of the OIFITS HDU. If units=m, the shape is (N_CHANNEL, ), 
#            otherwise (N_CHANNEL, ).
#        '''
#        assert units in ['micron', 'm'], 'units must be um or m.'
#
#        extver = self.get_extver(polarization)
#
#        if hasattr(self, f'_wave_{extver}') & (not fromdata):
#            return self.__getattribute__(f'_wave_{extver}')
#
#        wave = self._hdul['OI_WAVELENGTH', extver].data['EFF_WAVE']
#
#        if units == 'micron':
#            wave = wave * 1e6
#
#        self.__setattr__(f'_wave_{extver}', wave)
#
#        return wave
#
#
#    def plot_rdata_wavelength(self, data, name=None, polarization=None, axs=None, 
#                              plain=False, legend_kwargs=None, **kwargs):
#        '''
#        Plot the real data as a function of the wavelength.
#
#        Parameters
#        ----------
#        '''
#        if axs is None:
#            fig, axs = plt.subplots(N_BASELINE, 1, figsize=(12, 6), sharex=True, 
#                                    sharey=True)
#
#        wave = self.get_wavelength(polarization, units='micron')
#
#        if 'alpha' not in kwargs:
#            kwargs['alpha'] = 0.5
#
#        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
#            kwargs['ls'] = '-'
#            
#        for bsl in range(N_BASELINE):
#            kwargs_use = kwargs.copy()
#
#            ax = axs[bsl]
#            for dit in range(data.shape[0]):
#
#                if 'label' not in kwargs:
#                    if dit == 0:
#                        kwargs_use['label'] = f'{self._baseline[bsl]}'
#                    else:
#                        kwargs_use['label'] = None
#
#                if 'color' not in kwargs:
#                    kwargs_use['color'] = f'C{dit}'
#
#                ax.plot(wave, data[dit, bsl, :], **kwargs_use)
#
#        if not plain:
#            fig.subplots_adjust(hspace=0, wspace=0)
#            axo = fig.add_subplot(111, frameon=False) # The out axis
#            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
#            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
#            axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=24, labelpad=25)
#            axo.set_title(name, fontsize=24)
#            ax.minorticks_on()
#
#            if legend_kwargs is None:
#                legend_kwargs = {}
#
#            if 'fontsize' not in legend_kwargs:
#                legend_kwargs['fontsize'] = 14
#
#            if 'loc' not in legend_kwargs:
#                legend_kwargs['loc'] = 'lower right'
#            
#            if 'bbox_to_anchor' not in legend_kwargs:
#                legend_kwargs['bbox_to_anchor'] = (1, 1)
#
#            if 'ncols' not in legend_kwargs:
#                legend_kwargs['ncols'] = 2
#
#            axs[0].legend(**legend_kwargs)
#        
#        return axs
#
#    
#    def plot_cdata_wavelength(self, data, name=None, polarization=None, axs=None, 
#                              plain=False, legend_kwargs=None, **kwargs):
#        '''
#        Plot the complex data as a function of the wavelength.
#
#        Parameters
#        ----------
#        '''
#        if axs is None:
#            fig, axs = plt.subplots(N_BASELINE, 1, figsize=(12, 6), sharex=True, 
#                                    sharey=True)
#
#        wave = self.get_wavelength(polarization, units='micron')
#
#        if 'alpha' not in kwargs:
#            kwargs['alpha'] = 0.5
#
#        if 'marker' not in kwargs:
#            kwargs['marker'] = 'o'
#
#        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
#            kwargs['ls'] = '-'
#            
#        for bsl in range(N_BASELINE):
#            kwargs_use = kwargs.copy()
#
#            ax = axs[bsl]
#            for dit in range(data.shape[0]):
#
#                if 'label' not in kwargs:
#                    if dit == 0:
#                        kwargs_use['label'] = f'{self._baseline[bsl]}'
#                    else:
#                        kwargs_use['label'] = None
#
#                if 'color' not in kwargs:
#                    kwargs_use['color'] = f'C0'
#
#                ax.plot(
#                    wave, data[dit, bsl, :].real, **kwargs_use)
#
#                if 'color' not in kwargs:
#                    kwargs_use['color'] = f'C3'
#
#                ax.errorbar(
#                    wave, data[dit, bsl, :].imag, **kwargs_use)
#
#        if not plain:
#            fig.subplots_adjust(hspace=0, wspace=0)
#            axo = fig.add_subplot(111, frameon=False) # The out axis
#            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
#            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
#            axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=24, labelpad=25)
#            axo.set_title(name, fontsize=24)
#            ax.minorticks_on()
#
#            if legend_kwargs is None:
#                legend_kwargs = {}
#
#            if 'fontsize' not in legend_kwargs:
#                legend_kwargs['fontsize'] = 14
#
#            if 'loc' not in legend_kwargs:
#                legend_kwargs['loc'] = 'lower right'
#            
#            if 'bbox_to_anchor' not in legend_kwargs:
#                legend_kwargs['bbox_to_anchor'] = (1, 1)
#
#            if 'ncols' not in legend_kwargs:
#                legend_kwargs['ncols'] = 2
#
#            axs[0].legend(**legend_kwargs)
#        
#        return axs
#
#
#    def set_telescope(self, telescope):
#        '''
#        Set the telescope input of GRAVITY. The telescope, baseline, and 
#        triangle names are adjusted accordingly.
#
#        Parameters
#        ----------
#        telescope : str
#            The telescope. Either UT, AT, or GV.
#        '''
#        assert telescope in ['UT', 'AT', 'GV'], 'telescope must be UT, AT, or GV.'
#        self._telescope = telescope_names[telescope]
#        self._baseline = baseline_names[telescope]
#        self._triangle = triangle_names[telescope]
#
#
#
#    def compute_metzp_opd(
#            self, 
#            opd_lim : float = 3000, 
#            step : float = 1, 
#            zoom : float = 20, 
#            iterations : int = 2, 
#            progress : bool = False,
#            verbose=True) -> dict:
#        '''
#        Compute the metrology zero points from the OPD.
#        Note that the opdzp of two polarizations are different!
#        '''
#        opd1List = []
#        opd2List = []
#        for p in self._pol_list:
#            # Unswapped data
#            opd1List.append([oi.cal_opd_visdata(
#                polarization=p, opd_lim=opd_lim, step=step, 
#                zoom=zoom, iterations=iterations, progress=progress, 
#                plot=False, verbose=verbose) 
#                for oi in self[self._index_unswap]])
#        
#            # Swapped data
#            opd2List.append([oi.cal_opd_visdata(
#                polarization=p, opd_lim=opd_lim, step=step, 
#                zoom=zoom, iterations=iterations, progress=progress, 
#                plot=False, verbose=verbose) 
#                for oi in self[self._index_swap]])
#        opd1List = np.array(opd1List).mean(axis=(1, 2))
#        opd2List = np.array(opd2List).mean(axis=(1, 2))
#
#        self._opdzp = (opd1List + opd2List) / 2
#        self._fczp = np.array([self._opdzp[:, 2], 
#                               self._opdzp[:, 4], 
#                               self._opdzp[:, 5], 
#                               np.zeros(self._opdzp.shape[0])])
#
#        return self._opdzp, self._fczp
#    
#
#    def cal_offset_opd(self, polarization=None, average=True, verbose=True, **kwargs):
#        '''
#        Calculate the offset using the pseudo-inverse method.
#        '''
#        opd = self.cal_opd_visdata(polarization=polarization, verbose=verbose, **kwargs)
#        uvcoord = np.squeeze(self._uvcoord_m).swapaxes(0, 1)
#        offset = solve_offset(opd, uvcoord)
#
#        if self._swap:
#            offset *= -1
#
#        if average:
#            offset = np.mean(offset, axis=0)
#
#        return offset
#
#
#    def cal_opd_visdata(self, polarization=None, opd_lim=3000, step=1, zoom=20, 
#                        iterations=2, progress=False, plot=False, verbose=True):
#     '''
#     Calculate the OPD from the VISREF data.
#     '''
#     extver = self.get_extver(fiber='SC', polarization=polarization)
#     #visdata = getattr(self, f'_visdata_{extver}')
#     #opd = compute_gdelay(visdata, self._wave, max_width=2000)
#     visphi = np.angle(getattr(self, f'_visdata_{extver}'))
#
#     if progress:
#         iterate = tqdm(range(visphi.shape[0]))
#     else:
#         iterate = range(visphi.shape[0])
#
#     opd = np.array([
#         fit_opd_closure(visphi[dit, :, :], self._wave, opd_lim=opd_lim, step=step, 
#                         zoom=zoom, iterations=iterations, plot=plot, verbose=verbose)
#         for dit in iterate])
#     return opd
#
#
#    def cal_offset_opd(
#            self, 
#            verbose : bool = True, 
#            **kwargs) -> np.ndarray:
#        '''
#        Calculate the offset using the pseudo-inverse method.
#        '''
#        if verbose:
#            iteration = tqdm(self)
#
#        offsetList = []
#        for oi in iteration:
#            offset = []
#            for p in tqdm(self._pol_list, leave=False):
#                offset.append(oi.cal_offset_opd(polarization=p, average=True, verbose=False, **kwargs))
#            offsetList.append(offset)
#
#        offsetList = np.array(offsetList).swapaxes(1, 2)  #[NDIT, XY, POLARIZATION]
#
#        return offsetList
#
#