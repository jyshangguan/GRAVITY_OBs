from typing import List, Union
import logging
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
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
    

    def get_wavelength(self, fiber='SC', polarization=None, units='micron'):
        '''
        Get the wavelength of the OIFITS HDU.
        '''
        assert units in ['micron', 'm'], 'units must be micron or m.'

        extver = self.get_extver(fiber, polarization)
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

        self._wave_sc = self.get_wavelength(units='micron')

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
            v_opd = np.exp(2j * np.pi * np.array(opdzp)[:, np.newaxis] / self._wave_sc[np.newaxis, :])
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

        visdata = visdata * np.exp(1j * (phaseref + 2*np.pi / self._wave_sc * (opdSep - opdDisp - opdMetCorr) * 1e6))

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
                if dit == 0:
                    label = self._baseline[bsl]
                else:
                    label = None

                ax.plot(ruv[dit, bsl, :], np.angle(visdata[dit, bsl, :], deg=True), 
                        color=f'C{bsl}', label=label)
        
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
                if dit == 0:
                    label = self._baseline[bsl]
                else:
                    label = None

                ax.plot(ruv[dit, bsl, :], np.absolute(visdata[dit, bsl, :]), 
                        color=f'C{bsl}', label=label)
        
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

        # Loading data
        for fiber in ['SC', 'FT']:

            # Visibility data
            for p in self._pol_list:
                # Get the extver
                extver = self.get_extver(fiber=fiber, polarization=p)

                visdata = self.get_visdata(fiber=fiber, polarization=p, 
                                           normalize=normalize)
                setattr(self, f'_visdata_{extver}', visdata)

                visamp = self.get_vis('VISAMP', fiber=fiber, polarization=p)
                setattr(self, f'_visamp_{extver}', visamp)

                visphi = self.get_vis('VISPHI', fiber=fiber, polarization=p)
                setattr(self, f'_visphi_{extver}', visphi)


    @lazyproperty
    def _uvcoord_m(self):
        '''
        The uv coordinate in meter. The dimensions are: [NDIT, NBASELINE, 1]
        '''
        uvc = self.get_uvcoord_vis(fiber='SC', polarization=None, units='m')
        return np.array(uvc)


    @lazyproperty
    def _uvcoord_Mlambda_sc(self):
        '''
        The uv coordinate in meter. The dimensions are: [NDIT, NBASELINE, 1]
        '''
        uvc = self.get_uvcoord_vis(fiber='SC', polarization=None, units='Mlambda')
        return np.array(uvc)

    
    @lazyproperty
    def _uvcoord_Mlambda_ft(self):
        '''
        The uv coordinate in meter. The dimensions are: [NDIT, NBASELINE, 1]
        '''
        uvc = self.get_uvcoord_vis(fiber='FT', polarization=None, units='Mlambda')
        return np.array(uvc)


    @lazyproperty
    def _wave_ft(self):
        '''
        Get the wavelength of the SC channel.
        '''
        return self.get_wavelength(fiber='FT', polarization=None, units='micron')


    @lazyproperty
    def _wave_sc(self):
        '''
        Get the wavelength of the SC channel.
        '''
        return self.get_wavelength(fiber='SC', polarization=None, units='micron')


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
            extver = self.get_extver(fiber='SC', polarization=p)
            u, v = getattr(self, f'_uvcoord_Mlambda_{extver}')
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
        wave = self._wave_sc

        if met_jump is not None:
            corr_tel = np.array(met_jump)[:, np.newaxis] * 2*np.pi * (1 - lambda_met / wave)[np.newaxis, :]
            opdDisp_corr = np.dot(t2b_matrix, corr_tel)
            visdata *= np.exp(1j * opdDisp_corr)
            setattr(self, f'_visphi_{extver}', np.angle(visdata, deg=True))

        if opdzp is not None:
            v_opd = np.exp(2j * np.pi * np.array(opdzp)[:, np.newaxis] / wave[np.newaxis, :])
            visdata *= np.conj(v_opd)[np.newaxis, :, :]
            setattr(self, f'_visphi_{extver}', np.angle(visdata, deg=True))


    def diff_visphi(self, 
                    oi : 'AstroFits', 
                    polarization : int = None,
                    average : bool = False, 
                    plot : bool = False, 
                    ax : plt.axis = None,
                    plain : bool = False):
        '''
        Calculate the difference of the VISPHI between the VISDATA of 
        the current and input AstroFits file.
        '''
        extver = self.get_extver(fiber="SC", polarization=polarization)
        vd1 = getattr(self, f'_visdata_{extver}')
        vd2 = getattr(oi, f'_visdata_{extver}')

        if average:
            vd1 = np.ma.mean(vd1, axis=0, keepdims=True)
            vd2 = np.ma.mean(vd2, axis=0, keepdims=True)

        visphi = np.ma.angle(vd1 * np.conj(vd2))

        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(7, 7))

            ruv = self.get_uvdist(fiber='SC', polarization=polarization, 
                              units='Mlambda')
            for dit in range(visphi.shape[0]):
                for bsl in range(visphi.shape[1]):
                    if dit == 0:
                        label = self._baseline[bsl]
                    else:
                        label = None
                    ax.plot(ruv[dit, bsl, :], np.rad2deg(visphi[dit, bsl, :]), color=f'C{bsl}', label=label)
        
            if not plain:
                ax.set_ylim([-180, 180])
                ax.set_xlabel('uv distance (Mlambda)', fontsize=18)
                ax.set_ylabel(r'VISPHI ($^\circ$)', fontsize=18)
                ax.legend(loc='upper left', fontsize=14, handlelength=1,
                          bbox_to_anchor=(1, 1))
                ax.minorticks_on()
        return visphi


    def flag_visdata(
            self, 
            fiber : str = None,
            polarization : int = None, 
            dit : Union[int, List[int]] = None, 
            baseline : Union[int, List[int]] = None):
        '''
        Flag the visibility data.
        '''
        if polarization is None:
            pols = self._pol_list
        else:
            pols = [polarization]

        if fiber is None:
            fibers = ['SC', 'FT']
        else:
            fibers = [fiber]

        for p in pols:
            for f in fibers:
                extver = self.get_extver(fiber=f, polarization=p)
                visdata = getattr(self, f'_visdata_{extver}')
                visamp = getattr(self, f'_visamp_{extver}')
                visphi = getattr(self, f'_visphi_{extver}')

                if dit is None:
                    dits = range(visdata.shape[0])
                elif isinstance(dit, int):
                    dits = [dit]
                elif isinstance(dit, list):
                    dits = dit
                else:
                    raise ValueError('dit must be an integer or a list of integers.')
                
                if baseline is None:
                    baselines = range(visdata.shape[1])
                elif isinstance(baseline, int):
                    baselines = [baseline]
                elif isinstance(baseline, list):
                    baselines = baseline
                else:
                    raise ValueError('baseline must be an integer or a list of integers.')

                for d in dits:
                    for b in baselines:
                        visdata[d, b, :].mask = True
                        visamp[d, b, :].mask = True
                        visphi[d, b, :].mask = True


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
            The units of the uv coordinates. Either Mlambda, or m.
            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.
        
        Returns
        -------
        ucoord, vcoord : arrays
            The uv coordinate of the baselines, (NDIT, NBASELINE, NCHANNEL).
        '''
        assert units in ['Mlambda', 'm'], 'units must be Mlambda, per mass, or m.'

        ucoord = self.get_vis('UCOORD', fiber=fiber, polarization=polarization)
        vcoord = self.get_vis('VCOORD', fiber=fiber, polarization=polarization)
        
        if units == 'Mlambda':
            wave = self.get_wavelength(fiber=fiber, polarization=polarization, 
                                       units='micron')
            ucoord = ucoord / wave[np.newaxis, np.newaxis, :]
            vcoord = vcoord / wave[np.newaxis, np.newaxis, :]

        return ucoord, vcoord


    def get_uvdist(
            self, 
            fiber : str = 'SC',
            polarization : int = None,
            units : str = 'Mlambda'):
        '''
        Get the uv distance of the baselines.

        Parameters
        ----------
        units : str, optional
            The units of the uv distance. Either Mlambda, or m.
            Mlambda is million lambda is per milliarcsecond, and m is meter.
        
        Returns
        -------
        uvdist : array
            The uv distance of the baselines, (NDIT, NBASELINE, NCHANNEL).
        '''
        assert units in ['Mlambda', 'm'], 'units must be Mlambda or m.'
        assert fiber in ['SC', 'FT'], 'fiber must be SC or FT.'

        if units == 'm':
            ucoord, vcoord = self._uvcoord_m
        else:
            if fiber == 'SC':
                ucoord, vcoord = self._uvcoord_Mlambda_sc
            else:
                ucoord, vcoord = self._uvcoord_Mlambda_ft

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
    def plot_visphi(
            self, 
            fiber : str = 'SC', 
            polarization : int = None, 
            use_visdata : bool = False, 
            ax : plt.axis = None, 
            plain : bool = False):
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

        ruv = self.get_uvdist(fiber=fiber, polarization=polarization, 
                              units='Mlambda')

        for dit in range(visphi.shape[0]):
            for bsl in range(visphi.shape[1]):
                if dit == 0:
                    label = self._baseline[bsl]
                else:
                    label = None
                ax.plot(ruv[dit, bsl, :], visphi[dit, bsl, :], color=f'C{bsl}', label=label)
        
        if not plain:
            ax.set_ylim([-180, 180])
            ax.set_xlabel('uv distance (Mlambda)', fontsize=18)
            ax.set_ylabel(r'VISPHI ($^\circ$)', fontsize=18)
            ax.legend(loc='upper left', fontsize=14, handlelength=1,
                      bbox_to_anchor=(1, 1))
            ax.minorticks_on()
        return ax
    

    # Revised
    def plot_visamp(
            self, 
            fiber : str = 'SC', 
            polarization : int = None, 
            use_visdata : bool = False, 
            ax : plt.axis = None, 
            plain : bool = False):
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

        ruv = self.get_uvdist(fiber=fiber, polarization=polarization, 
                              units='Mlambda')
        
        for dit in range(visamp.shape[0]):
            for bsl in range(visamp.shape[1]):
                if dit == 0:
                    label = self._baseline[bsl]
                else:
                    label = None
                ax.plot(ruv[dit, bsl, :], visamp[dit, bsl, :], color=f'C{bsl}', label=label)
        
        if not plain:
            ax.set_ylim([0, None])
            ax.set_xlabel('uv distance (Mlambda)', fontsize=18)
            ax.set_ylabel(r'VISAMP', fontsize=18)
            ax.legend(loc='upper left', fontsize=14, handlelength=1,
                      bbox_to_anchor=(1, 1))
            ax.minorticks_on()
        return ax



class GraviList(object):
    '''
    A list of GRAVITY OIFITS object.
    '''
    def __init__(self, name='GraviList'):

        self._name = name
        self._logger = None


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


    def set_logger(self, log_name=None, verbose=True):
        '''
        Set the logger.
        '''
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
        super().__init__(name='AstroList')
        self.set_logger(log_name=log_name, verbose=verbose)
        
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
        wave = oi._wave_sc
        baseline_name = oi._baseline

        # Prepare deriving the metrology zeropoint; [NPOL, NBASELINE]
        self._opdzp, _ = compute_gdelay(visdata, wave, max_width=max_width, 
                                        closure=closure, closure_lim=closure_lim, 
                                        verbose=verbose, logger=self._logger)
        self._fczp = np.array([self._opdzp[:, 2], 
                               self._opdzp[:, 4], 
                               self._opdzp[:, 5], 
                               np.zeros(npol)])

        if plot:
            if axs is None:
                fig, axs = plt.subplots(6, npol, figsize=(8*npol, 12), sharex=True, sharey=True)
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
                                                ax=ax, plain=True))

            if ax is not None:
                ax.minorticks_on()
                ax.set_ylim([-180, 180])
                ax.legend(loc='best', ncols=3, handlelength=1, columnspacing=1, fontsize=14)
            
            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)

            offsetList.append(offset)

        offsetList = np.array(offsetList).swapaxes(1, 2)  #[NDIT, XY, POLARIZATION]

        return offsetList


    def get_pairs(self):
        '''
        Get the pair index for the unswap and swap data.
        '''
        return np.array(np.meshgrid(self._index_unswap, self._index_swap)).T.reshape(-1, 2)


    def plot_data(
            self, 
            report_name : str = None):
        '''
        Plot all data for a visual check.
        '''

        if report_name is not None:
            pdf = PdfPages(report_name)
        else:
            pdf = None

        # Plot the visibility of the SC and FT data
        if self._logger is not None:
            self._logger.info('Plotting the visibility data')

        for loop, oi in enumerate(self):
            fig, axs = plt.subplots(oi._npol, 2, figsize=(28, 7*oi._npol))
            axs = np.array(axs).reshape(-1, 2)
            fig.suptitle(f'Original data [{loop}]: {oi._arctime}', fontsize=18)

            for i, p in enumerate(self._pol_list):
                oi.plot_visamp(polarization=p, ax=axs[i, 0])
                oi.plot_visphi(polarization=p, ax=axs[i, 1])

                axs[i, 0].text(0.05, 0.05, f'P{p}, SC', fontsize=16,
                               transform=axs[i, 0].transAxes, va='bottom', ha='left',
                               bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))
                axs[i, 0].legend().set_visible(False)

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)

        # Plot the visphi of each swap pair
        pair_index = self.get_pairs()
            
        for loop, (i, j) in enumerate(pair_index):
            oi1 = self[int(i)]
            oi2 = self[int(j)]

            fig, axs = plt.subplots(1, oi1._npol, figsize=(7*oi1._npol, 7), sharex=True, sharey=True)
            fig.subplots_adjust(hspace=0.02, wspace=0.02)
            fig.suptitle(f'Swap pair {loop} [{i}-{j}]: {oi1._arctime} and {oi2._arctime}', fontsize=18) 
            axs = np.atleast_1d(axs)
            axs[0].set_ylabel(r'VISPHI ($^\circ$)', fontsize=18)

            ruv = oi1.get_uvdist(units='Mlambda')
            for i, p in enumerate(self._pol_list):
                wave = oi1._wave_sc
                visphi = oi1.diff_visphi(oi2, polarization=p, average=True)
                gd, _ = compute_gdelay(np.exp(1j*visphi), wave, logger=self._logger)

                ax = axs[i]
                for bsl in range(visphi.shape[1]):
                    l1 = oi1._baseline[bsl]
                    ax.plot(ruv[0, bsl, :], np.rad2deg(visphi[0, bsl, :]), ls='-', color=f'C{bsl}', label=l1)
                    
                for bsl in range(visphi.shape[1]):
                    if bsl == 0:
                        l2 = 'Gdelay'
                        l3 = r'$\vec{s} \cdot \vec{B}$'
                    else:
                        l2 = None
                        l3 = None
                    ax.plot(ruv[0, bsl, :], np.angle(np.exp(2j * np.pi * gd[0, bsl] / wave), deg=True), 
                            ls='-', color=f'gray', label=l2)

                txt = 'GDelay: ' + ', '.join([f'{v:.2f}' for v in gd.mean(axis=0)])
                ax.text(0.05, 0.05, txt, fontsize=14, 
                        transform=ax.transAxes, va='bottom', ha='left',
                        bbox=dict(facecolor='w', edgecolor='w', alpha=0.8))
                ax.set_xlabel(r'$uv$ distance (M$\lambda$)', fontsize=18)
                ax.text(0.05, 0.95, f'P{p}', fontsize=16, transform=ax.transAxes, va='top', ha='left',
                        bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))
            ax.legend(loc='upper left', fontsize=14, handlelength=1, 
                      bbox_to_anchor=(1, 1))
            ax.set_ylim([-180, 180])

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)
            else:
                plt.show()
        
        if pdf is not None:
            pdf.close()

        self._logger.info('All data plotted!')


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
        super().__init__(name='SciVisList')
        self.set_logger(log_name=log_name, verbose=verbose)
        
        self._datalist = []
        self._index_unswap = []
        self._index_swap = []
        for i, f in enumerate(files):
            oi = SciVisFits(f, ignore_flag=ignore_flag, normalize=normalize)
            self._datalist.append(oi)

            if oi._swap:
                self._index_swap.append(i)
            else:
                self._index_unswap.append(i)
            
            self._logger.info(f'Processing [{i}] {oi._arctime}, swap: {oi._swap}')

        self._pol_list = self._datalist[0]._pol_list
        self._sobj_x = self._datalist[0]._sobj_x
        self._sobj_y = self._datalist[0]._sobj_y


    def astrometry_swap(self, plot=True, report_name=None):
        '''
        The main function to measure astrometry and metrology zero point from 
        the swap data.
        '''
        assert len(self._index_unswap) > 0, 'There is no unswap data in the list!'
        assert len(self._index_swap) > 0, 'There is no swap data in the list!'

        if report_name is not None:
            pdf = PdfPages(report_name)
        else:
            pdf = None


        # Search for astrometry using the phase method
        self._logger.info('Grid search for astrometry solution')
        res = self.grid_search_phase(plot=False)
        txt = f"({res['ra_best_zoom']:.2f}, {res['dec_best_zoom']:.2f})"
        self._logger.info(f'First-run grid search: {txt}')


        # Measure metrology zero point
        self._logger.info('Measure metrology zero point')
        self.compute_metzp(res['ra_best'], res['dec_best'], plot=plot, pdf=pdf)


        # Correct the metrology zero point
        self._logger.info('Correct the metrology zeropoint')
        self.correct_metzp()


        # Plot the phase to evaluate the data correction
        self._logger.info('Plotting the metrology zeropoint corrected data')
        for loop, oi in enumerate(self):
            fig, axs = plt.subplots(1, oi._npol, figsize=(7*oi._npol, 7))
            axs = np.atleast_1d(axs)
            fig.suptitle(f'METZP corrected [{loop}]: {oi._arcfile}', fontsize=18)

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
        self._ra_best = res['ra_best_zoom']
        self._dec_best = res['dec_best_zoom']
        self._sobj_x_fit = self._sobj_x + res['ra_best_zoom']
        self._sobj_y_fit = self._sobj_y + res['dec_best_zoom']
        txt = f"({res['ra_best_zoom']:.2f}, {res['dec_best_zoom']:.2f})"
        self._logger.info(f'Final grid search: {txt}')
        self._logger.info(f'Measured FT-SC vector: ({self._sobj_x_fit:.2f}, {self._sobj_y_fit:.2f})')

        
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

        self._logger.info("Astrometry swap finished!")


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
                u, v = oi._uvcoord_Mlambda_sc
                phase = phase_model(ra, dec, u, v)
                model = np.exp(1j * phase)
                visdata = getattr(oi, f'_visdata_1{p}')
                gamma1.append(np.conj(model) * visdata)
                gooddata1.append(visdata.mask == False)
        
        gamma2 = []
        gooddata2 = []
        for oi in self[self._index_swap]:
            for p in self._pol_list:
                u, v = oi._uvcoord_Mlambda_sc
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
        mask1 = []
        mask2 = []
        for p in self._pol_list:
            # Unswapped data
            vd = []
            mk = []
            for oi in self[self._index_unswap]:
                u, v = oi._uvcoord_Mlambda_sc
                phase = phase_model(-ra, -dec, u, v)
                vis1 = getattr(oi, f'_visdata_1{p}')
                vd.append(vis1 * np.exp(1j * phase))
                mk.append(vis1.mask)
            visdata1.append(vd)
            mask1.append(mk)
    
            vd = []
            mk = []
            for oi in self[self._index_swap]:
                u, v = oi._uvcoord_Mlambda_sc
                phase = phase_model(ra, dec, u, v)
                vis1 = getattr(oi, f'_visdata_1{p}')
                vd.append(vis1 * np.exp(1j * phase))
                mk.append(vis1.mask)
            visdata2.append(vd)
            mask2.append(mk)

        visdata = 0.5 * (np.ma.mean(visdata1, axis=(1, 2)) + np.ma.mean(visdata2, axis=(1, 2)))
        mask = np.logical_or(np.logical_and.reduce(mask1, axis=(1, 2)), 
                             np.logical_and.reduce(mask2, axis=(1, 2)))
        visdata = np.ma.array(visdata, mask=mask)
        phi0 = np.angle(visdata, deg=True)
    
        npol = len(self._pol_list)
        wave = oi._wave_sc
        baseline_name = oi._baseline

        # Prepare deriving the metrology zeropoint; [NPOL, NBASELINE]
        self._opdzp, _ = compute_gdelay(visdata, wave, max_width=max_width, 
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
                fig.subplots_adjust(hspace=0.02, wspace=0.15)
                axo = fig.add_subplot(111, frameon=False) # The out axis
                axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18, labelpad=20)
                axo.set_ylabel(r'$\phi_0$ ($^\circ$)', fontsize=18, labelpad=50)

            else:
                assert len(axs) == 6, 'The number of axes must be 6!'

            for i, p in enumerate(self._pol_list):
                for bsl, ax in enumerate(axs[:, i]):
                    ax.plot(wave, phi0[i, bsl, :], color=f'C{bsl}')
                    ax.plot(wave, np.angle(matrix_opd(self._opdzp[i, bsl], wave), deg=True), 
                            color='gray', alpha=0.5, label='OPD model')
                    ax.text(0.95, 0.9, f'{self._opdzp[i, bsl]:.2f} $\mu$m', fontsize=14, 
                            color='k', transform=ax.transAxes, va='top', ha='right', 
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

        Parameters
        ----------
        index : int
            The index of the data in the list.
        met_jump : list
            The metrology phase jump in number of fringes.
        '''
        for p in self._pol_list:
            self._datalist[index].correct_visdata(polarization=p, met_jump=met_jump)


    def correct_met_jump_all(
            self, 
            met_jump_dict : dict):
        '''
        Correct all the metrology phase jumps from the provided dictionary.

        Parameters
        ----------
        met_jump_dict : dict
            A dictionary containing the metrology phase jump.
            The key is the index of the data in the list and the value is 
            the metrology phase jump.
        '''
        for i, met_jump in met_jump_dict.items():
            if self._logger is not None:
                self._logger.info(f'Correct the metrology phase jump of file {i}: {met_jump}')

            self.correct_met_jump(i, met_jump)


    def flag_visdata(
            self, 
            index : Union[int, List[int]] = None, 
            fiber : str = None,
            polarization : int = None, 
            dit : Union[int, List[int]] = None, 
            baseline : Union[int, List[int]] = None):
        '''
        Flag the visibility data.
        '''
        if index is None:
            for oi in self:
                oi.flag_visdata(fiber=fiber, polarization=polarization, 
                                dit=dit, baseline=baseline)
        elif isinstance(index, int):
            self[index].flag_visdata(fiber=fiber, polarization=polarization, 
                                     dit=dit, baseline=baseline)
        elif isinstance(index, list):
            for i in index:
                self[i].flag_visdata(fiber=fiber, polarization=polarization, 
                                     dit=dit, baseline=baseline)
        else:
            raise ValueError('The index must be an integer or a list of integers!')


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
        pair_index = self.get_pairs()

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
                                                ax=ax, plain=True))
                
                ra, dec = offset[-1]
                self._logger.info(f'GD astrometry [{i}-{j}]: {ra:.2f}, {dec:.2f}')

                if ax is not None:
                    ax.text(0.05, 0.05, f'RA: {ra:.2f}\nDec: {dec:.2f}', fontsize=14, 
                            transform=ax.transAxes, va='bottom', ha='left',
                            bbox=dict(facecolor='w', edgecolor='w', alpha=0.8))

            if ax is not None:
                ax.minorticks_on()
                ax.set_ylim([-180, 180])
                ax.legend(loc='best', ncols=3, handlelength=1, columnspacing=1, fontsize=14)
            
            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)

            offsetList.append(offset)

        offsetList = np.ma.array(offsetList).swapaxes(1, 2)  #[NDIT, XY, POLARIZATION]

        return offsetList


    def get_pairs(self):
        '''
        Get the pair index for the unswap and swap data.
        '''
        return np.array(np.meshgrid(self._index_unswap, self._index_swap)).T.reshape(-1, 2)


    def plot_data(
            self, 
            report_name : str = None):
        '''
        Plot all data for a visual check.
        '''

        if report_name is not None:
            pdf = PdfPages(report_name)
        else:
            pdf = None

        # Plot the visibility of the SC and FT data
        if self._logger is not None:
            self._logger.info('Plotting the visibility data')

        for loop, oi in enumerate(self):
            fig, axs = plt.subplots(oi._npol, 4, figsize=(28, 7*oi._npol))
            axs = np.array(axs).reshape(-1, 4)
            fig.suptitle(f'Original data [{loop}]: {oi._arctime}', fontsize=18)

            for i, p in enumerate(self._pol_list):
                oi.plot_visamp(fiber='SC', polarization=p, ax=axs[i, 0])
                oi.plot_visphi(fiber='SC', polarization=p, ax=axs[i, 1])
                oi.plot_visamp(fiber='FT', polarization=p, ax=axs[i, 2])
                oi.plot_visphi(fiber='FT', polarization=p, ax=axs[i, 3])

                axs[i, 0].text(0.05, 0.05, f'P{p}, SC', fontsize=16,
                               transform=axs[i, 0].transAxes, va='bottom', ha='left',
                               bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))
                axs[i, 2].text(0.05, 0.05, f'P{p}, FT', fontsize=16,
                               transform=axs[i, 2].transAxes, va='bottom', ha='left',
                               bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))

                for ax in axs[i, :3]:
                    ax.legend().set_visible(False)

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)

        # Plot the visphi of each swap pair
        pair_index = self.get_pairs()
            
        for loop, (i, j) in enumerate(pair_index):
            oi1 = self[int(i)]
            oi2 = self[int(j)]

            fig, axs = plt.subplots(1, oi1._npol, figsize=(7*oi1._npol, 7), sharex=True, sharey=True)
            fig.subplots_adjust(hspace=0.02, wspace=0.02)
            fig.suptitle(f'Swap pair {loop} [{i}-{j}]: {oi1._arctime} and {oi2._arctime}', fontsize=18) 
            axs = np.atleast_1d(axs)
            axs[0].set_ylabel(r'VISPHI ($^\circ$)', fontsize=18)

            ruv = oi1.get_uvdist(units='Mlambda')
            for i, p in enumerate(self._pol_list):
                wave = oi1._wave_sc
                visphi = oi1.diff_visphi(oi2, polarization=p)
                gd, _ = compute_gdelay(np.exp(1j*visphi), wave, logger=self._logger)

                ax = axs[i]
                for dit in range(visphi.shape[0]):
                    for bsl in range(visphi.shape[1]):
                        if dit == 0:
                            l1 = oi1._baseline[bsl]
                        else:
                            l1 = None
                        ax.plot(ruv[dit, bsl, :], np.rad2deg(visphi[dit, bsl, :]), ls='-', color=f'C{bsl}', label=l1)
                    
                    for bsl in range(visphi.shape[1]):
                        if (dit == 0) & (bsl == 0):
                            l2 = 'Gdelay'
                            l3 = r'$\vec{s} \cdot \vec{B}$'
                        else:
                            l2 = None
                            l3 = None
                        ax.plot(ruv[dit, bsl, :], np.angle(np.exp(2j * np.pi * gd[dit, bsl] / wave), deg=True), 
                                ls='-', color=f'gray', label=l2)

                txt = 'GDelay: ' + ', '.join([f'{v:.2f}' for v in gd.mean(axis=0)])
                ax.text(0.05, 0.05, txt, fontsize=14, 
                        transform=ax.transAxes, va='bottom', ha='left',
                        bbox=dict(facecolor='w', edgecolor='w', alpha=0.8))
                ax.set_xlabel(r'$uv$ distance (M$\lambda$)', fontsize=18)
                ax.text(0.05, 0.95, f'P{p}', fontsize=16, transform=ax.transAxes, va='top', ha='left',
                        bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))
            ax.legend(loc='upper left', fontsize=14, handlelength=1, 
                      bbox_to_anchor=(1, 1))
            ax.set_ylim([-180, 180])

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)
            else:
                plt.show()
        
        if pdf is not None:
            pdf.close()

        self._logger.info('All data plotted!')


    def run_swap_astrometry(self, met_jump_dict=None, report_name=None):

        if report_name is None:
            now = datetime.now()
            now_str = now.strftime("%Y-%m-%dT%H:%M:%S")
            report_name = f'astrometry_report_{now_str}'
        
        self.plot_data(report_name=f'{report_name}_plot.pdf')

        if met_jump_dict is not None:
            self.correct_met_jump_all(met_jump_dict)

        self.astrometry_swap(report_name=f'{report_name}_astrometry.pdf')

        self._logger.info("[run_swap_astrometry] pipeline completed!")

