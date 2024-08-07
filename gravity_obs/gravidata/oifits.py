import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.decorators import lazyproperty
from .gravi_utils import N_BASELINE, N_TRIANGLE, N_TELESCOPE
from .gravi_utils import baseline_names, telescope_names, triangle_names
from .gravi_utils import t2b_matrix, lambda_met
from .astro_utils import (phase_model, fit_opd_closure, matrix_opd, solve_offset,
                          grid_search_opd)


class SciVisFits(object):
    '''
    A class to read and plot GRAVITY SCIVIS data.
    '''
    def __init__(self, filename):
        '''
        Parameters
        ----------
        filename : str
            The name of the OIFITS file.
        '''
        self._filename = filename
        self._hdul = fits.open(filename)
        header = self._hdul[0].header
        self._header = header

        self._arcfile = header.get('ARCFILE', None)
        if self._arcfile is not None:
            self._arctime = self._arcfile.split('.')[1]
        else:
            self._arctime = None

        self._object = header.get('OBJECT', None)

        # Instrument mode
        self._pol = header.get('ESO INS POLA MODE', None)
        self._res = header.get('ESO INS SPEC RES', None)

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


    def get_t3phi(self, fiber='SC', polarization=None):
        '''
        Get the T3PHI and T3PHIERR of the OIFITS HDU.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        t3phi, t3phierr : masked arrays
            The T3PHI and T3PHIERR of the OIFITS HDU. The shape is (N_TRIANGLE, N_CHANNEL).
        '''
        extver = self.get_extver(fiber, polarization)
        flag = self._hdul['OI_T3', extver].data['FLAG']
        t3phi = np.ma.array(self._hdul['OI_T3', extver].data['T3PHI'], mask=flag)
        t3phierr = np.ma.array(self._hdul['OI_T3', extver].data['T3PHIERR'], mask=flag)
        t3phi = np.reshape(t3phi, (-1, N_TRIANGLE, t3phi.shape[1]))
        t3phierr = np.reshape(t3phierr, (-1, N_TRIANGLE, t3phierr.shape[1]))
        return t3phi, t3phierr
    
    
    def get_t3_uvcoord(self, fiber='SC', polarization=None, units='Mlambda'):
        '''
        Get the uv coordinates of the triangles. It returns two uv coordinates. 
        The third uv coordinate can be calculated as -u1-u2 and -v1-v2.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv coordinates. Either Mlambda, per mas, or m.
        
        Returns
        -------
        u1coord, v1coord, u2coord, v2coord : arrays
            The uv coordinate of the triangles. 
            If units=m, the shape is (N_TRIANGLE, ), otherwise (N_TRIANGLE, N_CHANNEL).
        '''
        assert units in ['Mlambda', 'per mas', 'm'], 'units must be Mlambda, per mas, or m.'

        extver = self.get_extver(fiber, polarization)
        u1coord = self._hdul['OI_T3', extver].data['U1COORD'].reshape(-1, N_TRIANGLE)
        v1coord = self._hdul['OI_T3', extver].data['V1COORD'].reshape(-1, N_TRIANGLE)
        u2coord = self._hdul['OI_T3', extver].data['U2COORD'].reshape(-1, N_TRIANGLE)
        v2coord = self._hdul['OI_T3', extver].data['V2COORD'].reshape(-1, N_TRIANGLE)
        
        if units != 'm':
            wave = self.get_wavelength(fiber, polarization, units='micron')
            u1coord = u1coord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
            v1coord = v1coord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
            u2coord = u2coord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
            v2coord = v2coord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
            
            if units == 'per mas':
                u1coord = u1coord * np.pi / 180. / 3600 * 1e3
                v1coord = v1coord * np.pi / 180. / 3600 * 1e3
                u2coord = u2coord * np.pi / 180. / 3600 * 1e3
                v2coord = v2coord * np.pi / 180. / 3600 * 1e3
        
        return u1coord, v1coord, u2coord, v2coord


    def get_uvdist(self, fiber='SC', polarization=None, units='Mlambda'):
        '''
        Get the uv distance of the baselines.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv coordinates. Either Mlambda, per mas, or m.

        Returns
        -------
        uvdist : array
            The uv distance of the baselines. If units=m, the shape is (N_BASELINE, ), 
            otherwise (N_BASELINE, N_CHANNEL).
        '''
        u, v = self.get_vis_uvcoord(fiber, polarization, units=units)
        uvdist = np.sqrt(u**2 + v**2)
        return uvdist
    

    def get_uvmax(self, fiber='SC', polarization=None, units='Mlambda'):
        '''
        Get the maximum uv distance of the triangles.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv coordinates. Either Mlambda, per mas, or m.

        Returns
        -------
        uvmax : array
            The maximum uv distance of the triangles. If units=m, the shape is (N_TRIANGLE, ), 
            otherwise (N_TRIANGLE, N_CHANNEL).
        '''
        u1, v1, u2, v2 = self.get_t3_uvcoord(fiber, polarization, units=units)
        u3 = -u1 - u2
        v3 = -v1 - v2
        uvmax = np.max([np.sqrt(u1**2 + v1**2), 
                        np.sqrt(u2**2 + v2**2), 
                        np.sqrt(u3**2 + v3**2)], axis=0)
        return uvmax


    def get_vis_uvcoord(self, fiber='SC', polarization=None, units='m'):
        '''
        Get the u and v coordinates of the baselines.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv coordinates. Either Mlambda, per mas, or m.
        
        Returns
        -------
        ucoord, vcoord : arrays
            The uv coordinate of the baselines. 
            If units=m, the shape is (N_BASELINE, ), otherwise (N_BASELINE, N_CHANNEL).
        '''
        assert units in ['Mlambda', 'per mas', 'm'], 'units must be Mlambda, per mass, or m.'

        extver = self.get_extver(fiber, polarization)
        ucoord = self._hdul['OI_VIS', extver].data['UCOORD'].reshape(-1, N_BASELINE)
        vcoord = self._hdul['OI_VIS', extver].data['VCOORD'].reshape(-1, N_BASELINE)
        
        if units != 'm':
            wave = self.get_wavelength(fiber, polarization, units='micron')
            ucoord = ucoord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
            vcoord = vcoord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
            
            if units == 'per mas':
                ucoord = ucoord * np.pi / 180. / 3600 * 1e3
                vcoord = vcoord * np.pi / 180. / 3600 * 1e3
        
        return ucoord, vcoord
    

    def get_visamp(self, fiber='SC', polarization=None, fromdata=False):
        '''
        Get the VISAMP and VISAMPERR of the OIFITS HDU.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        visamp, visamperr : masked arrays
            The VISAMP and VISAMPERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
        '''
        extver = self.get_extver(fiber, polarization)

        if hasattr(self, f'_visamp_{extver}') & (not fromdata):
            return self.__getattribute__(f'_visamp_{extver}'), self.__getattribute__(f'_visamperr_{extver}')

        flag = self._hdul['OI_VIS', extver].data['FLAG']
        visamp = np.ma.array(self._hdul['OI_VIS', extver].data['VISAMP'], mask=flag)
        visamperr = np.ma.array(self._hdul['OI_VIS', extver].data['VISAMPERR'], mask=flag)
        visamp = np.reshape(visamp, (-1, N_BASELINE, visamp.shape[1]))
        visamperr = np.reshape(visamperr, (-1, N_BASELINE, visamperr.shape[1]))

        self.__setattr__(f'_visamp_{extver}', visamp)
        self.__setattr__(f'_visamperr_{extver}', visamperr)

        return visamp, visamperr
    

    def get_visdata(self, fiber='SC', polarization=None, fromdata=False, per_exp=False):
        '''
        Get the VISDATA and VISERR of the OIFITS HDU.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        visdata, viserr : masked arrays
            The VISDATA and VISERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
        '''
        extver = self.get_extver(fiber, polarization)

        if hasattr(self, f'_visdata_{extver}') & (not fromdata):
            return self.__getattribute__(f'_visdata_{extver}'), self.__getattribute__(f'_viserr_{extver}')

        flag = self._hdul['OI_VIS', extver].data['FLAG']
        visdata = np.ma.array(self._hdul['OI_VIS', extver].data['VISDATA'], mask=flag)
        viserr = np.ma.array(self._hdul['OI_VIS', extver].data['VISERR'], mask=flag)
        visdata = np.reshape(visdata, (-1, N_BASELINE, visdata.shape[1]))
        viserr = np.reshape(viserr, (-1, N_BASELINE, viserr.shape[1]))

        if per_exp:
            visdata /= self._dit * self._ndit
            viserr /= self._dit * self._ndit

        self.__setattr__(f'_visdata_{extver}', visdata)
        self.__setattr__(f'_viserr_{extver}', viserr)

        return visdata, viserr
    

    def get_visphi(self, fiber='SC', polarization=None, fromdata=False):
        '''
        Get the VISPHI and VISPHIERR of the OIFITS HDU.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        visphi, visphierr : masked arrays
            The VISPHI and VISPHIERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
        '''
        extver = self.get_extver(fiber, polarization)

        if hasattr(self, f'_visphi_{extver}') & (not fromdata):
            return self.__getattribute__(f'_visphi_{extver}'), self.__getattribute__(f'_visphierr_{extver}')

        flag = self._hdul['OI_VIS', extver].data['FLAG']
        visphi = np.ma.array(self._hdul['OI_VIS', extver].data['VISPHI'], mask=flag)
        visphierr = np.ma.array(self._hdul['OI_VIS', extver].data['VISPHIERR'], mask=flag)
        visphi = np.reshape(visphi, (-1, N_BASELINE, visphi.shape[1]))
        visphierr = np.reshape(visphierr, (-1, N_BASELINE, visphierr.shape[1]))

        self.__setattr__(f'_visphi_{extver}', visphi)
        self.__setattr__(f'_visphierr_{extver}', visphierr)

        return visphi, visphierr


    def get_vis2data(self, fiber='SC', polarization=None, fromdata=False):
        '''
        Get the VIS2DATA and VIS2ERR of the OIFITS HDU.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for
            COMBINED and SPLIT, respectively.

        Returns
        -------
        vis2data, vis2err : masked arrays
            The VIS2DATA and VIS2ERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
        '''
        extver = self.get_extver(fiber, polarization)

        if hasattr(self, f'_vis2data_{extver}') & hasattr(self, f'_vis2err_{extver}') & (not fromdata):
            return self.__getattribute__(f'_vis2data_{extver}'), self.__getattribute__(f'_vis2err_{extver}')

        flag = self._hdul['OI_VIS', extver].data['FLAG']
        vis2data = np.ma.array(self._hdul['OI_VIS2', extver].data['VIS2DATA'], mask=flag)
        vis2err = np.ma.array(self._hdul['OI_VIS2', extver].data['VIS2ERR'], mask=flag)
        vis2data = np.reshape(vis2data, (-1, N_BASELINE, vis2data.shape[1]))
        vis2err = np.reshape(vis2err, (-1, N_BASELINE, vis2err.shape[1]))

        self.__setattr__(f'_vis2data_{extver}', vis2data)
        self.__setattr__(f'_vis2err_{extver}', vis2err)

        return vis2data, vis2err


    def get_wavelength(self, fiber='SC', polarization=None, units='micron'):
        '''
        Get the wavelength of the OIFITS HDU.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the wavelength. Either micron or m.
        
        Returns
        -------
        wave : array
            The wavelength of the OIFITS HDU. If units=m, the shape is (N_CHANNEL, ), 
            otherwise (N_CHANNEL, ).
        '''
        assert units in ['micron', 'm'], 'units must be um or m.'

        extver = self.get_extver(fiber, polarization)
        wave = self._hdul['OI_WAVELENGTH', extver].data['EFF_WAVE']

        if units == 'micron':
            wave = wave * 1e6

        return wave


    def plot_t3phi_ruv(self, fiber='SC', polarization=None, units='Mlambda', 
                       show_average=False, ax=None, plain=False, legend_kwargs=None, 
                       **kwargs):
        '''
        Plot the T3PHI as a function of the uv distance.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv distance. Either Mlambda or per mas.
        show_average : bool, optional
            If True, the average T3PHI will be shown. Default is False.
        ax : matplotlib axis, optional
            The axis to plot the T3PHI as a function of the uv distance. If None, 
            a new figure and axis will be created.
        plain : bool, optional
            If True, the axis labels and legend will not be plotted.
        legend_kwargs : dict, optional
            The keyword arguments of the legend.
        kwargs : dict, optional
            The keyword arguments of the errorbar plot.

        Returns
        -------
        ax : matplotlib axis
            The axis of the T3PHI as a function of the uv distance.
        '''
        assert units in ['Mlambda', 'per mas'], 'The units must be Mlambda or per mas.'

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        uvdist = self.get_uvmax(fiber, polarization, units=units)
        t3phi, t3phierr = self.get_t3phi(fiber, polarization)

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5
        
        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'
        
        if 'ecolor' not in kwargs:
            kwargs['ecolor'] = 'gray'

        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
            kwargs['ls'] = '-'

        for trl in range(N_TRIANGLE):
            kwargs_use = kwargs.copy()

            if 'color' not in kwargs:
                kwargs_use['color'] = f'C{trl}'

            for dit in range(t3phi.shape[0]):

                if 'label' not in kwargs:
                    if dit == 0:
                        kwargs_use['label'] = f'{self._triangle[trl]}'
                    else:
                        kwargs_use['label'] = None

                ax.errorbar(
                    uvdist[dit, trl], t3phi[dit, trl, :], yerr=t3phierr[dit, trl, :], 
                    **kwargs_use)

        if show_average:
            t3ave = np.nanmean(t3phi[0, :, :], axis=-1)
            t3std = np.nanstd(t3phi[0, :, :], axis=-1)
            t3phi_text = '\n'.join([fr'{self._triangle[ii]}: {t3ave[ii]:.1f} +/- {t3std[ii]:.1f}$^\circ$' 
                                    for ii in range(N_TRIANGLE)])
            ax.text(0.45, 0.95, t3phi_text, fontsize=14, transform=ax.transAxes, ha='left', va='top', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        if not plain:
            if units == 'Mlambda':
                ax.set_xlabel(r'$uv$ distance (M$\lambda$)', fontsize=24)
            else:
                ax.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24)
            ax.set_ylabel(r'T3PHI ($^\circ$)', fontsize=24)
            ax.minorticks_on()

            if legend_kwargs is None:
                legend_kwargs = {}

            if 'fontsize' not in legend_kwargs:
                legend_kwargs['fontsize'] = 14

            if 'loc' not in legend_kwargs:
                legend_kwargs['loc'] = 'upper left'

            if 'ncols' not in legend_kwargs:
                legend_kwargs['ncols'] = 2

            ax.legend(**legend_kwargs)
        return ax
    

    def plot_visamp_uvdist(self, fiber='SC', polarization=None, units='Mlambda',
                           show_average=False, ax=None, plain=False, legend_kwargs=None, 
                           **kwargs):
        '''
        Plot the VISAMP as a function of the uv distance.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv distance. Either Mlambda or per mas.
        show_average : bool, optional
            If True, the average VISAMP will be shown. Default is False.
        ax : matplotlib axis, optional
            The axis to plot the VISAMP as a function of the uv distance. If None, 
            a new figure and axis will be created.
        plain : bool, optional
            If True, the axis labels and legend will not be plotted.
        legend_kwargs : dict, optional
            The keyword arguments of the legend.
        kwargs : dict, optional
            The keyword arguments of the errorbar plot.
                
        Returns
        -------
        ax : matplotlib axis
            The axis of the VISAMP as a function of the uv distance.
        '''
        assert units in ['Mlambda', 'per mas'], 'The units must be Mlambda or per mas.'

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        uvdist = self.get_uvdist(fiber, polarization, units=units)
        visamp, visamperr = self.get_visamp(fiber, polarization)

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'

        if 'ecolor' not in kwargs:
            kwargs['ecolor'] = 'gray'

        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
            kwargs['ls'] = '-'
            
        for bsl in range(N_BASELINE):
            kwargs_use = kwargs.copy()

            if 'color' not in kwargs:
                kwargs_use['color'] = f'C{bsl}'

            for dit in range(visamp.shape[0]):

                if 'label' not in kwargs:
                    if dit == 0:
                        kwargs_use['label'] = f'{self._baseline[bsl]}'
                    else:
                        kwargs_use['label'] = None

                ax.errorbar(
                    uvdist[dit, bsl], visamp[dit, bsl, :], yerr=visamperr[dit, bsl, :], 
                    **kwargs_use)
                
        if show_average:
            visave = np.nanmean(visamp[0, :, :], axis=-1)
            visstd = np.nanstd(visamp[0, :, :], axis=-1)
            visamp_text = '\n'.join([fr'{self._baseline[ii]}: {visave[ii]:.1f} +/- {visstd[ii]:.1f}' 
                                    for ii in range(N_BASELINE)])
            ax.text(0.45, 0.95, visamp_text, fontsize=14, transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            
        if not plain:
            if units == 'Mlambda':
                ax.set_xlabel(r'$uv$ distance (M$\lambda$)', fontsize=24)
            else:
                ax.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24)
            ax.set_ylabel('VISAMP', fontsize=24)
            ax.minorticks_on()

            if legend_kwargs is None:
                legend_kwargs = {}

            if 'fontsize' not in legend_kwargs:
                legend_kwargs['fontsize'] = 14

            if 'loc' not in legend_kwargs:
                legend_kwargs['loc'] = 'upper left'

            if 'ncols' not in legend_kwargs:
                legend_kwargs['ncols'] = 2

            ax.legend(**legend_kwargs)

        return ax
    

    def plot_visdata_wavelength(self, fiber='SC', polarization=None, axs=None, 
                                plain=False, legend_kwargs=None, **kwargs):
        '''
        Plot the VISDATA as a function of the wavelength.

        Parameters
        ----------
        '''
        if axs is None:
            fig, axs = plt.subplots(N_BASELINE, 1, figsize=(12, 6), sharex=True, 
                                    sharey=True)

        wave = self.get_wavelength(fiber, polarization)
        visdata, viserr = self.get_visdata(fiber, polarization)

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'

        if 'ecolor' not in kwargs:
            kwargs['ecolor'] = 'gray'

        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
            kwargs['ls'] = '-'
            
        for bsl in range(N_BASELINE):
            kwargs_use = kwargs.copy()

            ax = axs[bsl]
            for dit in range(visdata.shape[0]):

                if 'label' not in kwargs:
                    if dit == 0:
                        kwargs_use['label'] = f'{self._baseline[bsl]}'
                    else:
                        kwargs_use['label'] = None

                if 'color' not in kwargs:
                    kwargs_use['color'] = f'C0'

                ax.errorbar(
                    wave, visdata[dit, bsl, :].real, yerr=viserr[dit, bsl, :].real, **kwargs_use)

                if 'color' not in kwargs:
                    kwargs_use['color'] = f'C3'

                ax.errorbar(
                    wave, visdata[dit, bsl, :].imag, yerr=viserr[dit, bsl, :].imag, **kwargs_use)

        if not plain:
            fig.subplots_adjust(hspace=0, wspace=0)
            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=24, labelpad=25)
            axo.set_title('VISDATA', fontsize=24)
            ax.minorticks_on()

            if legend_kwargs is None:
                legend_kwargs = {}

            if 'fontsize' not in legend_kwargs:
                legend_kwargs['fontsize'] = 14

            if 'loc' not in legend_kwargs:
                legend_kwargs['loc'] = 'lower right'
            
            if 'bbox_to_anchor' not in legend_kwargs:
                legend_kwargs['bbox_to_anchor'] = (1, 1)

            if 'ncols' not in legend_kwargs:
                legend_kwargs['ncols'] = 2

            axs[0].legend(**legend_kwargs)
        
        return axs


    def plot_visphi_uvdist(self, fiber='SC', polarization=None, units='Mlambda',
                           show_average=False, ax=None, plain=False, legend_kwargs=None, 
                           **kwargs):
        '''
        Plot the VISPHI as a function of the uv distance.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv distance. Either Mlambda or per mas.
        show_average : bool, optional
            If True, the average VISPHI will be shown. Default is False.
        ax : matplotlib axis, optional
            The axis to plot the VISPHI as a function of the uv distance. If None, 
            a new figure and axis will be created.
        plain : bool, optional
            If True, the axis labels and legend will not be plotted.
        legend_kwargs : dict, optional
            The keyword arguments of the legend.
        kwargs : dict, optional
            The keyword arguments of the errorbar plot.
                
        Returns
        -------
        ax : matplotlib axis
            The axis of the VISPHI as a function of the uv distance.
        '''
        assert units in ['Mlambda', 'per mas'], 'The units must be Mlambda or per mas.'

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        uvdist = self.get_uvdist(fiber, polarization, units=units)
        visphi, visphierr = self.get_visphi(fiber, polarization)

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'

        if 'ecolor' not in kwargs:
            kwargs['ecolor'] = 'gray'

        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
            kwargs['ls'] = '-'
            
        for bsl in range(N_BASELINE):
            kwargs_use = kwargs.copy()

            if 'color' not in kwargs:
                kwargs_use['color'] = f'C{bsl}'

            for dit in range(visphi.shape[0]):

                if 'label' not in kwargs:
                    if dit == 0:
                        kwargs_use['label'] = f'{self._baseline[bsl]}'
                    else:
                        kwargs_use['label'] = None

                ax.errorbar(
                    uvdist[dit, bsl], visphi[dit, bsl, :], yerr=visphierr[dit, bsl, :], 
                    **kwargs_use)
                
        if show_average:
            visave = np.nanmean(visphi[0, :, :], axis=-1)
            visstd = np.nanstd(visphi[0, :, :], axis=-1)
            visphi_text = '\n'.join([fr'{self._baseline[ii]}: {visave[ii]:.1f} +/- {visstd[ii]:.1f}$^\circ$' 
                                    for ii in range(N_BASELINE)])
            ax.text(0.45, 0.95, visphi_text, fontsize=14, transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            
        if not plain:
            if units == 'Mlambda':
                ax.set_xlabel(r'$uv$ distance (M$\lambda$)', fontsize=24)
            else:
                ax.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24)
            ax.set_ylabel('VISPHI ($^\circ$)', fontsize=24)
            ax.minorticks_on()

            if legend_kwargs is None:
                legend_kwargs = {}

            if 'fontsize' not in legend_kwargs:
                legend_kwargs['fontsize'] = 14

            if 'loc' not in legend_kwargs:
                legend_kwargs['loc'] = 'upper left'

            if 'ncols' not in legend_kwargs:
                legend_kwargs['ncols'] = 2

            ax.legend(**legend_kwargs)

        return

    
    def plot_vis2data_uvdist(self, fiber='SC', polarization=None, units='Mlambda',
                             show_average=False, ax=None, plain=False, legend_kwargs=None, 
                             **kwargs):
        '''
        Plot the VIS2 as a function of the uv distance.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv distance. Either Mlambda or per mas.
        show_average : bool, optional
            If True, the average VIS2 will be shown. Default is False.
        ax : matplotlib axis, optional
            The axis to plot the VIS2 as a function of the uv distance. If None, 
            a new figure and axis will be created.
        plain : bool, optional
            If True, the axis labels and legend will not be plotted.
        legend_kwargs : dict, optional
            The keyword arguments of the legend.
        kwargs : dict, optional
            The keyword arguments of the errorbar plot.
                
        Returns
        -------
        ax : matplotlib axis
            The axis of the VIS2 as a function of the uv distance.
        '''
        assert units in ['Mlambda', 'per mas'], 'The units must be Mlambda or per mas.'

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        uvdist = self.get_uvdist(fiber, polarization, units=units)
        vis2data, vis2err = self.get_vis2data(fiber, polarization)

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'

        if 'ecolor' not in kwargs:
            kwargs['ecolor'] = 'gray'

        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
            kwargs['ls'] = '-'
            
        for bsl in range(N_BASELINE):
            kwargs_use = kwargs.copy()

            if 'color' not in kwargs:
                kwargs_use['color'] = f'C{bsl}'

            for dit in range(vis2data.shape[0]):

                if 'label' not in kwargs:
                    if dit == 0:
                        kwargs_use['label'] = f'{self._baseline[bsl]}'
                    else:
                        kwargs_use['label'] = None

                ax.errorbar(
                    uvdist[dit, bsl], vis2data[dit, bsl, :], yerr=vis2err[dit, bsl, :], 
                    **kwargs_use)
        
        if show_average:
            visave = np.nanmean(vis2data[0, :, :], axis=-1)
            visstd = np.nanstd(vis2data[0, :, :], axis=-1)
            visphi_text = '\n'.join([fr'{self._baseline[ii]}: {visave[ii]:.1f} +/- {visstd[ii]:.1f}$^\circ$' 
                                    for ii in range(N_BASELINE)])
            ax.text(0.45, 0.95, visphi_text, fontsize=14, transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
            
        if not plain:
            if units == 'Mlambda':
                ax.set_xlabel(r'$uv$ distance (M$\lambda$)', fontsize=24)
            else:
                ax.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24)
            ax.set_ylabel('VIS2DATA', fontsize=24)
            ax.minorticks_on()

            if legend_kwargs is None:
                legend_kwargs = {}

            if 'fontsize' not in legend_kwargs:
                legend_kwargs['fontsize'] = 14

            if 'loc' not in legend_kwargs:
                legend_kwargs['loc'] = 'upper left'

            if 'ncols' not in legend_kwargs:
                legend_kwargs['ncols'] = 2

            ax.legend(**legend_kwargs)
        return
    

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


class AstroFits(object):
    '''
    A class to read and plot GRAVITY ASTROREDUCED data.
    '''
    def __init__(self, filename):
        '''
        Parameters
        ----------
        filename : str
            The name of the OIFITS file.
        '''
        self._filename = filename
        self._hdul = fits.open(filename)
        header = self._hdul[0].header
        self._header = header

        self._arcfile = header.get('ARCFILE', None)
        if self._arcfile is not None:
            self._arctime = self._arcfile.split('.')[1]
        else:
            self._arctime = None

        self._object = header.get('OBJECT', None)

        # Instrument mode
        self._pol = header.get('ESO INS POLA MODE', None)
        self._res = header.get('ESO INS SPEC RES', None)

        # Science target
        self._sobj_x = header.get('ESO INS SOBJ X', None)
        self._sobj_y = header.get('ESO INS SOBJ Y', None)
        self._sobj_offx = header.get('ESO INS SOBJ OFFX', None)
        self._sobj_offy = header.get('ESO INS SOBJ OFFY', None)
        self._swap = header.get('ESO INS SOBJ SWAP', None) == 'YES'
        self._dit = header.get('ESO DET2 SEQ1 DIT', None)

        telescop = header.get('TELESCOP', None)
        if 'U1234' in telescop:
            self.set_telescope('UT')
        elif 'A1234' in telescop:
            self.set_telescope('AT')
        else:
            self.set_telescope('GV')


    def correct_met_jump(self, fringe_tel):
        '''
        Correct the metrology phase jump.

        Parameters
        ----------
        opd_tel : array
            The correction values add to OPD_DISP, in the unit of fringe.
            The input value is telescope based, [UT4, UT3, UT2, UT1].
        '''
        wave = self.get_wavelength(units='micron')
        corr_tel = np.array(fringe_tel)[:, np.newaxis] * 2*np.pi * (1 - lambda_met / wave)[np.newaxis, :]
        self._opdDisp_corr = np.dot(t2b_matrix, corr_tel)


    def get_extver(self, polarization=None):
        '''
        Get the extver of the OIFITS HDU. The first digit is the fiber type 
        (1 or 2), and the second digit is the polarization (0, 1, or 2).

        Parameters
        ----------
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        extver : int
            The extver of the OIFITS HDU.
        '''
        if polarization is None:
            if self._pol == 'SPLIT':
                polarization = 1
            else:
                polarization = 0
        assert polarization in [0, 1, 2], 'polarization must be 0, 1, or 2.'

        extver = int(f'1{polarization}')
        return extver
    

    def get_f1f2(self, polarization=None, fromdata=False):
        '''
        Get the geometric flux (F1F2) of the OIFITS HDU.
        '''
        extver = self.get_extver(polarization)

        if hasattr(self, f'_f1f2_{extver}') & (not fromdata):
            return self.__getattribute__(f'_f1f2_{extver}')
        
        f1f2 = self._hdul['OI_VIS', extver].data['F1F2']
        f1f2 = np.reshape(f1f2, (-1, N_BASELINE, f1f2.shape[1]))

        flag = self.get_visflag(polarization, fromdata=fromdata)
        f1f2 = np.ma.array(f1f2, mask=flag)

        self.__setattr__(f'_f1f2_{extver}', f1f2)

        return f1f2


    def get_gdelay(self, polarization=None, fromdata=False, field='GDELAY_BOOT'):
        '''
        Get the GDELAY_BOOT.
        '''
        extver = self.get_extver(polarization)

        if hasattr(self, f'_gdelay_{extver}') & (not fromdata):
            return self.__getattribute__(f'_gdelay_{extver}')
        
        gdelay = self._hdul['OI_VIS', extver].data[field].reshape(-1, N_BASELINE)

        self.__setattr__(f'_gdelay_{extver}', gdelay)

        return gdelay


    def get_offset(self, opdzp, polarization=None, method='leastsq', fromdata=False, 
                   kws_opd_visref={}, plot=False, **kwargs):
        '''
        Get the astrometry offset.
        '''
        opd = self.get_opd_visref(polarization=polarization, fromdata=fromdata, 
                                  opdzp=opdzp, **kws_opd_visref)
        uvcoord = self.get_vis_uvcoord(polarization=polarization, units='m', 
                                       fromdata=fromdata)

        if method == 'leastsq':
            uvcoord = np.array(uvcoord).swapaxes(0, 1)
            offset = np.mean(solve_offset(opd, uvcoord), axis=0)
            chi2_grid = None
            chi2_grid_zoom = None

        elif method == 'grid':
            if 'ra_init' not in kwargs:
                kwargs['ra_init'] = self._sobj_x
            if 'dec_init' not in kwargs:
                kwargs['dec_init'] = self._sobj_y
            
            res = grid_search_opd(opd, uvcoord, plot=plot, **kwargs)

            offset = np.array([res['ra_best'], res['dec_best']])
            chi2_grid = res['chi2_grid']
            chi2_grid_zoom = res['chi2_grid_zoom']
        else:
            raise ValueError(f'The method {method} is not supported!')
        
        if self._swap:
            offset *= -1

        res = dict(offset=offset, chi2_grid=chi2_grid, chi2_grid_zoom=chi2_grid_zoom)

        return res


    def get_opd_visref(self, polarization=None, fromdata=False, opdzp=None, opd_lim=3000, 
                       step=1, zoom=20, iterations=2, progress=False, plot=False):
        '''
        Calculate the OPD from the VISREF data.
        '''
        extver = self.get_extver(polarization)
        visref = self.get_visref(polarization, fromdata=fromdata, opdzp=opdzp)
        visphi = np.angle(visref)
        wave = self.get_wavelength(polarization=polarization, units='micron')

        if progress:
            iterate = tqdm(range(visphi.shape[0]))
        else:
            iterate = range(visphi.shape[0])

        opd = np.array([
            fit_opd_closure(visphi[dit, :, :], wave, opd_lim=opd_lim, step=step, 
                            zoom=zoom, iterations=iterations, plot=plot)
            for dit in iterate])
        return opd


    def get_opdDisp(self, polarization=None, fromdata=False):
        '''
        Get the OPD_DISP data from the OIFITS file.
        '''
        extver = self.get_extver(polarization)
        attr_name = f'_opddisp_{extver}'

        if hasattr(self, attr_name) & (not fromdata):
            return self.__getattribute__(attr_name)
        
        opddisp = self._hdul['OI_VIS', extver].data['OPD_DISP']
        opddisp = np.reshape(opddisp, (-1, N_BASELINE, opddisp.shape[1]))

        self.__setattr__(attr_name, opddisp)

        return opddisp
    

    def get_opdMetCorr(self, polarization=None, fromdata=False):
        '''
        '''
        extver = self.get_extver(polarization)

        if hasattr(self, f'_opdMetCorr_{extver}') & (not fromdata):
            return self.__getattribute__(f'_opdMetCorr_{extver}')

        opdMetFcCorr = self.get_opdMetFcCorr(polarization=polarization, fromdata=fromdata)
        opdMetTelFcMCorr = self.get_opdMetTelFcMCorr(polarization=polarization, fromdata=fromdata)
        opdMetCorr = (opdMetFcCorr + opdMetTelFcMCorr)[:, :, np.newaxis]

        self.__setattr__(f'_opdMetCorr_{extver}', opdMetCorr)

        return opdMetCorr


    def get_opdMetTelFcMCorr(self, polarization=None, fromdata=False):
        '''
        '''
        extver = self.get_extver(polarization)

        if hasattr(self, f'_opdMetTelFcMCorr_{extver}') & (not fromdata):
            return self.__getattribute__(f'_opdMetTelFcMCorr_{extver}')
        
        opdMetTelFcMCorr = self._hdul['OI_VIS', extver].data['OPD_MET_TELFC_MCORR'].reshape(-1, N_BASELINE)

        return opdMetTelFcMCorr
    

    def get_opdMetFcCorr(self, polarization=None, fromdata=False):
        '''
        '''
        extver = self.get_extver(polarization)

        if hasattr(self, f'_opdMetFcCorr_{extver}') & (not fromdata):
            return self.__getattribute__(f'_opdMetFcCorr_{extver}')
        
        opdMetFcCorr = self._hdul['OI_VIS', extver].data['OPD_MET_FC_CORR'].reshape(-1, N_BASELINE)

        return opdMetFcCorr


    def get_phaseref(self, polarization=None, fromdata=False):
        '''
        '''
        extver = self.get_extver(polarization)

        if hasattr(self, f'_phaseref_{extver}') & (not fromdata):
            return self.__getattribute__(f'_phaseref_{extver}')
        
        phaseref = self._hdul['OI_VIS', extver].data['PHASE_REF']
        phaseref = np.reshape(phaseref, (-1, N_BASELINE, phaseref.shape[1]))

        self.__setattr__(f'_phaseref_{extver}', phaseref)

        return phaseref


    def get_uvdist(self, fiber='SC', polarization=None, units='Mlambda', fromdata=False):
        '''
        Get the uv distance of the baselines.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv coordinates. Either Mlambda, permas, or m.
            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.

        Returns
        -------
        uvdist : array
            The uv distance of the baselines. If units=m, the shape is (N_BASELINE, ), 
            otherwise (N_BASELINE, N_CHANNEL).
        '''
        u, v = self.get_vis_uvcoord(polarization, units=units, fromdata=fromdata)
        uvdist = np.sqrt(u**2 + v**2)
        return uvdist


    def get_vis_uvcoord(self, polarization=None, units='m', fromdata=False):
        '''
        Get the u and v coordinates of the baselines.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv coordinates. Either Mlambda, permas, or m.
            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.
        
        Returns
        -------
        ucoord, vcoord : arrays
            The uv coordinate of the baselines. 
            If units=m, the shape is (N_BASELINE, ), otherwise (N_BASELINE, N_CHANNEL).
        '''
        assert units in ['Mlambda', 'permas', 'm'], 'units must be Mlambda, per mass, or m.'

        if hasattr(self, f'_vis_ucoord_{units}') & hasattr(self, f'_vis_vcoord_{units}') & (not fromdata):
            return self.__getattribute__(f'_vis_ucoord_{units}'), self.__getattribute__(f'_vis_vcoord_{units}')

        extver = self.get_extver(polarization)
        ucoord = self._hdul['OI_VIS', extver].data['UCOORD'].reshape(-1, N_BASELINE)
        vcoord = self._hdul['OI_VIS', extver].data['VCOORD'].reshape(-1, N_BASELINE)
        
        if units != 'm':
            wave = self.get_wavelength(polarization, units='micron')
            ucoord = ucoord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
            vcoord = vcoord[:, :, np.newaxis] / wave[np.newaxis, np.newaxis, :]
            
            if units == 'permas':
                ucoord = ucoord * np.pi / 180. / 3600 * 1e3
                vcoord = vcoord * np.pi / 180. / 3600 * 1e3

        setattr(self, f'_vis_ucoord_{units}', ucoord)
        setattr(self, f'_vis_vcoord_{units}', vcoord)
        
        return ucoord, vcoord


    def get_visflag(self, polarization=None, fromdata=False):
        '''
        '''
        extver = self.get_extver(polarization)

        if hasattr(self, f'_visflag_{extver}') & (not fromdata):
            return self.__getattribute__(f'_visflag_{extver}')
        
        flag = self._hdul['OI_VIS', extver].data['FLAG']
        try:
            rejflag  = self._hdul['OI_VIS', extver].data('REJECTION_FLAG')
        except:
            rejflag = np.zeros_like(flag)
        flag = flag | ((rejflag & 19) > 0)
        flag = np.reshape(flag, (-1, N_BASELINE, flag.shape[1]))

        self.__setattr__(f'_visflag_{extver}', flag)

        return flag


    def get_visdata(self, polarization=None, fromdata=False):
        '''
        Get the VISDATA and VISERR of the OIFITS HDU.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        visdata, viserr : masked arrays
            The VISDATA and VISERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
        '''
        extver = self.get_extver(polarization)

        if hasattr(self, f'_visdata_{extver}') & hasattr(self, f'_viserr_{extver}') & (not fromdata):
            return self.__getattribute__(f'_visdata_{extver}'), self.__getattribute__(f'_viserr_{extver}')

        visdata = self._hdul['OI_VIS', extver].data['VISDATA']
        viserr = self._hdul['OI_VIS', extver].data['VISERR']
        visdata = np.reshape(visdata, (-1, N_BASELINE, visdata.shape[1]))
        viserr = np.reshape(viserr, (-1, N_BASELINE, viserr.shape[1]))
        
        flag = self.get_visflag(polarization, fromdata=fromdata)
        visdata = np.ma.array(visdata, mask=flag)
        viserr = np.ma.array(viserr, mask=flag)

        self.__setattr__(f'_visdata_{extver}', visdata)
        self.__setattr__(f'_viserr_{extver}', viserr)

        return visdata, viserr


    def get_visref(self, polarization=None, fromdata=False, opdzp=None, per_dit=False, normalized=False):
        '''
        Get the phase referenced VISDATA (VISREF) for astrometry measurement.
        '''
        extver = self.get_extver(polarization)

        attr_name = f'_visref_{extver}'

        if hasattr(self, '_opdDisp_corr'):
            attr_name += '_metcorr'

        if per_dit:
            attr_name += '_perdit'
        
        if normalized:
            attr_name += '_normalized'

        if opdzp is not None:
            attr_name += f'_zpcorr'

        if hasattr(self, attr_name) & (not fromdata):
            return self.__getattribute__(attr_name)
        
        wave = self.get_wavelength(polarization=polarization, fromdata=fromdata, units='micron')
        visdata, _ = self.get_visdata(polarization=polarization, fromdata=fromdata)
        phaseref= self.get_phaseref(polarization=polarization, fromdata=fromdata)
        opdDisp = self.get_opdDisp(polarization=polarization, fromdata=fromdata)
        opdMetCorr = self.get_opdMetCorr(polarization=polarization, fromdata=fromdata)

        visref = visdata * np.exp(1j * (phaseref - 2*np.pi / wave * (opdDisp + opdMetCorr) * 1e6))

        if hasattr(self, '_opdDisp_corr'):
            visref *= np.exp(1j * self._opdDisp_corr)

        if opdzp is not None:
            v_opd = matrix_opd(opdzp, wave).T
            visref *= np.conj(v_opd[np.newaxis, :, :])

        if per_dit:
            visref /= self._dit

        if normalized:
            f1f2 = self.get_f1f2(polarization=polarization, fromdata=fromdata)
            visref /= np.sqrt(f1f2)
        
        self.__setattr__(f'_visref_{extver}', visref)

        return visref


    def get_visref_rephase(self, phi0=0, radec=None, polarization=None, 
                           fromdata=False, per_dit=False, normalized=False, 
                           plot=False, axs=None, plain=False):
        '''
        Get the re-phased VISREF data to check the astrometry measurement.
        '''
        visref = self.get_visref(polarization=polarization, fromdata=fromdata, 
                                 per_dit=per_dit, normalized=normalized)

        if radec is not None:
            ra, dec = radec
            u, v = self.get_vis_uvcoord(polarization=polarization, units='Mlambda', fromdata=fromdata)
            phase_source = phase_model(u, v, ra, dec)
        else:
            phase_source = 0

        visref_rephase = visref * np.exp(1j * (phase_source - phi0))

        if plot:
            uvdist = self.get_uvdist(polarization=polarization, units='permas', fromdata=fromdata)
            x = uvdist.mean(axis=0)
            y = visref_rephase.mean(axis=0)

            if axs is None:
                fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
                fig.subplots_adjust(hspace=0.02)
            else:
                assert len(axs) == 2, 'axs must have 2 axes.'

            for bsl in range(N_BASELINE):
                ax = axs[0]
                ax.plot(x[bsl, :], np.absolute(y[bsl, :]), color=f'C{bsl}', label=self._baseline[bsl])

                if not plain:
                    ax.legend(loc='lower left', ncols=3, fontsize=14, handlelength=1, columnspacing=1)
                    ax.set_ylabel('VISAMP', fontsize=16)
                    ax.set_ylim([0, 1.1])
                    ax.minorticks_on()

                ax = axs[1]
                ax.plot(x[bsl, :], np.angle(y[bsl, :], deg=True), color=f'C{bsl}')
                
                if not plain:
                    ax.set_ylabel(r'VISPHI ($^\circ$)', fontsize=16)
                    ax.set_ylim([-180, 180])
                    ax.set_xlabel(r'Space frequency (mas$^{-1}$)', fontsize=16)
                    ax.minorticks_on()
        
        return visref_rephase


    def get_wavelength(self, polarization=None, units='micron', fromdata=False):
        '''
        Get the wavelength of the OIFITS HDU.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the wavelength. Either micron or m.
        
        Returns
        -------
        wave : array
            The wavelength of the OIFITS HDU. If units=m, the shape is (N_CHANNEL, ), 
            otherwise (N_CHANNEL, ).
        '''
        assert units in ['micron', 'm'], 'units must be um or m.'

        extver = self.get_extver(polarization)

        if hasattr(self, f'_wave_{extver}') & (not fromdata):
            return self.__getattribute__(f'_wave_{extver}')

        wave = self._hdul['OI_WAVELENGTH', extver].data['EFF_WAVE']

        if units == 'micron':
            wave = wave * 1e6

        self.__setattr__(f'_wave_{extver}', wave)

        return wave


    def plot_rdata_wavelength(self, data, name=None, polarization=None, axs=None, 
                              plain=False, legend_kwargs=None, **kwargs):
        '''
        Plot the real data as a function of the wavelength.

        Parameters
        ----------
        '''
        if axs is None:
            fig, axs = plt.subplots(N_BASELINE, 1, figsize=(12, 6), sharex=True, 
                                    sharey=True)

        wave = self.get_wavelength(polarization, units='micron')

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
            kwargs['ls'] = '-'
            
        for bsl in range(N_BASELINE):
            kwargs_use = kwargs.copy()

            ax = axs[bsl]
            for dit in range(data.shape[0]):

                if 'label' not in kwargs:
                    if dit == 0:
                        kwargs_use['label'] = f'{self._baseline[bsl]}'
                    else:
                        kwargs_use['label'] = None

                if 'color' not in kwargs:
                    kwargs_use['color'] = f'C{dit}'

                ax.plot(wave, data[dit, bsl, :], **kwargs_use)

        if not plain:
            fig.subplots_adjust(hspace=0, wspace=0)
            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=24, labelpad=25)
            axo.set_title(name, fontsize=24)
            ax.minorticks_on()

            if legend_kwargs is None:
                legend_kwargs = {}

            if 'fontsize' not in legend_kwargs:
                legend_kwargs['fontsize'] = 14

            if 'loc' not in legend_kwargs:
                legend_kwargs['loc'] = 'lower right'
            
            if 'bbox_to_anchor' not in legend_kwargs:
                legend_kwargs['bbox_to_anchor'] = (1, 1)

            if 'ncols' not in legend_kwargs:
                legend_kwargs['ncols'] = 2

            axs[0].legend(**legend_kwargs)
        
        return axs

    
    def plot_cdata_wavelength(self, data, name=None, polarization=None, axs=None, 
                              plain=False, legend_kwargs=None, **kwargs):
        '''
        Plot the complex data as a function of the wavelength.

        Parameters
        ----------
        '''
        if axs is None:
            fig, axs = plt.subplots(N_BASELINE, 1, figsize=(12, 6), sharex=True, 
                                    sharey=True)

        wave = self.get_wavelength(polarization, units='micron')

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'

        if ('ls' not in kwargs) & ('linestyle' not in kwargs):
            kwargs['ls'] = '-'
            
        for bsl in range(N_BASELINE):
            kwargs_use = kwargs.copy()

            ax = axs[bsl]
            for dit in range(data.shape[0]):

                if 'label' not in kwargs:
                    if dit == 0:
                        kwargs_use['label'] = f'{self._baseline[bsl]}'
                    else:
                        kwargs_use['label'] = None

                if 'color' not in kwargs:
                    kwargs_use['color'] = f'C0'

                ax.plot(
                    wave, data[dit, bsl, :].real, **kwargs_use)

                if 'color' not in kwargs:
                    kwargs_use['color'] = f'C3'

                ax.errorbar(
                    wave, data[dit, bsl, :].imag, **kwargs_use)

        if not plain:
            fig.subplots_adjust(hspace=0, wspace=0)
            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=24, labelpad=25)
            axo.set_title(name, fontsize=24)
            ax.minorticks_on()

            if legend_kwargs is None:
                legend_kwargs = {}

            if 'fontsize' not in legend_kwargs:
                legend_kwargs['fontsize'] = 14

            if 'loc' not in legend_kwargs:
                legend_kwargs['loc'] = 'lower right'
            
            if 'bbox_to_anchor' not in legend_kwargs:
                legend_kwargs['bbox_to_anchor'] = (1, 1)

            if 'ncols' not in legend_kwargs:
                legend_kwargs['ncols'] = 2

            axs[0].legend(**legend_kwargs)
        
        return axs


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