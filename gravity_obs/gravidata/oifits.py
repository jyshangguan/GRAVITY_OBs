import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.decorators import lazyproperty
from .gravi_utils import *


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


    def get_t3_t3phi(self, fiber='SC', polarization=None):
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
    

    def get_vis_visamp(self, fiber='SC', polarization=None):
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
        flag = self._hdul['OI_VIS', extver].data['FLAG']
        visamp = np.ma.array(self._hdul['OI_VIS', extver].data['VISAMP'], mask=flag)
        visamperr = np.ma.array(self._hdul['OI_VIS', extver].data['VISAMPERR'], mask=flag)
        visamp = np.reshape(visamp, (-1, N_BASELINE, visamp.shape[1]))
        visamperr = np.reshape(visamperr, (-1, N_BASELINE, visamperr.shape[1]))
        return visamp, visamperr
    

    def get_vis_visdata(self, fiber='SC', polarization=None):
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
        flag = self._hdul['OI_VIS', extver].data['FLAG']
        visdata = np.ma.array(self._hdul['OI_VIS', extver].data['VISDATA'], mask=flag)
        viserr = np.ma.array(self._hdul['OI_VIS', extver].data['VISERR'], mask=flag)
        visdata = np.reshape(visdata, (-1, N_BASELINE, visdata.shape[1]))
        viserr = np.reshape(viserr, (-1, N_BASELINE, viserr.shape[1]))
        return visdata, viserr
    

    def get_vis_visphi(self, fiber='SC', polarization=None):
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
        flag = self._hdul['OI_VIS', extver].data['FLAG']
        visphi = np.ma.array(self._hdul['OI_VIS', extver].data['VISPHI'], mask=flag)
        visphierr = np.ma.array(self._hdul['OI_VIS', extver].data['VISPHIERR'], mask=flag)
        visphi = np.reshape(visphi, (-1, N_BASELINE, visphi.shape[1]))
        visphierr = np.reshape(visphierr, (-1, N_BASELINE, visphierr.shape[1]))
        return visphi, visphierr


    def get_vis2_vis2data(self, fiber='SC', polarization=None):
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
        flag = self._hdul['OI_VIS', extver].data['FLAG']
        vis2data = np.ma.array(self._hdul['OI_VIS2', extver].data['VIS2DATA'], mask=flag)
        vis2err = np.ma.array(self._hdul['OI_VIS2', extver].data['VIS2ERR'], mask=flag)
        vis2data = np.reshape(vis2data, (-1, N_BASELINE, vis2data.shape[1]))
        vis2err = np.reshape(vis2err, (-1, N_BASELINE, vis2err.shape[1]))
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
        t3phi, t3phierr = self.get_t3_t3phi(fiber, polarization)

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
        visamp, visamperr = self.get_vis_visamp(fiber, polarization)

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
        visdata, viserr = self.get_vis_visdata(fiber, polarization)

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
        visphi, visphierr = self.get_vis_visphi(fiber, polarization)

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
        vis2data, vis2err = self.get_vis2_vis2data(fiber, polarization)

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