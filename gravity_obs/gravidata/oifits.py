import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from .gravi_utils import *


class oifits(object):
    '''
    A class to read and plot the OIFITS data of GRAVITY.
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
        self._pol = header.get('ESO INS POLA MODE', None)
        self._res = header.get('ESO INS SPEC RES', None)

        telescop = header.get('TELESCOP', None)
        if telescop == 'U1234':
            self.set_telescope('UT')
        elif telescop == 'A1234':
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
        '''
        assert fiber in ['SC', 'FT'], 'fiber must be SC or FT.'

        if polarization is None:
            if self._pol == 'SPLIT':
                polarization = 1
            else:
                polarization = 0
        assert polarization in [0, 1, 2], 'polarization must be 0, 1, or 2.'

        extver = self.get_extver(fiber, polarization)
        t3phi = self._hdul['OI_T3', extver].data['T3PHI']
        t3phierr = self._hdul['OI_T3', extver].data['T3PHIERR']
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
        u1coord = self._hdul['OI_T3', extver].data['U1COORD'] 
        v1coord = self._hdul['OI_T3', extver].data['V1COORD']
        u2coord = self._hdul['OI_T3', extver].data['U2COORD'] 
        v2coord = self._hdul['OI_T3', extver].data['V2COORD']
        
        if units != 'm':
            wave = self.get_wavelength(fiber, polarization, units='micron')
            u1coord = u1coord[:, np.newaxis] / wave[np.newaxis, :]
            v1coord = v1coord[:, np.newaxis] / wave[np.newaxis, :]
            u2coord = u2coord[:, np.newaxis] / wave[np.newaxis, :]
            v2coord = v2coord[:, np.newaxis] / wave[np.newaxis, :]
            
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
        ucoord = self._hdul['OI_VIS', extver].data['UCOORD'] 
        vcoord = self._hdul['OI_VIS', extver].data['VCOORD']
        
        if units != 'm':
            wave = self.get_wavelength(fiber, polarization, units='micron')
            ucoord = ucoord[:, np.newaxis] / wave[np.newaxis, :]
            vcoord = vcoord[:, np.newaxis] / wave[np.newaxis, :]
            
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
        visamp, visamperr : arrays
            The VISAMP and VISAMPERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
        '''
        extver = self.get_extver(fiber, polarization)
        visamp = self._hdul['OI_VIS', extver].data['VISAMP']
        visamperr = self._hdul['OI_VIS', extver].data['VISAMPERR']
        visamp = np.reshape(visamp, (-1, N_BASELINE, visamp.shape[1]))
        visamperr = np.reshape(visamperr, (-1, N_BASELINE, visamperr.shape[1]))
        return visamp, visamperr
    

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
        vis2data, vis2err : arrays
            The VIS2DATA and VIS2ERR of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
        '''
        extver = self.get_extver(fiber, polarization)
        vis2data = self._hdul['OI_VIS2', extver].data['VIS2DATA']
        vis2err = self._hdul['OI_VIS2', extver].data['VIS2ERR']
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
    

    def plot_t3_ruv(self, fiber='SC', polarization=None, units='Mlambda', 
                    ax=None, plain=False, legend_kwargs=None, **kwargs):
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
                    uvdist[trl], t3phi[dit, trl, :], yerr=t3phierr[dit, trl, :], 
                    **kwargs_use)

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

    