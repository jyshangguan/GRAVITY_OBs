import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.visualization import simple_norm
from tqdm import tqdm
from scipy.optimize import minimize
from .gravi_utils import t2b_matrix


def phase_model(ra, dec, u, v):
    '''
    Calculate the phase model.

    Parameters
    ----------
    ra : float
        Right ascension offset in milliarcsec.
    dec : float
        Declination offset in milliarcsec.
    u : float or array
        U coordinate in Mlambda; (NDIT, NBASELINE, NCHANEL).
    v : float or array
        V coordinate in Mlambda; (NDIT, NBASELINE, NCHANEL).
    '''
    phase = 2 * np.pi * (np.pi / 3.6 / 180) * (u * ra + v * dec)
    return phase


def compute_metzp(oi1List, oi2List, ra, dec, pol=2, opd_lim=3000, step=0.1, zoom=30, plot=False, axs=None, verbose=True):
    '''
    Calculate the metrology zero point
    '''
    visref1 = []
    for oi in oi1List:
        assert oi._swap == False
        u, v = oi.get_vis_uvcoord(polarization=pol, units='Mlambda')
        phase = phase_model(ra, dec, u, v)
        visref = oi.get_visref(polarization=pol, per_dit=True)
        visref1.append(visref * np.exp(1j * phase))
    visref1 = np.mean(np.concatenate(visref1), axis=0)
    
    visref2 = []
    for oi in oi2List:
        assert oi._swap == True
        u, v = oi.get_vis_uvcoord(polarization=pol, units='Mlambda')
        phase = phase_model(-ra, -dec, u, v)
        visref = oi.get_visref(polarization=pol, per_dit=True)
        visref2.append(visref * np.exp(1j * phase))
    visref2 = np.mean(np.concatenate(visref2), axis=0)

    phi0 = np.angle(0.5 * (visref1 + visref2))
    
    # Prepare deriving the metrology zeropoint
    wave = oi.get_wavelength(polarization=pol)

    opdzp = fit_opd_closure(phi0, wave, opd_lim=opd_lim, step=step, zoom=zoom, 
                            plot=False, verbose=verbose)
    fczp = np.array([opdzp[2], opdzp[4], opdzp[5], 0])

    if plot:
        if axs is None:
            fig, axs = plt.subplots(6, 1, figsize=(8, 8), sharex=True)
            fig.subplots_adjust(hspace=0.02)
            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18, labelpad=20)
            axo.set_ylabel(r'$\phi_0$ (rad)', fontsize=18, labelpad=50)
            axo.set_title(f'Metrology zero point in phase', fontsize=12, loc='left')

        else:
            assert len(axs) == 6, 'The number of axes must be 6!'

        for bsl, ax in enumerate(axs):
            ax.plot(wave, phi0[bsl, :], color=f'C{bsl}')
            ax.plot(wave, np.angle(matrix_opd(opdzp[bsl], wave)), color='gray', alpha=0.5, 
                    label='OPD model')
            ax.text(0.95, 0.9, f'{opdzp[bsl]:.2f} $\mu$m', fontsize=14, color='k', 
                    transform=ax.transAxes, va='top', ha='right', 
                    bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))

            ax.minorticks_on()
            ax.text(0.02, 0.9, oi._baseline[bsl], transform=ax.transAxes, fontsize=14, 
                    va='top', ha='left', color=f'C{bsl}', fontweight='bold',
                    bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))
        axs[0].legend(fontsize=14, loc='lower right', handlelength=1, 
                      bbox_to_anchor=(1, 1))
        
    return phi0, opdzp, fczp


def matrix_opd(opd, wave):
    '''
    Calculate the phase 
    '''
    opd = np.atleast_1d(opd)
    return np.exp(2j * np.pi * opd[None, :] / wave[:, None])


def opd_model(l1, l2, l3, l4, wave):
    '''
    Compute the OPD model for the given baseline configuration and wavelength.

    Parameters
    ----------
    l1, l2, l3, l4 : float
        The opd for each telescope in micron.
        Note that we adopt l1 to l4 corresponds to UT4 to UT1.
    wave : array
        The wavelength in micron.

    Returns
    -------
    v : array
        The complex array with the phase corresponds to the OPD model.
    '''
    v = np.exp(2j * np.pi / wave[None, :] * np.dot(t2b_matrix, np.array([l1, l2, l3, l4]))[:, None])
    return v


def lossfunc(l, k, phi0, wave): 
    '''
    Compute the loss function for the given baseline configuration and wavelength.
    '''
    l1, l2, l3 = l
    args = [l1, l2, l3, wave]
    args.insert(k, 0)
    model = opd_model(*args)
    return np.sum(np.angle(np.exp(1j * phi0) * np.conj(model))**2)


def fit_zerofc(phi0, opd0, wave, opd_lim=1):
    '''
    Fit the zero frequency of the metrology zero point in phase.
    '''
    # Find a good triangle
    l_try = {3: [opd0[2], opd0[4], opd0[5]],     # l4=0
             2: [opd0[1], opd0[3], -opd0[5]],    # l3=0
             1: [opd0[0], -opd0[3], -opd0[4]],   # l2=0
             0: [-opd0[0], -opd0[1], -opd0[2]]}  # l1=0

    flag_fail = True
    for k, l_init in l_try.items():
        if ~np.isnan(np.sum(l_init)):
            flag_fail = False
            break

    if flag_fail:
        raise AssertionError('Cannot find a good triangle!')

    # Optimize group delay
    bounds = [(l-opd_lim, l+opd_lim) for l in l_init]
    res = minimize(lossfunc, l_init, args=(k, phi0, wave), bounds=bounds)

    if res.success:
        zerofc = np.array([res.x[0], res.x[1], res.x[2], 0])
    else:
        raise ValueError('The optimization is not successful!')

    return zerofc


def solve_offset(opd, uvcoord):
    '''
    Solve the astrometry offsets from the OPD and UV coordinates.

    Parameters
    ----------
    opd : np.ndarray
        The OPD values in micron, [NDIT, NBASELINE].
    uvcoord : np.ndarray
        The UV coordinates in meter, [NDIT, 2, NBASELINE].
    '''
    offset = []
    for dit in range(opd.shape[0]):
        uvcoord_pseudo_inverse = np.linalg.pinv(uvcoord[dit, :, :] * 1e6)
        offset.append(np.ma.dot(opd[dit, :], uvcoord_pseudo_inverse)) 
    offset = np.array(offset) / np.pi * 180 * 3600 * 1000
    return offset


def grid_search(
            chi2_func : callable, 
            chi2_func_args : dict = {},
            ra_lim : float = 30, 
            dec_lim : float = 30, 
            nra : int = 100, 
            ndec : int = 100, 
            zoom : int = 5, 
            plot : bool = True, 
            axs : plt.axes = None, 
            percent : float = 99.5):
        '''
        Perform a grid search to find the best RA and Dec offsets.
        '''
        ra_grid = np.linspace(-ra_lim, ra_lim, nra)
        dec_grid = np.linspace(-dec_lim, dec_lim, ndec)
        chi2_grid = np.zeros((nra, ndec))

        chi2_best = np.inf
        # Add a description to the progress bar
        for i, ra in tqdm(list(enumerate(ra_grid)), desc='Initial grid search'):
            for j, dec in tqdm(list(enumerate(dec_grid)), leave=False):
                chi2_grid[i, j] = chi2_func(ra, dec, **chi2_func_args)[0]
    
                if chi2_grid[i, j] < chi2_best:
                    chi2_best = chi2_grid[i, j]
                    ra_best, dec_best = ra, dec

        ra_grid_zoom = ra_best + np.linspace(-ra_lim, ra_lim, nra) / zoom
        dec_grid_zoom = dec_best + np.linspace(-dec_lim, dec_lim, ndec) / zoom
        chi2_grid_zoom = np.zeros((len(dec_grid), len(ra_grid)))
        chi2_grid_bsl_zoom = np.zeros((len(dec_grid), len(ra_grid), 6))

        chi2_best_zoom = np.inf
        for i, ra in tqdm(list(enumerate(ra_grid_zoom)), desc='Zoomed grid search'):
            for j, dec in tqdm(list(enumerate(dec_grid_zoom)), leave=False):
                chi2_grid_zoom[i, j], chi2_grid_bsl_zoom[i, j] = chi2_func(ra, dec, **chi2_func_args)

                if chi2_grid_zoom[i, j] < chi2_best_zoom:
                    chi2_best_zoom = chi2_grid_zoom[i, j]
                    ra_best_zoom, dec_best_zoom = ra, dec
    
        if plot:
            # Plot both the full grid and the zoomed grid
            if axs is None:
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            else:
                assert len(axs) == 2, 'The number of axes must be 2!'

            ax = axs[0]
            norm = simple_norm(chi2_grid, stretch='linear', percent=percent)
            dra, ddec = np.diff(ra_grid)[0], np.diff(dec_grid)[0]
            extent = [ra_grid[0]-dra/2, ra_grid[-1]+dra/2, dec_grid[0]-ddec/2, dec_grid[-1]+ddec/2]
            im = ax.imshow(chi2_grid.T, origin='lower', norm=norm, extent=extent)
            rect = patches.Rectangle((ra_best-ra_lim/zoom, dec_best-dec_lim/zoom), 
                                     ra_lim/zoom*2, dec_lim/zoom*2, linewidth=1, 
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.set_xlabel(r'$\Delta$RA (mas)', fontsize=18)
            ax.set_ylabel(r'$\Delta$Dec (mas)', fontsize=18)
            ax.set_title(f'Full grid ({ra_best:.2f}, {dec_best:.2f})', fontsize=16)
            ax.plot(0, 0, marker='+', ls='none', color='C1', ms=15, label='Initial')
            ax.plot(ra_best, dec_best, marker='x', ls='none', color='C3', ms=15, label='Best-fit')
            ax.legend(loc='upper right', fontsize=14, frameon=True, framealpha=0.8, handlelength=1)
            ax.minorticks_on()

            ax = axs[1]
            norm = simple_norm(chi2_grid_zoom, stretch='linear', percent=percent)
            dra, ddec = np.diff(ra_grid_zoom)[0], np.diff(dec_grid_zoom)[0]
            extent = [ra_grid_zoom[0]-dra/2, ra_grid_zoom[-1]+dra/2, dec_grid_zoom[0]-ddec/2, dec_grid_zoom[-1]+ddec/2]
            im = ax.imshow(chi2_grid_zoom.T, origin='lower', norm=norm, extent=extent)
            ax.set_xlabel(r'$\Delta$RA (mas)', fontsize=18)
            ax.set_title(f'Zoomed grid ({ra_best_zoom:.2f}, {dec_best_zoom:.2f})', fontsize=16)
            ax.plot(0, 0, marker='+', ls='none', color='C1', ms=15, label='Initial')
            ax.plot(ra_best_zoom, dec_best_zoom, marker='x', ls='none', color='C3', ms=15, label='Best-fit')
            ax.minorticks_on()

        results = dict(ra_best=ra_best,
                       dec_best=dec_best,
                       chi2_best=chi2_best,
                       ra_best_zoom=ra_best_zoom, 
                       dec_best_zoom=dec_best_zoom, 
                       chi2_best_zoom=chi2_best_zoom, 
                       chi2_grid=chi2_grid,
                       chi2_grid_zoom=chi2_grid_zoom, 
                       chi2_grid_bsl_zoom=chi2_grid_bsl_zoom,
                       axs=axs,
                       ra_lim=ra_lim,
                       dec_lim=dec_lim,
                       nra=nra,
                       ndec=ndec,
                       zoom=zoom)

        return results


def compute_gdelay(
        visdata : np.array, 
        wave : np.array, 
        max_width : float = 2000,
        closure : bool = True,
        closure_lim : float = 1.2,
        verbose : bool = True,
        logger : logging.Logger = None):
    '''
    Compute the group delay from the visdata. 
    Same method as GRAVITY pipeline, except that I maximize the real part of 
    the visibility in the last pass.

    Parameters
    ----------
    visdata : np.array
        The visibility data, [NDIT, NBASELINE, NCHANNEL].
    wave : np.array
        The wavelength in micron, [NCHANNEL].
    max_width : float
        The maximum width of the OPD in micron.

    Returns
    -------
    gd : np.array
        The group delay in micron, [NDIT, NBASELINE].
    '''
    lbd = wave.mean()
    sigma = 1 / wave
    coherence = 0.5 * len(sigma) / np.abs(sigma[0] - sigma[-1])

    # First pass; less than max_width   
    width1 = np.min([coherence, max_width])
    step1 = 1 * lbd
    nstep1 = int(width1 / step1)
    opd1 = np.linspace(-width1 / 2, width1 / 2, nstep1)
    waveform1 = np.exp(-2j * np.pi * sigma[:, np.newaxis] * opd1[np.newaxis, :])

    # Second pass; less than 6 * nstep1
    width2 = 6 * step1
    step2 = 0.1 * lbd
    nstep2 = int(width2 / step2)
    opd2 = np.linspace(-width2 / 2, width2 / 2, nstep2)
    waveform2 = np.exp(-2j * np.pi * sigma[:, np.newaxis] * opd2[np.newaxis, :])

    # Third pass; less than 6 * nstep2
    width3 = 6 * step2
    step3 = 0.01 * lbd
    nstep3 = int(width3 / step3)
    opd3 = np.linspace(-width3 / 2, width3 / 2, nstep3)
    waveform3 = np.exp(-2j * np.pi * sigma[:, np.newaxis] * opd3[np.newaxis, :])

    # Compute the group delay for 0th order
    ds = np.mean(np.diff(sigma))
    dphi = np.ma.angle(visdata[:, :, 1:] * np.conj(visdata[:, :, :-1]))
    gd0 = np.ma.mean(dphi, axis=-1) / (2 * np.pi * ds)
    visdata_zero = visdata * np.exp(-2j * np.pi * sigma[np.newaxis, np.newaxis, :] * gd0[:, :, np.newaxis])

    # Find the higher order group delays
    opdList = [opd1, opd2]
    wfList = [waveform1, waveform2]
    gdList = [gd0]

    for opd, wf in zip(opdList, wfList):
        amp = np.abs(np.ma.dot(visdata_zero, wf))
        gd = opd[np.ma.argmax(amp, axis=-1)]
        gdList.append(gd)
        visdata_zero = visdata_zero * np.exp(-2j * np.pi * sigma[np.newaxis, np.newaxis, :] * gd[:, :, np.newaxis])
    
    # Last pass; maximize the real part
    amp = np.real(np.ma.dot(visdata_zero, waveform3))
    gd = opd3[np.ma.argmax(amp, axis=-1)]
    gdList.append(gd)

    gd = np.sum(gdList, axis=0)
    gd[gd0.mask] = np.nan

    if closure:
        if logger is not None:
            logger.info('Fitting for the closure...')
        elif verbose:
            print('Fitting for the closure...')

        flag = []
        visphi = np.angle(visdata)
        for dit in range(gd.shape[0]):
            try:
                zerofc = fit_zerofc(visphi[dit, :, :], gd[dit, :], wave, opd_lim=closure_lim)
                gd[dit, :] = np.dot(t2b_matrix, zerofc)
                flag.append(True)
            except ValueError:
                if logger is not None:
                    logger.warning(f'Cannot find a closed solution within {closure_lim} fringe. Return the initial OPD results!')
                elif verbose:
                    print(f'Cannot find a closed solution within {closure_lim} fringe. Return the initial OPD results!')
            except AssertionError as e:
                if logger is not None:
                    logger.warning(e)
                elif verbose:
                    print(e)
                
                flag.append(False)
        flag = np.array(flag)

    else:
        flag = None

    return gd, flag


def gdelay_astrometry(
        oi1, 
        oi2, 
        polarization : int = None,
        average : bool = True,
        max_width : float = 2000,
        closure : bool = True,
        closure_lim : float = 1.2,
        plot : bool = False,
        ax : plt.axes = None,
        plain : bool = False):
    '''
    Compute the astrometry with a pair of swap data using the group delay method.
    '''
    wave = oi1._wave_sc
    visphi = oi1.diff_visphi(oi2, polarization=polarization, average=True)
    uvcoord = (np.array(oi1._uvcoord_m)[:, :, :, 0].mean(axis=1, keepdims=True) + 
               np.array(oi2._uvcoord_m)[:, :, :, 0].mean(axis=1, keepdims=True)).swapaxes(0, 1)

    gd, _ = compute_gdelay(np.ma.exp(1j*visphi), wave, max_width=max_width, 
                           closure=closure, closure_lim=closure_lim, 
                           verbose=False)
    offset = solve_offset(gd, uvcoord)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        for dit in range(visphi.shape[0]):
            ruv = oi1.get_uvdist(units='Mlambda')[dit, :, :]

            s_B = np.pi / 180 / 3.6 * np.dot(offset[dit, :], uvcoord[dit, :, :])[:, np.newaxis]
            model = np.angle(np.exp(1j * 2*np.pi * s_B / wave[np.newaxis, :]), deg=True)
            for bsl in range(visphi.shape[1]):
                if dit == 0:
                    l1 = oi1._baseline[bsl]
                else:
                    l1 = None
                ax.plot(ruv[bsl, :], np.rad2deg(visphi[dit, bsl, :]), ls='-', color=f'C{bsl}', label=l1)

            for bsl in range(visphi.shape[1]):
                if (dit == 0) & (bsl == 0):
                    l2 = 'Gdelay'
                    l3 = r'$\vec{s} \cdot \vec{B}$'
                else:
                    l2 = None
                    l3 = None
                ax.plot(ruv[bsl, :], np.angle(np.exp(2j * np.pi * gd[dit, bsl] / wave), deg=True), 
                        ls='-', color=f'gray', label=l2)
                ax.plot(ruv[bsl, :], model[bsl, :], ls='-', color=f'k', label=l3)
        
        if not plain:
            ra, dec = offset.mean(axis=0)
            ax.text(0.05, 0.05, f'RA: {ra:.2f}\nDec: {dec:.2f}', fontsize=14, 
                    transform=ax.transAxes, va='bottom', ha='left',
                    bbox=dict(facecolor='w', edgecolor='w', alpha=0.8))
            ax.legend(loc='best', fontsize=14, handlelength=1, columnspacing=1, ncols=3)
            ax.set_ylim([-180, 180])
            ax.set_xlabel(r'UV distance (mas$^{-1}$)', fontsize=18)
            ax.set_ylabel(r'VISPHI (rad)', fontsize=18)
            ax.minorticks_on()

    return np.squeeze(offset)

