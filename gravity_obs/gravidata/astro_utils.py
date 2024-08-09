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


def lossfunc(l, phi0, wave): 
    '''
    Compute the loss function for the given baseline configuration and wavelength.
    '''
    l1, l2, l3 = l
    model = opd_model(l1, l2, l3, 0, wave)
    return np.sum(np.angle(np.exp(1j * phi0) * np.conj(model))**2)


def fit_zerofc(phi0, opd0, wave, opd_lim=1):
    '''
    Fit the zero frequency of the metrology zero point in phase.
    '''
    # l1, l2, l3 with the assumption that l4=0
    l_init = [opd0[2], opd0[4], opd0[5]]
    bounds = [(l-opd_lim, l+opd_lim) for l in l_init]
    res = minimize(lossfunc, l_init, args=(phi0, wave), bounds=bounds)

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
        offset.append(np.dot(opd[dit, :], uvcoord_pseudo_inverse)) 
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

    gd = np.ma.sum(gdList, axis=0)

    if closure:
        if logger is not None:
            logger.info('Fitting for the closure...')
        elif verbose:
            print('Fitting for the closure...')

        visphi = np.angle(visdata)
        for dit in range(gd.shape[0]):
            try:
                zerofc = fit_zerofc(visphi[dit, :, :], gd[dit, :], wave, opd_lim=closure_lim)
                gd[dit, :] = np.dot(t2b_matrix, zerofc)
            except ValueError:
                if logger is not None:
                    logger.warning(f'Cannot find a closed solution within {closure_lim} fringe. Return the initial OPD results!')
                elif verbose:
                    print(f'Cannot find a closed solution within {closure_lim} fringe. Return the initial OPD results!')

    return gd


def gdelay_astrometry(
        oi1, 
        oi2, 
        polarization : int = None,
        average : bool = True,
        max_width : float = 2000,
        closure : bool = True,
        closure_lim : float = 1.2,
        plot : bool = False,
        ax : plt.axes = None):
    '''
    Compute the astrometry with a pair of swap data using the group delay method.
    '''
    wave = oi1._wave
    visphi = oi1.diff_visphi(oi2, polarization=polarization, average=average)
    uvcoord = (np.array(oi1._uvcoord_m)[:, :, :, 0] + 
               np.array(oi2._uvcoord_m)[:, :, :, 0]).swapaxes(0, 1)

    if average:
        uvcoord = uvcoord.mean(axis=0, keepdims=True)

    gd = compute_gdelay(np.exp(1j*visphi), wave, max_width=max_width, 
                        closure=closure, closure_lim=closure_lim, 
                        verbose=False)
    offset = solve_offset(gd, uvcoord)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            plain = False
        else:
            plain = True

        for dit in range(visphi.shape[0]):
            ruv = oi1.get_uvdist(units='permas')[dit, :, :]

            s_B = np.pi / 180 / 3.6 * np.dot(offset[dit, :], uvcoord[dit, :, :])[:, np.newaxis]
            model = np.angle(np.exp(1j * 2*np.pi * s_B / wave[np.newaxis, :]))
            for bsl in range(visphi.shape[1]):
                if dit == 0:
                    l1 = oi1._baseline[bsl]
                else:
                    l1 = None
                ax.plot(ruv[bsl, :], visphi[dit, bsl, :], ls='-', color=f'C{bsl}', label=l1)

            for bsl in range(visphi.shape[1]):
                if (dit == 0) & (bsl == 0):
                    l2 = 'Gdelay'
                    l3 = r'$\vec{s} \cdot \vec{B}$'
                else:
                    l2 = None
                    l3 = None
                ax.plot(ruv[bsl, :], np.angle(np.exp(2j * np.pi * gd[dit, bsl] / wave)), 
                        ls='-', color=f'gray', label=l2)
                ax.plot(ruv[bsl, :], model[bsl, :], ls='-', color=f'k', label=l3)
        
        if not plain:
            ax.legend(loc='best', fontsize=14, handlelength=1, columnspacing=1, ncols=3)
            ax.set_ylim([-np.pi, np.pi])
            ax.set_xlabel(r'UV distance (mas$^{-1}$)', fontsize=18)
            ax.set_ylabel(r'VISPHI (rad)', fontsize=18)
            ax.minorticks_on()

    if average & (len(offset.shape) > 1):
        offset = offset[0, :]

    return offset




#def compute_chi2_astro(ra, dec, oi1List, oi2List, pol=2, per_dit=False, normalized=True):
#    '''
#    Calculate the chi2 using the AstroFits object (ASTROREDUCED data).
#
#    Parameters
#    ----------
#    ra : float
#        Right ascension offset in milliarcsec.
#    dec : float
#        Declination offset in milliarcsec.
#    oi1List, oi2List : list
#        OIFITS data between swap observations, in meters; 
#        ((NDIT, NBASELINE), (NDIT, NBASELINE)).
#    wave : float
#    '''
#    gamma1 = []
#    gooddata1 = []
#    for oi in oi1List:
#        visref = oi.get_visref(polarization=pol, per_dit=per_dit, normalized=normalized)
#        u, v = oi.get_vis_uvcoord(polarization=pol, units='Mlambda')
#        phase = phase_model(ra, dec, u, v)
#        model = np.exp(1j * phase)
#        gamma1.append(model * visref)
#        gooddata1.append(visref.mask == False)
#    
#    gamma2 = []
#    gooddata2 = []
#    for oi in oi2List:
#        visref = oi.get_visref(polarization=pol, per_dit=per_dit, normalized=normalized)
#        u, v = oi.get_vis_uvcoord(polarization=pol, units='Mlambda')
#        phase = phase_model(ra, dec, u, v)
#        model = np.exp(1j * phase)
#        gamma2.append(np.conj(model) * visref)
#        gooddata2.append(visref.mask == False)
#    
#    gamma1 = np.ma.sum(np.concatenate(gamma1), axis=0) / np.sum(np.concatenate(gooddata1), axis=0)
#    gamma2 = np.ma.sum(np.concatenate(gamma2), axis=0) / np.sum(np.concatenate(gooddata2), axis=0)
#    gamma_swap = (np.conj(gamma1) * gamma2)**0.5  # Important not to use np.sqrt() here!
#    chi2 = np.ma.sum(gamma_swap.imag**2)
#    chi2_baseline = np.ma.sum(gamma_swap.imag**2, axis=1)
#    
#    return chi2, chi2_baseline
#
#
#def grid_search_astro(oi1List, oi2List, ra_init, dec_init, pol=2, per_dit=False, 
#                      normalized=True, ra_lim=30, dec_lim=30, nra=100, ndec=100, 
#                      zoom=5, plot=True, axs=None, percent=99.5):
#    '''
#    Perform a grid search to find the best RA and Dec offsets using ASTROREDUCED data.
#    '''
#    ra_grid = np.linspace(ra_init-ra_lim, ra_init+ra_lim, nra)
#    dec_grid = np.linspace(dec_init-dec_lim, dec_init+dec_lim, ndec)
#    chi2_grid = np.zeros((nra, ndec))
#    
#    chi2_best = np.inf
#    for i, ra in tqdm(list(enumerate(ra_grid))):
#        for j, dec in tqdm(list(enumerate(dec_grid)), leave=False):
#            chi2_grid[i, j] = compute_chi2_astro(
#                ra, dec, oi1List, oi2List, pol=pol, per_dit=per_dit, 
#                normalized=normalized)[0]
#
#            if chi2_grid[i, j] < chi2_best:
#                chi2_best = chi2_grid[i, j]
#                ra_best, dec_best = ra, dec
#    
#    ra_grid_zoom = ra_best + np.linspace(-ra_lim, ra_lim, nra) / zoom
#    dec_grid_zoom = dec_best + np.linspace(-dec_lim, dec_lim, ndec) / zoom
#    chi2_grid_zoom = np.zeros((len(dec_grid), len(ra_grid)))
#    chi2_grid_bsl_zoom = np.zeros((len(dec_grid), len(ra_grid), 6))
#
#    chi2_best_zoom = np.inf
#    for i, ra in tqdm(list(enumerate(ra_grid_zoom))):
#        for j, dec in tqdm(list(enumerate(dec_grid_zoom)), leave=False):
#            chi2_grid_zoom[i, j], chi2_grid_bsl_zoom[i, j] = compute_chi2_astro(
#                ra, dec, oi1List, oi2List, pol=pol, per_dit=per_dit, 
#                normalized=normalized)
#
#            if chi2_grid_zoom[i, j] < chi2_best_zoom:
#                chi2_best_zoom = chi2_grid_zoom[i, j]
#                ra_best_zoom, dec_best_zoom = ra, dec
#    
#    if plot:
#        # Plot both the full grid and the zoomed grid
#        if axs is None:
#            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#        else:
#            assert len(axs) == 2, 'The number of axes must be 2!'
#
#        ax = axs[0]
#        norm = simple_norm(chi2_grid, stretch='linear', percent=percent)
#        im = ax.imshow(chi2_grid.T, origin='lower', norm=norm, extent=[ra_grid[0], ra_grid[-1], dec_grid[0], dec_grid[-1]])
#        rect = patches.Rectangle((ra_best-ra_lim/zoom, dec_best-dec_lim/zoom), 
#                                 ra_lim/zoom*2, dec_lim/zoom*2, linewidth=1, 
#                                 edgecolor='r', facecolor='none')
#        ax.add_patch(rect)
#        ax.set_xlabel('RA (mas)', fontsize=18)
#        ax.set_ylabel('Dec (mas)', fontsize=18)
#        ax.set_title(f'Full grid ({ra_best:.2f}, {dec_best:.2f})', fontsize=16)
#        ax.plot(ra_init, dec_init, marker='x', ls='none', color='C1', ms=15, label='Initial')
#        ax.plot(ra_best, dec_best, marker='+', ls='none', color='C3', ms=15, label='Best-fit')
#
#        ax = axs[1]
#        norm = simple_norm(chi2_grid_zoom, stretch='linear', percent=percent)
#        im = ax.imshow(chi2_grid_zoom.T, origin='lower', norm=norm, extent=[ra_grid_zoom[0], ra_grid_zoom[-1], dec_grid_zoom[0], dec_grid_zoom[-1]])
#        ax.set_xlabel('RA (mas)', fontsize=18)
#        ax.set_title(f'Zoomed grid ({ra_best_zoom:.2f}, {dec_best_zoom:.2f})', fontsize=16)
#        ax.plot(ra_init, dec_init, marker='x', ls='none', color='C1', ms=15, label='Initial')
#        ax.plot(ra_best_zoom, dec_best_zoom, marker='+', ls='none', color='C3', ms=15, label='Best-fit')
#        ax.legend(loc='upper right', fontsize=14, frameon=True, framealpha=0.8, handlelength=1)
#
#    results = dict(ra_best=ra_best_zoom, 
#                   dec_best=dec_best_zoom, 
#                   chi2_best=chi2_best_zoom, 
#                   chi2_grid_zoom=chi2_grid_zoom, 
#                   chi2_grid_bsl_zoom=chi2_grid_bsl_zoom)   
#
#    return results


#def compute_metzp_gd(oi1List, oi2List, polarization=None, fromdata=False, 
#                     opd_lim=3000, step=1, zoom=20, iterations=2, progress=False):
#    '''
#    Compute the metrology zero points from the group delay.
#    '''
#    opd1List = [oi.get_opd_visref(polarization=polarization, fromdata=fromdata, 
#                                  opd_lim=opd_lim, step=step, zoom=zoom, 
#                                  iterations=iterations, progress=progress) 
#                for oi in oi1List]
#    opd2List = [oi.get_opd_visref(polarization=polarization, fromdata=fromdata, 
#                                  opd_lim=opd_lim, step=step, zoom=zoom, 
#                                  iterations=iterations, progress=progress) 
#                for oi in oi2List]
#
#    opd1 = np.mean(opd1List, axis=(0, 1))
#    opd2 = np.mean(opd2List, axis=(0, 1))
#    opdzp = (opd1 + opd2) / 2
#    fczp = np.array([opdzp[2], opdzp[4], opdzp[5], 0])
#
#    results = dict(opdzp=opdzp, fczp=fczp)
#    return results


#def search_opd(wave, phi0, opd_lim=3000, step=1, zoom=20, iterations=2, plot=False, axs=None):
#    '''
#    Search for the OPD of the metrology zero point in phase.
#    '''
#    if zoom == 1:
#        iterations = 1
#
#    opd_search = [0]
#    phi_search = [phi0]
#    opd_limit_search = [opd_lim]
#    step_search = [step]
#    for loop in range(iterations):
#        opd = np.arange(-opd_limit_search[-1], opd_limit_search[-1], step_search[-1])
#        v_opd = matrix_opd(opd, wave)
#        v_phi = np.exp(1j * phi_search[-1])
#        lnl = np.absolute(np.dot(v_phi, np.conj(v_opd)))
#
#        # Compile results for this iteration
#        opd_search.append(opd[np.argmax(lnl)])
#        v_opd = matrix_opd(opd_search[-1], wave)[:, 0]
#        phi_search.append(np.angle(np.exp(1j*phi_search[-1]) * np.conj(v_opd)))
#        opd_limit_search.append(opd_limit_search[-1] / zoom)
#        step_search.append(step_search[-1] / zoom)
#    
#    opd_best = np.sum(opd_search)
#    
#    # Search within a fringe
#    opd = opd_best + np.arange(-1, 1, 0.001)   # 1 nm step is fine enough
#    v_opd = matrix_opd(opd, wave)
#    lnl = np.real(np.dot(np.exp(1j * phi0), np.conj(v_opd)))
#    opd_best = opd[np.argmax(lnl)]
#
#    if plot:
#        if axs is None:
#            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
#
#        ax = axs[0]
#        ax.plot(opd, lnl, color='k')
#        ax.axvline(opd_best, color='r', linestyle='--')
#        ax.text(0.05, 0.95, '(a)', fontsize=18, color='k', transform=ax.transAxes, va='top', ha='left',
#                bbox=dict(facecolor='w', edgecolor='w', alpha=0.5))
#        ax.text(0.95, 0.95, f'Best-fit OPD:\n{opd_best:.2f} $\mu$m', fontsize=16, color='k', 
#                transform=ax.transAxes, va='top', ha='right')
#        ax.set_xlabel(r'OPD ($\mu$m)', fontsize=18)
#        ax.set_ylabel('Likelihood', fontsize=18)
#        ax.minorticks_on()
#
#        ax = axs[1]
#        ax.plot(wave, phi0, color='C0', alpha=0.5, label='Data')
#        ax.plot(wave, np.angle(matrix_opd(opd_best, wave)), color='C3', alpha=0.5, label='Model')
#        ax.legend(fontsize=16, loc='upper right', handlelength=1)
#        ax.text(0.05, 0.95, '(b)', fontsize=18, color='k', transform=ax.transAxes, va='top', ha='left',
#                bbox=dict(facecolor='w', edgecolor='w', alpha=0.5))
#        ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18)
#        ax.set_ylabel(r'Phase ($^\circ$)', fontsize=18)
#
#    return opd_best
#
#
#def GridSearch_OPD_AstroFitsList(astList, opdzp, ra_init, dec_init, polarization=None, 
#                                 fromdata=False, ra_lim=30, dec_lim=30, nra=100, ndec=100, 
#                                 zoom=5, plot=False, show_each=True, axs=None, percent=99.5,
#                                 verbose=True, **kwargs):
#    '''
#    OPD astrometry using grid search method for a list of AstroFits files.
#    '''
#    # Prepare data
#    if verbose:
#        print('Prepare the data...')
#
#    opd = np.concatenate([
#        ast.get_opd_visref(
#            polarization=polarization, fromdata=fromdata, opdzp=opdzp, **kwargs) 
#            for ast in astList])
#    
#    uList = []
#    vList = []
#    for ast in astList:
#        u, v = ast.get_vis_uvcoord(polarization=polarization, units='m', fromdata=fromdata) 
#
#        if ast._swap:
#            u, v = -u, -v
#        
#        uList.append(u)
#        vList.append(v)
#    uvcoord = (np.concatenate(uList), np.concatenate(vList))
#
#    # Grid search
#    if verbose:
#        print('Grid search...')
#    res = grid_search_opd(opd=opd, uvcoord=uvcoord, ra_init=ra_init, dec_init=dec_init, 
#                          ra_lim=ra_lim, dec_lim=dec_lim, nra=nra, ndec=ndec, zoom=zoom, 
#                          plot=plot, axs=axs, percent=percent)
#    
#    # Show each file
#    if show_each & plot:
#        if verbose:
#            print('Show each file...')
#
#        ax = res['axs'][1]
#        h, l = ax.get_legend_handles_labels()
#
#        labelList = []
#        handleList = []
#        for i, ast in enumerate(astList):
#            s = ast.get_offset(opdzp=opdzp, polarization=polarization, 
#                               method='leastsq', fromdata=fromdata, 
#                               plot=False)['offset']
#            if ast._swap:
#                color='C3'
#                labelList.append('Swap')
#            else:
#                color='C0'
#                labelList.append('Unswap')
#            hd, = ax.plot(s[0], s[1], marker='o', ls='none', mfc=color, mec='w', ms=8, label=labelList[-1])
#            handleList.append(hd)
#
#        h.append(handleList[0])
#        l.append(labelList[0])
#        for i in range(1, len(handleList)):
#            if labelList[i] != l[-1]:
#                h.append(handleList[i])
#                l.append(labelList[i])
#                break
#        ax.legend(h, l, fontsize=14, loc='upper right', handlelength=1)
#
#    return res


#def compute_chi2_opd(ra, dec, u, v, opd):
#    '''
#    Compute the chi2 of the OPD astrometry.
#
#    Parameters
#    ----------
#    ra : float
#        Right ascension offset in mas.
#    dec : float
#        Declination offset in mas.
#    u, v : array
#        UV coordinates in meters, [NBASELINE].
#    opd : array
#        OPD data in micron, [NBASELINE].
#    '''
#    # Compute the OPD model
#    model = np.exp(1j * (u * ra + v * dec) * np.pi / 180 / 3.6)  # converted to micron
#
#    # Compute the chi2
#    chi2 = np.sum((model * np.exp(1j * opd)).imag**2)
#    return chi2
#
#
#def grid_search_opd(opd, uvcoord, ra_init, dec_init, ra_lim=30, dec_lim=30, nra=100, ndec=100, 
#                    zoom=5, plot=False, axs=None, percent=99.5):
#    '''
#    Perform a grid search to find the best RA and Dec offsets using the OPD data.
#    '''
#    u, v = uvcoord
#    ra_grid = np.linspace(ra_init-ra_lim, ra_init+ra_lim, nra)
#    dec_grid = np.linspace(dec_init-dec_lim, dec_init+dec_lim, ndec)
#    chi2_grid = np.zeros((nra, ndec))
#    
#    chi2_best = np.inf
#    for i, ra in tqdm(list(enumerate(ra_grid))):
#        for j, dec in tqdm(list(enumerate(dec_grid)), leave=False):
#            chi2_grid[i, j] = compute_chi2_opd(ra, dec, u, v, opd)
#
#            if chi2_grid[i, j] < chi2_best:
#                chi2_best = chi2_grid[i, j]
#                ra_best, dec_best = ra, dec
#    
#    ra_grid_zoom = ra_best + np.linspace(-ra_lim, ra_lim, nra) / zoom
#    dec_grid_zoom = dec_best + np.linspace(-dec_lim, dec_lim, ndec) / zoom
#    chi2_grid_zoom = np.zeros((len(dec_grid), len(ra_grid)))
#
#    chi2_best_zoom = np.inf
#    for i, ra in tqdm(list(enumerate(ra_grid_zoom))):
#        for j, dec in tqdm(list(enumerate(dec_grid_zoom)), leave=False):
#            chi2_grid_zoom[i, j] = compute_chi2_opd(ra, dec, u, v, opd)
#
#            if chi2_grid_zoom[i, j] < chi2_best_zoom:
#                chi2_best_zoom = chi2_grid_zoom[i, j]
#                ra_best_zoom, dec_best_zoom = ra, dec
#
#    if plot:
#        # Plot both the full grid and the zoomed grid
#        if axs is None:
#            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#        else:
#            assert len(axs) == 2, 'The number of axes must be 2!'
#
#        ax = axs[0]
#        norm = simple_norm(chi2_grid, stretch='linear', percent=percent)
#        im = ax.imshow(chi2_grid.T, origin='lower', norm=norm, extent=[ra_grid[0], ra_grid[-1], dec_grid[0], dec_grid[-1]])
#        rect = patches.Rectangle((ra_best-ra_lim/zoom, dec_best-dec_lim/zoom), 
#                                 ra_lim/zoom*2, dec_lim/zoom*2, linewidth=1, 
#                                 edgecolor='r', facecolor='none')
#        ax.add_patch(rect)
#        ax.set_xlabel('RA (mas)', fontsize=18)
#        ax.set_ylabel('Dec (mas)', fontsize=18)
#        ax.set_title(f'Full grid ({ra_best:.2f}, {dec_best:.2f})', fontsize=16)
#        ax.plot(ra_init, dec_init, marker='x', ls='none', color='C1', ms=15, label='Initial')
#        ax.plot(ra_best, dec_best, marker='+', ls='none', color='C3', ms=15, label='Best-fit')
#
#        ax = axs[1]
#        norm = simple_norm(chi2_grid_zoom, stretch='linear', percent=percent)
#        im = ax.imshow(chi2_grid_zoom.T, origin='lower', norm=norm, extent=[ra_grid_zoom[0], ra_grid_zoom[-1], dec_grid_zoom[0], dec_grid_zoom[-1]])
#        ax.set_xlabel('RA (mas)', fontsize=18)
#        ax.set_title(f'Zoomed grid ({ra_best_zoom:.2f}, {dec_best_zoom:.2f})', fontsize=16)
#        ax.plot(ra_init, dec_init, marker='x', ls='none', color='C1', ms=15, label='Initial')
#        ax.plot(ra_best_zoom, dec_best_zoom, marker='+', ls='none', color='C3', ms=15, label='Best-fit')
#        ax.legend(loc='upper right', fontsize=14, frameon=True, framealpha=0.8, handlelength=1)
#
#    res = dict(ra_best=ra_best_zoom, 
#               dec_best=dec_best_zoom, 
#               chi2_best=chi2_best_zoom, 
#               chi2_grid=chi2_grid,
#               chi2_grid_zoom=chi2_grid_zoom,
#               axs=axs)
#
#    return res
#
#
#def fit_opd_closure(visphi : np.array, 
#                    wave : np.array, 
#                    opd_lim : float = 3000, 
#                    step : float = 0.1, 
#                    zoom : float = 30, 
#                    iterations : int = 2, 
#                    closure : bool = True, 
#                    fringe_lim : float = 1,
#                    plot : bool = False, 
#                    axs : plt.axes = None,
#                    verbose : bool = True):
#    '''
#    Fit the OPD that follow the constraint of the closure to be zero
#    '''
#    # Fit the OPD
#    if verbose:
#        print('Searching for the OPD...')
#
#    opd = [search_opd(
#        wave, visphi[bsl, :], opd_lim=opd_lim, step=step, zoom=zoom, 
#        iterations=iterations, plot=False) 
#        for bsl in range(6)]
#
#    if closure:
#        if verbose:
#            print('Fitting for the closure...')
#
#        try:
#            zerofc = fit_zerofc(visphi, opd, wave, opd_lim=fringe_lim)
#            opd = np.dot(t2b_matrix, zerofc)
#        except ValueError:
#            if verbose:
#                print(f'Cannot find a closed solution within {fringe_lim} fringe. Return the initial OPD results!')
#
#    if plot:
#        if axs is None:
#            fig, axs = plt.subplots(6, 1, figsize=(8, 8), sharex=True, sharey=True)
#            fig.subplots_adjust(hspace=0.02)
#            axo = fig.add_subplot(111, frameon=False) # The out axis
#            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
#            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
#            axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18, labelpad=20)
#            axo.set_ylabel(r'VISPHI (rad)', fontsize=18, labelpad=50)
#
#        for bsl, ax in enumerate(axs):
#            ax.plot(wave, visphi[bsl, :], ls='-', c=f'C{bsl}')
#            ax.plot(wave, np.angle(matrix_opd(opd[bsl], wave)), ls='--', c=f'gray')
#            ax.text(0.95, 0.9, f'{opd[bsl]:.2f} $\mu$m', fontsize=14, color='k', 
#                    transform=ax.transAxes, va='top', ha='right', 
#                    bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))
#            ax.minorticks_on()
#
#    return opd


#def fit_opd(wave, phi0, opd_lim=1000, step=0.1, zoom=1, plot=False, axs=None):
#    '''
#    Fit the OPD of the metrology zero point in phase.
#    '''
#    opd = np.arange(-opd_lim, opd_lim, step)
#    v_opd = matrix_opd(opd, wave)
#    lnl = np.absolute(np.dot(np.exp(1j * phi0), np.conj(v_opd)))
#    opd_best = opd[np.argmax(lnl)]
#
#    if zoom > 1:
#        opd = opd_best + np.arange(-opd_lim, opd_lim+step, step) / zoom
#        v_opd = matrix_opd(opd, wave)
#        lnl = np.absolute(np.dot(np.exp(1j * phi0), np.conj(v_opd)))
#        opd_best = opd[np.argmax(lnl)]
#
#    if plot:
#        if axs is None:
#            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
#
#        ax = axs[0]
#        ax.plot(opd, lnl, color='k')
#        ax.axvline(opd_best, color='r', linestyle='--')
#        ax.text(0.05, 0.95, '(a)', fontsize=18, color='k', transform=ax.transAxes, va='top', ha='left',
#                bbox=dict(facecolor='w', edgecolor='w', alpha=0.5))
#        ax.text(0.95, 0.95, f'Best-fit OPD:\n{opd_best:.2f} $\mu$m', fontsize=16, color='k', 
#                transform=ax.transAxes, va='top', ha='right')
#        ax.set_xlabel(r'OPD ($\mu$m)', fontsize=18)
#        ax.set_ylabel('Likelihood', fontsize=18)
#        ax.minorticks_on()
#
#        ax = axs[1]
#        ax.plot(wave, phi0, color='C0', alpha=0.5, label='Data')
#        ax.plot(wave, np.angle(matrix_opd(opd_best, wave)), color='C3', alpha=0.5, label='Model')
#        ax.legend(fontsize=16, loc='upper right', handlelength=1)
#        ax.text(0.05, 0.95, '(b)', fontsize=18, color='k', transform=ax.transAxes, va='top', ha='left',
#                bbox=dict(facecolor='w', edgecolor='w', alpha=0.5))
#        ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18)
#        ax.set_ylabel(r'Phase ($^\circ$)', fontsize=18)
#
#    return opd_best