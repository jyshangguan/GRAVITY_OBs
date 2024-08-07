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


def compute_chi2_astro(ra, dec, oi1List, oi2List, pol=2, per_dit=False, normalized=True):
    '''
    Calculate the chi2 using the AstroFits object (ASTROREDUCED data).

    Parameters
    ----------
    ra : float
        Right ascension offset in milliarcsec.
    dec : float
        Declination offset in milliarcsec.
    oi1List, oi2List : list
        OIFITS data between swap observations, in meters; 
        ((NDIT, NBASELINE), (NDIT, NBASELINE)).
    wave : float
    '''
    gamma1 = []
    gooddata1 = []
    for oi in oi1List:
        visref = oi.get_visref(polarization=pol, per_dit=per_dit, normalized=normalized)
        u, v = oi.get_vis_uvcoord(polarization=pol, units='Mlambda')
        phase = phase_model(ra, dec, u, v)
        model = np.exp(1j * phase)
        gamma1.append(model * visref)
        gooddata1.append(visref.mask == False)
    
    gamma2 = []
    gooddata2 = []
    for oi in oi2List:
        visref = oi.get_visref(polarization=pol, per_dit=per_dit, normalized=normalized)
        u, v = oi.get_vis_uvcoord(polarization=pol, units='Mlambda')
        phase = phase_model(ra, dec, u, v)
        model = np.exp(1j * phase)
        gamma2.append(np.conj(model) * visref)
        gooddata2.append(visref.mask == False)
    
    gamma1 = np.ma.sum(np.concatenate(gamma1), axis=0) / np.sum(np.concatenate(gooddata1), axis=0)
    gamma2 = np.ma.sum(np.concatenate(gamma2), axis=0) / np.sum(np.concatenate(gooddata2), axis=0)
    gamma_swap = (np.conj(gamma1) * gamma2)**0.5  # Important not to use np.sqrt() here!
    chi2 = np.ma.sum(gamma_swap.imag**2)
    chi2_baseline = np.ma.sum(gamma_swap.imag**2, axis=1)
    
    return chi2, chi2_baseline


def grid_search_astro(oi1List, oi2List, ra_init, dec_init, pol=2, per_dit=False, 
                      normalized=True, ra_lim=30, dec_lim=30, nra=100, ndec=100, 
                      zoom=5, plot=True, axs=None, percent=99.5):
    '''
    Perform a grid search to find the best RA and Dec offsets using ASTROREDUCED data.
    '''
    ra_grid = np.linspace(ra_init-ra_lim, ra_init+ra_lim, nra)
    dec_grid = np.linspace(dec_init-dec_lim, dec_init+dec_lim, ndec)
    chi2_grid = np.zeros((nra, ndec))
    
    chi2_best = np.inf
    for i, ra in tqdm(list(enumerate(ra_grid))):
        for j, dec in tqdm(list(enumerate(dec_grid)), leave=False):
            chi2_grid[i, j] = compute_chi2_astro(
                ra, dec, oi1List, oi2List, pol=pol, per_dit=per_dit, 
                normalized=normalized)[0]

            if chi2_grid[i, j] < chi2_best:
                chi2_best = chi2_grid[i, j]
                ra_best, dec_best = ra, dec
    
    ra_grid_zoom = ra_best + np.linspace(-ra_lim, ra_lim, nra) / zoom
    dec_grid_zoom = dec_best + np.linspace(-dec_lim, dec_lim, ndec) / zoom
    chi2_grid_zoom = np.zeros((len(dec_grid), len(ra_grid)))
    chi2_grid_bsl_zoom = np.zeros((len(dec_grid), len(ra_grid), 6))

    chi2_best_zoom = np.inf
    for i, ra in tqdm(list(enumerate(ra_grid_zoom))):
        for j, dec in tqdm(list(enumerate(dec_grid_zoom)), leave=False):
            chi2_grid_zoom[i, j], chi2_grid_bsl_zoom[i, j] = compute_chi2_astro(
                ra, dec, oi1List, oi2List, pol=pol, per_dit=per_dit, 
                normalized=normalized)

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
        im = ax.imshow(chi2_grid.T, origin='lower', norm=norm, extent=[ra_grid[0], ra_grid[-1], dec_grid[0], dec_grid[-1]])
        rect = patches.Rectangle((ra_best-ra_lim/zoom, dec_best-dec_lim/zoom), 
                                 ra_lim/zoom*2, dec_lim/zoom*2, linewidth=1, 
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_xlabel('RA (mas)', fontsize=18)
        ax.set_ylabel('Dec (mas)', fontsize=18)
        ax.set_title(f'Full grid ({ra_best:.2f}, {dec_best:.2f})', fontsize=16)
        ax.plot(ra_init, dec_init, marker='x', ls='none', color='C1', ms=15, label='Initial')
        ax.plot(ra_best, dec_best, marker='+', ls='none', color='C3', ms=15, label='Best-fit')

        ax = axs[1]
        norm = simple_norm(chi2_grid_zoom, stretch='linear', percent=percent)
        im = ax.imshow(chi2_grid_zoom.T, origin='lower', norm=norm, extent=[ra_grid_zoom[0], ra_grid_zoom[-1], dec_grid_zoom[0], dec_grid_zoom[-1]])
        ax.set_xlabel('RA (mas)', fontsize=18)
        ax.set_title(f'Zoomed grid ({ra_best_zoom:.2f}, {dec_best_zoom:.2f})', fontsize=16)
        ax.plot(ra_init, dec_init, marker='x', ls='none', color='C1', ms=15, label='Initial')
        ax.plot(ra_best_zoom, dec_best_zoom, marker='+', ls='none', color='C3', ms=15, label='Best-fit')
        ax.legend(loc='upper right', fontsize=14, frameon=True, framealpha=0.8, handlelength=1)

    results = dict(ra_best=ra_best_zoom, 
                   dec_best=dec_best_zoom, 
                   chi2_best=chi2_best_zoom, 
                   chi2_grid_zoom=chi2_grid_zoom, 
                   chi2_grid_bsl_zoom=chi2_grid_bsl_zoom)   

    return results


def compute_metzp(oi1List, oi2List, ra, dec, pol=2, opd_lim=3000, step=0.1, zoom=30, plot=False, axs=None):
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

    opdzp = fit_opd_closure(phi0, wave, opd_lim=opd_lim, step=step, zoom=zoom, plot=False)
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


def compute_metzp_gd(oi1List, oi2List, polarization=None, fromdata=False, 
                     opd_lim=3000, step=1, zoom=20, iterations=2, progress=False):
    '''
    Compute the metrology zero points from the group delay.
    '''
    opd1List = [oi.get_opd_visref(polarization=polarization, fromdata=fromdata, 
                                  opd_lim=opd_lim, step=step, zoom=zoom, 
                                  iterations=iterations, progress=progress) 
                for oi in oi1List]
    opd2List = [oi.get_opd_visref(polarization=polarization, fromdata=fromdata, 
                                  opd_lim=opd_lim, step=step, zoom=zoom, 
                                  iterations=iterations, progress=progress) 
                for oi in oi2List]

    opd1 = np.mean(opd1List, axis=(0, 1))
    opd2 = np.mean(opd2List, axis=(0, 1))
    opdzp = (opd1 + opd2) / 2
    fczp = np.array([opdzp[2], opdzp[4], opdzp[5], 0])

    results = dict(opdzp=opdzp, fczp=fczp)
    return results


def matrix_opd(opd, wave):
    '''
    Calculate the phase 
    '''
    opd = np.atleast_1d(opd)
    return np.exp(2j * np.pi * opd[None, :] / wave[:, None])


def search_opd(wave, phi0, opd_lim=3000, step=1, zoom=20, iterations=2, plot=False, axs=None):
    '''
    Search for the OPD of the metrology zero point in phase.
    '''
    if zoom == 1:
        iterations = 1

    opd_search = [0]
    phi_search = [phi0]
    opd_limit_search = [opd_lim]
    step_search = [step]
    for loop in range(iterations):
        opd = np.arange(-opd_limit_search[-1], opd_limit_search[-1], step_search[-1])
        v_opd = matrix_opd(opd, wave)
        v_phi = np.exp(1j * phi_search[-1])
        lnl = np.absolute(np.dot(v_phi, np.conj(v_opd)))

        # Compile results for this iteration
        opd_search.append(opd[np.argmax(lnl)])
        v_opd = matrix_opd(opd_search[-1], wave)[:, 0]
        phi_search.append(np.angle(np.exp(1j*phi_search[-1]) * np.conj(v_opd)))
        opd_limit_search.append(opd_limit_search[-1] / zoom)
        step_search.append(step_search[-1] / zoom)
    
    opd_best = np.sum(opd_search)
    
    # Search within a fringe
    opd = opd_best + np.arange(-1, 1, 0.001)   # 1 nm step is fine enough
    v_opd = matrix_opd(opd, wave)
    lnl = np.real(np.dot(np.exp(1j * phi0), np.conj(v_opd)))
    opd_best = opd[np.argmax(lnl)]

    if plot:
        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))

        ax = axs[0]
        ax.plot(opd, lnl, color='k')
        ax.axvline(opd_best, color='r', linestyle='--')
        ax.text(0.05, 0.95, '(a)', fontsize=18, color='k', transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='w', edgecolor='w', alpha=0.5))
        ax.text(0.95, 0.95, f'Best-fit OPD:\n{opd_best:.2f} $\mu$m', fontsize=16, color='k', 
                transform=ax.transAxes, va='top', ha='right')
        ax.set_xlabel(r'OPD ($\mu$m)', fontsize=18)
        ax.set_ylabel('Likelihood', fontsize=18)
        ax.minorticks_on()

        ax = axs[1]
        ax.plot(wave, phi0, color='C0', alpha=0.5, label='Data')
        ax.plot(wave, np.angle(matrix_opd(opd_best, wave)), color='C3', alpha=0.5, label='Model')
        ax.legend(fontsize=16, loc='upper right', handlelength=1)
        ax.text(0.05, 0.95, '(b)', fontsize=18, color='k', transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='w', edgecolor='w', alpha=0.5))
        ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18)
        ax.set_ylabel(r'Phase ($^\circ$)', fontsize=18)

    return opd_best


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


def fit_opd_closure(visphi, wave, opd_lim=3000, step=0.1, zoom=30, iterations=2, closure=True, plot=False, axs=None):
    '''
    Fit the OPD that follow the constraint of the closure to be zero
    '''
    # Fit the OPD
    opd = [search_opd(
        wave, visphi[bsl, :], opd_lim=opd_lim, step=step, zoom=zoom, 
        iterations=iterations, plot=False) 
        for bsl in range(6)]

    if closure:
        zerofc = fit_zerofc(visphi, opd, wave, opd_lim=1)
        opd = np.dot(t2b_matrix, zerofc)

    if plot:
        if axs is None:
            fig, axs = plt.subplots(6, 1, figsize=(8, 8), sharex=True, sharey=True)
            fig.subplots_adjust(hspace=0.02)
            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18, labelpad=20)
            axo.set_ylabel(r'VISPHI (rad)', fontsize=18, labelpad=50)

        for bsl, ax in enumerate(axs):
            ax.plot(wave, visphi[bsl, :], ls='-', c=f'C{bsl}')
            ax.plot(wave, np.angle(matrix_opd(opd[bsl], wave)), ls='--', c=f'gray')
            ax.text(0.95, 0.9, f'{opd[bsl]:.2f} $\mu$m', fontsize=14, color='k', 
                    transform=ax.transAxes, va='top', ha='right', 
                    bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))
            ax.minorticks_on()

    return opd


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
        offset.append(-np.dot(opd[dit, :], uvcoord_pseudo_inverse)) 
    offset = np.array(offset) / np.pi * 180 * 3600 * 1000
    return offset


def compute_chi2_opd(ra, dec, u, v, opd):
    '''
    Compute the chi2 of the OPD astrometry.

    Parameters
    ----------
    ra : float
        Right ascension offset in mas.
    dec : float
        Declination offset in mas.
    u, v : array
        UV coordinates in meters, [NBASELINE].
    opd : array
        OPD data in micron, [NBASELINE].
    '''
    # Compute the OPD model
    model = np.exp(1j * (u * ra + v * dec) * np.pi / 180 / 3.6)  # converted to micron

    # Compute the chi2
    chi2 = np.sum((model * np.exp(1j * opd)).imag**2)
    return chi2


def grid_search_opd(opd, uvcoord, ra_init, dec_init, ra_lim=30, dec_lim=30, nra=100, ndec=100, 
                    zoom=5, plot=False, axs=None, percent=99.5):
    '''
    Perform a grid search to find the best RA and Dec offsets using the OPD data.
    '''
    u, v = uvcoord
    ra_grid = np.linspace(ra_init-ra_lim, ra_init+ra_lim, nra)
    dec_grid = np.linspace(dec_init-dec_lim, dec_init+dec_lim, ndec)
    chi2_grid = np.zeros((nra, ndec))
    
    chi2_best = np.inf
    for i, ra in tqdm(list(enumerate(ra_grid))):
        for j, dec in tqdm(list(enumerate(dec_grid)), leave=False):
            chi2_grid[i, j] = compute_chi2_opd(ra, dec, u, v, opd)

            if chi2_grid[i, j] < chi2_best:
                chi2_best = chi2_grid[i, j]
                ra_best, dec_best = ra, dec
    
    ra_grid_zoom = ra_best + np.linspace(-ra_lim, ra_lim, nra) / zoom
    dec_grid_zoom = dec_best + np.linspace(-dec_lim, dec_lim, ndec) / zoom
    chi2_grid_zoom = np.zeros((len(dec_grid), len(ra_grid)))

    chi2_best_zoom = np.inf
    for i, ra in tqdm(list(enumerate(ra_grid_zoom))):
        for j, dec in tqdm(list(enumerate(dec_grid_zoom)), leave=False):
            chi2_grid_zoom[i, j] = compute_chi2_opd(ra, dec, u, v, opd)

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
        im = ax.imshow(chi2_grid.T, origin='lower', norm=norm, extent=[ra_grid[0], ra_grid[-1], dec_grid[0], dec_grid[-1]])
        rect = patches.Rectangle((ra_best-ra_lim/zoom, dec_best-dec_lim/zoom), 
                                 ra_lim/zoom*2, dec_lim/zoom*2, linewidth=1, 
                                 edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_xlabel('RA (mas)', fontsize=18)
        ax.set_ylabel('Dec (mas)', fontsize=18)
        ax.set_title(f'Full grid ({ra_best:.2f}, {dec_best:.2f})', fontsize=16)
        ax.plot(ra_init, dec_init, marker='x', ls='none', color='C1', ms=15, label='Initial')
        ax.plot(ra_best, dec_best, marker='+', ls='none', color='C3', ms=15, label='Best-fit')

        ax = axs[1]
        norm = simple_norm(chi2_grid_zoom, stretch='linear', percent=percent)
        im = ax.imshow(chi2_grid_zoom.T, origin='lower', norm=norm, extent=[ra_grid_zoom[0], ra_grid_zoom[-1], dec_grid_zoom[0], dec_grid_zoom[-1]])
        ax.set_xlabel('RA (mas)', fontsize=18)
        ax.set_title(f'Zoomed grid ({ra_best_zoom:.2f}, {dec_best_zoom:.2f})', fontsize=16)
        ax.plot(ra_init, dec_init, marker='x', ls='none', color='C1', ms=15, label='Initial')
        ax.plot(ra_best_zoom, dec_best_zoom, marker='+', ls='none', color='C3', ms=15, label='Best-fit')
        ax.legend(loc='upper right', fontsize=14, frameon=True, framealpha=0.8, handlelength=1)

    res = dict(ra_best=ra_best_zoom, 
               dec_best=dec_best_zoom, 
               chi2_best=chi2_best_zoom, 
               chi2_grid=chi2_grid,
               chi2_grid_zoom=chi2_grid_zoom,
               axs=axs)

    return res


def GridSearch_OPD_AstroFitsList(astList, opdzp, ra_init, dec_init, polarization=None, 
                                 fromdata=False, ra_lim=30, dec_lim=30, nra=100, ndec=100, 
                                 zoom=5, plot=False, show_each=True, axs=None, percent=99.5,
                                 verbose=True, **kwargs):
    '''
    OPD astrometry using grid search method for a list of AstroFits files.
    '''
    # Prepare data
    if verbose:
        print('Prepare the data...')

    opd = np.concatenate([
        ast.get_opd_visref(
            polarization=polarization, fromdata=fromdata, opdzp=opdzp, **kwargs) 
            for ast in astList])
    
    uList = []
    vList = []
    for ast in astList:
        u, v = ast.get_vis_uvcoord(polarization=polarization, units='m', fromdata=fromdata) 

        if ast._swap:
            u, v = -u, -v
        
        uList.append(u)
        vList.append(v)
    uvcoord = (np.concatenate(uList), np.concatenate(vList))

    # Grid search
    if verbose:
        print('Grid search...')
    res = grid_search_opd(opd=opd, uvcoord=uvcoord, ra_init=ra_init, dec_init=dec_init, 
                          ra_lim=ra_lim, dec_lim=dec_lim, nra=nra, ndec=ndec, zoom=zoom, 
                          plot=plot, axs=axs, percent=percent)
    
    # Show each file
    if show_each & plot:
        if verbose:
            print('Show each file...')

        ax = res['axs'][1]
        h, l = ax.get_legend_handles_labels()

        labelList = []
        handleList = []
        for i, ast in enumerate(astList):
            s = ast.get_offset(opdzp=opdzp, polarization=polarization, 
                               method='leastsq', fromdata=fromdata, 
                               plot=False)['offset']
            if ast._swap:
                color='C3'
                labelList.append('Swap')
            else:
                color='C0'
                labelList.append('Unswap')
            hd, = ax.plot(s[0], s[1], marker='o', ls='none', mfc=color, mec='w', ms=8, label=labelList[-1])
            handleList.append(hd)

        h.append(handleList[0])
        l.append(labelList[0])
        for i in range(1, len(handleList)):
            if labelList[i] != l[-1]:
                h.append(handleList[i])
                l.append(labelList[i])
                break
        ax.legend(h, l, fontsize=14, loc='upper right', handlelength=1)

    return res


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