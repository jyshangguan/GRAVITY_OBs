import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import PercentileInterval, LogStretch, LinearStretch, ImageNormalize
from astropy.modeling import models, fitting


def get_acq_field(hdul, ymin=0, ymax=250):
    '''
    Get the acquisition field.
    '''
    acqImg = hdul['IMAGING_DATA_ACQ'].data
    acqImg_avg = np.average(acqImg, axis=0)[ymin:ymax, :]
    return acqImg_avg


def get_fiber_position(header):
    '''
    Get the fiber position from the header.
    '''
    xList = []
    yList = []
    for loop in range(16):
        xList.append(header['HIERARCH ESO DET1 FRAM{} STRX'.format(1+loop)])
        yList.append(header['HIERARCH ESO DET1 FRAM{} STRY'.format(1+loop)])
    xList = np.array(xList)
    yList = np.array(yList)
    
    fltr = yList < 400
    xList_sel = xList[fltr]
    yList_sel = yList[fltr]
    
    idx = np.argsort(xList_sel)
    xList_sel = xList_sel[idx]
    yList_sel = yList_sel[idx]
    
    ftXList = []
    ftYList = []
    scXList = []
    scYList = []
    for loop in range(4):
        ftXList.append(header['HIERARCH ESO ACQ FIBER FT{}X'.format(loop+1)])
        ftYList.append(header['HIERARCH ESO ACQ FIBER FT{}Y'.format(loop+1)])
        scXList.append(header['HIERARCH ESO ACQ FIBER SC{}X'.format(loop+1)])
        scYList.append(header['HIERARCH ESO ACQ FIBER SC{}Y'.format(loop+1)])
    ftXList = np.array(ftXList)
    ftYList = np.array(ftYList)
    scXList = np.array(scXList)
    scYList = np.array(scYList)

    shiftList = np.array([0, 250, 500, 750])
    ft_x = (ftXList - xList_sel) + shiftList
    ft_y = (ftYList - yList_sel)
    sc_x = (scXList - xList_sel) + shiftList
    sc_y = (scYList - yList_sel)

    fiber_pos_ft = np.array([ft_x, ft_y]).T
    fiber_pos_sc = np.array([sc_x, sc_y]).T
    return fiber_pos_ft, fiber_pos_sc


def plot_fiber_position(filepath, percentile=99.0):
    '''
    Plot the fiber position.
    '''
    hdul = fits.open(filepath)
    header = hdul[0].header
    acqImg_avg = get_acq_field(hdul)
    
    pos_ft, pos_sc = get_fiber_position(header)
    
    fig, ax = plt.subplots(figsize=(20, 5))
    
    norm = ImageNormalize(acqImg_avg, interval=PercentileInterval(percentile), stretch=LinearStretch())
    ax.imshow(acqImg_avg, origin='lower', cmap='hot', norm=norm)
    
    ax.plot(pos_ft[:, 0], pos_ft[:, 1], ls='none', color='k', marker='+', ms=10, zorder=5)
    ax.plot(pos_sc[:, 0], pos_sc[:, 1], ls='none', color='green', marker='+', ms=5, zorder=5)
    #ax.set_ylim([0, 250])

    filename = filepath.split('/')[-1]
    ax.set_title(filename, fontsize=18)
    plt.show()


def get_north_angle(header):
    '''
    Get the north angle from the header.
    '''
    return np.array([header[f'ESO QC ACQ FIELD{tel+1} NORTH_ANGLE'] for tel in range(4)])


def cal_north_angle(header):
    '''
    Calculate the north angle from the header.
    '''
    drottoff = np.array([header[f'ESO INS DROTOFF{ii}'] for ii in range(1, 5)])
    sobj_x = header['ESO INS SOBJ X']
    sobj_y = header['ESO INS SOBJ Y']

    posangle = np.arctan2(sobj_x, sobj_y) * 180 / np.pi
    fangle = - posangle - drottoff + 270

    for loop in range(len(fangle)):
        if (fangle[loop] >= 180):
            fangle[loop] -= 360.0
        if (fangle[loop] < -180):
            fangle[loop] += 360.0
        if (fangle[loop] >= 180):
            fangle[loop] -= 360.0
        if (fangle[loop] < -180):
            fangle[loop] += 360.0
    return fangle


def rotmat(x):
    '''
    Rotation matrix.
    '''
    return np.array([[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]])


def fit_source_position(image, xy, box_size=10, amplitude=None, plot=True, percentile=99.0):
    '''
    Fit the source position.
    '''
    x0, y0 = xy
    x_min = np.max([int(x0 - box_size // 2), 0])
    x_max = np.min([int(x0 + (box_size - box_size // 2)), image.shape[1]])
    y_min = np.max([int(y0 - box_size // 2), 0])
    y_max = np.min([int(y0 + (box_size - box_size // 2)), image.shape[0]])
    img_fit = image[y_min:y_max, x_min:x_max]

    ycoord, xcoord = np.meshgrid(np.arange(box_size), np.arange(box_size))

    if amplitude is None:
        amplitude = np.max(img_fit)

    p_init = models.Gaussian2D(amplitude=amplitude, x_mean=x0-x_min, y_mean=y0-y_min)
    fit_p = fitting.LevMarLSQFitter()
    p = fit_p(p_init, xcoord, ycoord, img_fit, maxiter=10000)

    x_fit = p.x_mean.value + x_min
    y_fit = p.y_mean.value + y_min

    if plot:
        fig, ax = plt.subplots(figsize=(20, 5))
    
        norm = ImageNormalize(image, interval=PercentileInterval(percentile), stretch=LinearStretch())
        ax.imshow(image, origin='lower', cmap='hot', norm=norm)
        ax.plot(x0, y0, ls='none', color='C0', marker='+', ms=10, zorder=5, label='Initial')
        ax.plot(x_fit, y_fit, ls='none', color='C1', marker='+', ms=10, zorder=5, label='Fit')
        ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1.0, 1))

    return x_fit, y_fit