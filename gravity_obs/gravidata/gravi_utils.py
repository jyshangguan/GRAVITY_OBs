import numpy as np
from scipy.ndimage import median_filter
from astropy.stats import sigma_clipped_stats

N_TELESCOPE = 4
N_BASELINE = 6
N_TRIANGLE = 4


telescope_names = {
    'UT': ['U4', 'U3', 'U2', 'U1'],
    'AT': ['A4', 'A3', 'A2', 'A1'],
    'GV': ['G1', 'G2', 'G3', 'G4']
}


baseline_names = {
    'UT': ['U43', 'U42', 'U41', 'U32', 'U31', 'U21'],
    'AT': ['A43', 'A42', 'A41', 'A32', 'A31', 'A21'],
    'GV': ['G12', 'G13', 'G14', 'G23', 'G24', 'G34'],
}


triangle_names = {
    'UT': ['U432', 'U431', 'U421', 'U321'],
    'AT': ['A432', 'A431', 'A421', 'A321'],
    'GV': ['G123', 'G124', 'G134', 'G234'],
}


t2b_matrix = np.array([[1, -1, 0, 0],
                       [1, 0, -1, 0],
                       [1, 0, 0, -1],
                       [0, 1, -1, 0],
                       [0, 1, 0, -1],
                       [0, 0, 1, -1]])

lambda_met = 1.908  # micron


def mask_outlier(x, size=10, nsigma=3, maxiters=5):
    '''
    '''
    med = median_filter(x, size=size)
    res = x - med
    _, _, stddev = sigma_clipped_stats(res, sigma=nsigma, maxiters=maxiters)
    mask = res < 5 * stddev
    return mask