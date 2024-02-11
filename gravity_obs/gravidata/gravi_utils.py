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