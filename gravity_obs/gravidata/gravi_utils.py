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


def set_docstring(doc_template, data_name):
    def decorator(func):
        func.__doc__ = doc_template.format(data_name=data_name, data_name_lower=data_name.lower())
        return func
    return decorator


doc_plot = '''
Plot the {data_name} as a function of the uv distance.

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
    If True, the average {data_name} will be shown. Default is False.
ax : matplotlib axis, optional
    The axis to plot the {data_name} as a function of the uv distance. If None, 
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
    The axis of the {data_name} as a function of the uv distance.
'''

doc_get_vis = '''
Get the {data_name} and {data_name}ERR of the OIFITS HDU.

Parameters
----------
fiber : str
    The fiber type. Either SC or FT.
polarization : int, optional
    The polarization. If None, the polarization 0 and 1 will be used for
    COMBINED and SPLIT, respectively.
        
Returns
-------
{data_name_lower}, {data_name_lower}err : masked arrays
    The {data_name} and {data_name} of the OIFITS HDU. The shape is (N_BASELINE, N_CHANNEL).
'''