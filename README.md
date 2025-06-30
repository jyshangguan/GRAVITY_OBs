# GRAVITY_OBs

Python package for preparing VLTI/GRAVITY observations and analyzing GRAVITY data.

## Overview

GRAVITY_OBs is a comprehensive Python package designed to help astronomers prepare for VLTI/GRAVITY observations. It provides tools for:

- **Observation Block (OB) Creation**: Automated generation of ESO Phase 2 observation blocks using the p2api
- **Coordinate Utilities**: Comprehensive coordinate conversion and proper motion calculations  
- **Catalog Integration**: Direct access to Gaia, 2MASS, and SIMBAD catalogs
- **Data Analysis**: Built-in tools for GRAVITY data analysis including astrometry and visibility processing
- **ASPRO Integration**: Create target lists compatible with ASPRO observation planning software

## Quick Start

### Installation

```bash
git clone https://github.com/user/GRAVITY_OBs.git
cd GRAVITY_OBs
pip install -e .
```

### Basic Usage

```python
import gravity_obs as go

# Calculate optimal DIT/NDIT for a source
dit, ndit = go.get_dit_med(8.5, tel='UT')

# Search for target information
target_info = go.search_simbad('HD 164492')

# Convert coordinates
ra_deg, dec_deg = go.coord_colon_to_degree('18:36:56.336', '-29:49:41.17')
```

### Creating Observation Blocks

```python
from gravity_obs import p2api_GRAVITY

# Initialize P2 API connection
api = p2api_GRAVITY('109.23CR.001', username, password)

# Create acquisition OB
api.add_GRAVITY_dual_offaxis_acq(
    name='HD164492_acq',
    folder_name='Calibrators',
    ft_name='HD164492',
    ft_kmag=8.5,
    sc_name='Science_Target',
    sc_kmag=12.0,
    sobj_x=100.0,  # mas offset
    sobj_y=50.0    # mas offset
)
```

## Documentation

View the complete documentation at: `docs/index.html`

The documentation includes:
- Detailed function references for all modules
- Interactive examples and tutorials
- Installation and setup guides
- API reference for P2 integration

## Modules

- **`gravity_obs.p2_tools`**: ESO Phase 2 API integration and OB creation
- **`gravity_obs.obs_utils`**: Coordinate utilities and catalog access  
- **`gravity_obs.aspro_utils`**: ASPRO compatibility tools
- **`gravity_obs.colab_utils`**: Google Colab integration
- **`gravity_obs.gravidata`**: GRAVITY data analysis tools

## Requirements

- Python 3.7+
- numpy, astropy, matplotlib
- astroquery (for catalog access)
- p2api (for ESO Phase 2 operations)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
