import numpy as np
from astropy.table import Table

from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()

gc = gspread.authorize(creds)

__all__ = ['load_gspread_url']

def load_gspread_url(sheet_url, sheet_name=None, header_row=0, skip_rows=0):
    '''
    Load the Google Sheet to Astropy Table.
    '''
    sht = gc.open_by_url(sheet_url)
  
    if sheet_name is None:
        ws = sht.sheet1
    else:
        ws = sht.worksheet(sheet_name)
    
    data = ws.get_all_values()
    
    if header_row is None:
        names = None
    else:
        assert  header_row >= 0, f'header_row ({header_row}) must be >= 0!'
        names = data[header_row]
    
    data_row = header_row + 1 + skip_rows
    columns = list(np.array(data[data_row:]).T)
    tb = Table(columns, names=names)
    return tb

    



