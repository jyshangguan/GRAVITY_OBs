import numpy as np
from astropy.table import Table

try:
    from google.colab import auth
    auth.authenticate_user()
    
    from google.auth import default
    creds, _ = default()
    
    import gspread
    gc = gspread.authorize(creds)
except:
    gc = None

__all__ = ['load_gspread_url']

def load_gspread_url(sheet_url, sheet_name=None, header_row=0, skip_rows=0):
    '''
    Load the Google Sheet to Astropy Table.

    Parameters
    ----------
    sheet_url : str
        The URL of the Google Sheet.
    sheet_name : str, optional
        The name of the worksheet. The default is None and the first worksheet 
        is used.
    header_row : int, optional
        The row number of the header. The default is 0.
    skip_rows : int, optional
        The number of rows to skip after the header. The default is 0.
    
    Returns
    -------
    tb : astropy.table.Table
        The table loaded from the Google Sheet.
    '''
    if gc is None:
        raise ImportError('Not in Google Colab!')

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

    



