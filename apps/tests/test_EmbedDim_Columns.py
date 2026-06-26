'''Validation tests for pyEDM/apps/EmbedDim_Columns'''
import pytest
from   pyEDM import sampleData

from conftest import EmbedDim_ColumnsArgs, ValidData, pyEDM_FlyData
from conftest import GetMP_ContextName
from EmbedDim_Columns import EmbedDim_Columns

#------------------------------------------------------------
def test_EmbedDim_Columns_1():
    '''Lorenz5D'''
    data   = pyEDM_FlyData().copy()
    kwargs = EmbedDim_ColumnsArgs.copy()
    kwargs.update( dict( target   = 'FWD',
                         lib      = [1,300],
                         pred     = [301,600],
                         firstMax = True,
                         mpMethod = GetMP_ContextName() ) )

    df = EmbedDim_Columns(data, **kwargs)

    # EmbedDim_Columns() returns E uint16, rho float32
    df['E']   = df['E'].astype( int )
    df['rho'] = df['rho'].astype( float )
    
    dfv = ValidData("EmbedDim_Columns_Lorenz5D_1_Valid.csv")

    assert df.round(5).equals( dfv.round(5) )
