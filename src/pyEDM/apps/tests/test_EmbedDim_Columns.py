'''Validation tests for pyEDM/apps/EmbedDim_Columns'''
import pytest
from   pyEDM import sampleData

from conftest import EmbedDim_ColumnsArgs, ValidData, LoadData
from conftest import GetMP_ContextName
from EmbedDim_Columns import EmbedDim_Columns

#------------------------------------------------------------
def test_EmbedDim_Columns_1():
    '''Lorenz5D'''
    data   = LoadData('Fly20_norm_1061.csv').copy()
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
    
    dfv = ValidData("EmbedDim_Columns_1_Fly_Valid.csv")

    assert df.round(5).equals( dfv.round(5) )
