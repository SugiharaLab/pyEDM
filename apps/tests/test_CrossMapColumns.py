'''Validation tests for pyEDM/apps/SMap_Tp'''
import pytest
from   pyEDM  import sampleData
from   pandas import concat

from conftest import CrossMap_ColumnsArgs, ValidData, GetMP_ContextName
from CrossMap_Columns import CrossMap_Columns

#------------------------------------------------------------
def test_CrossMap_Columns_1():
    '''Lorenz5D'''
    data   = sampleData['Lorenz5D'].copy()
    kwargs = CrossMap_ColumnsArgs.copy()
    kwargs.update( dict( target      = 'V1',
                         E           = 5,
                         Evec        = None,
                         Tp          = 1,
                         tau         = -3,
                         lib         = [1,500],
                         pred        = [601,900],
                         returnError = True,
                         mpMethod    = GetMP_ContextName() ) )

    D = CrossMap_Columns(data, **kwargs)

    # D.keys() : ['V1:V1', 'V2:V1', 'V3:V1', 'V4:V1', 'V5:V1']
    df = concat( [_ for _ in D.values()] )

    dfv = ValidData( 'CrossMap_Columns_1_Valid.csv', index_col = 0 )
    
    assert df.round(5).equals(dfv.round(5))
