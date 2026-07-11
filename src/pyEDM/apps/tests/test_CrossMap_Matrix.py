'''Validation tests for pyEDM/apps/SMap_Tp'''
import pytest

from conftest import CrossMap_MatrixArgs, ValidData, LoadData
from conftest import GetMP_ContextName 
from CrossMap_Matrix import CrossMap_Matrix

#------------------------------------------------------------
def test_CrossMap_Matrix_1():
    '''Fly 20 subset'''
    data   = LoadData('Fly20_norm_1061.csv').copy()
    kwargs = CrossMap_MatrixArgs.copy()
    kwargs.update( dict( E           = 7,
                         Tp          = 1,
                         tau         = -3,
                         lib         = [1,1061],
                         pred        = [1,1061],
                         threshold   = 0.1,
                         returnValue = 'dataframe',  # or 'matrix'
                         mpMethod    = GetMP_ContextName() ) )

    df = CrossMap_Matrix(data, **kwargs)

    dfv = ValidData('CrossMap_Matrix_1_Valid.feather')

    assert df.round(5).equals( dfv.round(5) )
