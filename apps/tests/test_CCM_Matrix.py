'''Validation tests for pyEDM/apps/CCM_Matrix'''
import pytest

from pandas import DataFrame

from conftest import CCM_MatrixArgs, ValidData, pyEDM_FlyData, GetMP_ContextName
from CCM_Matrix import CCM_Matrix

#------------------------------------------------------------
def test_CCM_Matrix_1():
    '''Fly 20 subset'''
    data    = pyEDM_FlyData().copy()
    kwargs  = CCM_MatrixArgs.copy()
    kwargs.update( dict( E        = 7,
                         seed     = 7777,
                         mpMethod = GetMP_ContextName() ) )

    ccm_    = CCM_Matrix(data, **kwargs)
    D       = ccm_.Run()
    tensor  = D['tensor']
    columns = D['columns']
    
    ccmL4 = tensor[:,:,3] # CCM matrix at libSize[3] : float16
    df    = DataFrame( ccmL4, columns = columns, index = columns )

    dfv = ValidData('CCM_Matrix_1_Valid.feather') # float16

    assert df.equals(dfv)
