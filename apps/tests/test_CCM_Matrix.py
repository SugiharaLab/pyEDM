'''Validation tests for pyEDM/apps/CCM_Matrix'''
import pytest

from pandas import DataFrame

from conftest import CCM_MatrixArgs, ValidData, LoadData, GetMP_ContextName
from CCM_Matrix import CCM_Matrix

#------------------------------------------------------------
def test_CCM_Matrix_1():
    '''Fly 20 subset'''
    data   = LoadData('Fly20_norm_1061.csv').copy()
    kwargs = CCM_MatrixArgs.copy()
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
    
#------------------------------------------------------------
def test_CCM_Matrix_2():
    '''Fly 20 subset with EDim'''
    data   = LoadData('Fly20_norm_1061.csv').copy()
    EDim   = LoadData('EDim_FWD_Fly80XY.csv').copy()
    kwargs = CCM_MatrixArgs.copy()
    kwargs.update( dict( E        = EDim['E'],
                         seed     = 7777,
                         mpMethod = GetMP_ContextName() ) )

    ccm_    = CCM_Matrix(data, **kwargs)
    D       = ccm_.Run()
    tensor  = D['tensor']
    columns = D['columns']
    
    ccmL4 = tensor[:,:,3] # CCM matrix at libSize[3] : float16
    df    = DataFrame( ccmL4, columns = columns, index = columns )

    dfv = ValidData('CCM_Matrix_2_Valid.feather') # float16

    assert df.equals(dfv)
