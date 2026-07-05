'''Validation tests for pyEDM/apps/CCM_Matrix'''
from itertools import combinations

import pytest
from   pandas import DataFrame
import numpy as np
from   pyEDM import CCM

from conftest import CCM_MatrixArgs, ValidData, LoadData, GetMP_ContextName
from CCM_Matrix import CCM_Matrix

#------------------------------------------------------------
def test_CCM_Matrix_1():
    '''Fly 20 subset E = 7'''
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

#------------------------------------------------------------
def test_CCM_Matrix_3( generate = False ):
    '''Compare pyEDM CCM with apps/CCM_Matrix on Fly 20 subset
       Differences due to different handling of seed'''

    data    = LoadData('Fly20_norm_1061.csv').copy()
    L       = [100,150,950,1000]
    E       = 7
    Tp      = 1
    tau     = -1
    sample  = 20
    seed    = 7777
    columns = data.columns[1:].to_list()

    kwargs = CCM_MatrixArgs.copy()
    kwargs.update( dict( E        = E,
                         libSizes = L,
                         Tp       = Tp,
                         tau      = tau,
                         sample   = sample,
                         seed     = seed,
                         mpMethod = GetMP_ContextName() ) )

    # CCM_Matrix ---------------------
    ccmMat = CCM_Matrix(data, **kwargs)

    D = ccmMat.Run()

    ccmMatTensor  = D['tensor']
    ccmMatSlope   = D['slope']
    ccmMatColumns = D['columns']

    if generate :
        # pyEDM CCM ---------------------------------------------------
        pairs     = list(combinations(columns, 2))
        col_pairs = [(a,b) for a, b in pairs]

        CCM_D = {}
        for column, target in col_pairs :
            CCM_D[ f'{column}:{target}' ] = CCM( data,
                                                 columns  = column,
                                                 target   = target,
                                                 libSizes = L,
                                                 E        = E,
                                                 Tp       = Tp,
                                                 tau      = tau,
                                                 sample   = sample,
                                                 seed     = seed,
                                                 mpMethod = GetMP_ContextName() )

        # Map of columns string to index in columns for ccmMatTensor lookup:
        #    i.e. ccmMatTensor[ col_i_D['TS45'], col_i_D['FWD'], :]
        col_i_D = dict( zip( columns, [columns.index(c) for c in columns] ) )

        # Tensor for direct CCM results
        ccmTensor = np.full( ccmMatTensor.shape, np.nan )

        # Map CCM_D DataFrame Series into ccmTensor
        for ccmKey in CCM_D.keys():
            col,tgt = ccmKey.split(':')[0], ccmKey.split(':')[1]

            # CCM DataFrame has columns: LibSize, col:tgt, tgt:col
            # LibSize maps to outer index of ccmTensor [:,:,L]
            ccmDF   = CCM_D[ccmKey]
            col_tgt = ccmDF[f'{col}:{tgt}'] # Series
            tgt_col = ccmDF[f'{tgt}:{col}'] # Series
            col_i   = col_i_D[ col ]
            tgt_i   = col_i_D[ tgt ]
            ccmTensor[ tgt_i, col_i, : ] = tgt_col.to_numpy()
            ccmTensor[ col_i, tgt_i, : ] = col_tgt.to_numpy()

        ccmTensor16 = ccmTensor.astype(np.float16)
    else :
        ccmTensor16 = ValidData('CCM_Matrix_3_Valid.npy') # float16

    valid = np.allclose( ccmMatTensor, ccmTensor16,
                         rtol = 0, atol = 0.08, equal_nan = True )
    assert valid
