"""
test_ccm.py — EDM.CCM tests.
"""

from numpy import nan
import pyEDM as EDM

from conftest import CCMArgs, ValidData, GetMP_ContextName


def test_ccm_anchovy_sst():
    '''sardine/anchovy/sst dataset'''
    data = EDM.sampleData['sardine_anchovy_sst']
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'anchovy',
                         target   = 'np_sst',
                         libSizes = [10, 20, 30, 40, 50, 60, 70, 75],
                         sample   = 100,
                         E        = 3,
                         Tp       = 0,
                         tau      = -1,
                         seed     = 777 ) )
    df  = EDM.CCM( data, **kwargs )
    dfv = round( ValidData( 'CCM_anch_sst_valid.csv' ), 2 )

    assert dfv.equals( round( df, 2 ) )


def test_ccm_multivariate_lorenz():
    '''Multivariate columns'''
    data = EDM.sampleData['Lorenz5D']
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'V3 V5',
                         target   = 'V1',
                         libSizes = [20, 200, 500, 950],
                         sample   = 30,
                         E        = 5,
                         Tp       = 10,
                         tau      = -5,
                         seed     = 777 ) )
    df  = EDM.CCM( data, **kwargs )
    dfv = round( ValidData( 'CCM_Lorenz5D_MV_valid.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )


def test_ccm_nan():
    '''nan in data'''
    data = EDM.sampleData['circle'].copy()
    data.iloc[ [5,  6, 12], 1 ] = nan
    data.iloc[ [10, 11, 17], 2 ] = nan

    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'x',
                         target   = 'y',
                         libSizes = [10, 190, 10],
                         sample   = 20,
                         E        = 2,
                         Tp       = 5,
                         tau      = -1,
                         seed     = 777 ) )
    df  = EDM.CCM( data, **kwargs )
    dfv = round( ValidData( 'CCM_nan_valid.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )


def test_ccm_multivariate_column_names_with_spaces():
    '''Multivariate column names containing spaces'''
    data = EDM.sampleData['columnNameSpace']
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = ['Var 1', 'Var3', 'Var 5 1'],
                         target   = ['Var 2', 'Var 4 A'],
                         libSizes = [20, 50, 90],
                         sample   = 1,
                         E        = 5,
                         Tp       = 0,
                         tau      = -1,
                         seed     = 777 ) )
    df  = EDM.CCM( data, **kwargs )
    dfv = round( ValidData( 'CCM_Lorenz5D_MV_Space_valid.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )


def test_ccm_negative_tp():
    '''Tp = -5'''
    data = EDM.sampleData['circle']
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'x',
                         target   = 'y',
                         libSizes = [20, 200, 50],
                         sample   = 10,
                         E        = 2,
                         Tp       = -5,
                         tau      = -1,
                         seed     = 777 ) )
    df  = EDM.CCM( data, **kwargs )
    dfv = round( ValidData( 'CCM_NegativeTp.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )


def test_ccm_exclusion_radius():
    '''exclusionRadius = 5'''
    data = EDM.sampleData['circle']
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns         = 'x',
                         target          = 'y',
                         libSizes        = [20, 200, 30],
                         sample          = 10,
                         E               = 2,
                         Tp              = 3,
                         tau             = -1,
                         exclusionRadius = 5,
                         seed            = 777 ) )
    df  = EDM.CCM( data, **kwargs )
    dfv = round( ValidData( 'CCM_exclusionRadius.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )


def test_ccm_positive_tau():
    '''tau = +3 (positive)'''
    data = EDM.sampleData['circle']
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'x',
                         target   = 'y',
                         libSizes = [20, 200, 30],
                         sample   = 10,
                         E        = 2,
                         Tp       = 0,
                         tau      = 3,
                         seed     = 777 ) )
    df  = EDM.CCM( data, **kwargs )
    dfv = round( ValidData( 'CCM_positiveTau.csv' ), 4 )

    assert dfv.equals( round( df, 4 ) )


def test_ccm_legacy():
    '''CCM legacy'''
    data = EDM.sampleData["block_3sp"]
    kwargs = CCMArgs.copy()
    kwargs.update( dict( columns  = 'x_t',
                         target   = 'z_t',
                         libSizes = [20, 190, 10],
                         sample   = 100,
                         E        = 5,
                         mpMethod = GetMP_ContextName(), # Remove python > 3.13
                         seed     = 777 ) )

    df  = EDM.CCM(data, **kwargs)
    kwargs.update(legacy=True)
    dfL = EDM.CCM(data, **kwargs)

    ccm  = round( df.iloc[:,1:], 2 )
    ccmL = round( df.iloc[:,1:], 2 )
    assert ccm.equals( ccmL )
