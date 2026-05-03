"""
test_simplex.py — EDM.Simplex tests.
"""

from datetime import datetime

from numpy  import array, array_equal
from pandas import DataFrame
from numpy  import nan
import pyEDM as EDM

from conftest import SimplexArgs, ValidData


def test_simplex_E3_block_3sp():
    '''embedded = False, E = 3'''
    data = EDM.sampleData['block_3sp']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = 'x_t',
                         target  = 'x_t',
                         lib     = [1, 100],
                         pred    = [101, 195],
                         E       = 3 ) )
    df  = EDM.Simplex( data, **kwargs )
    dfv = ValidData( 'Smplx_E3_block_3sp_valid.csv' )

    S1 = round( dfv.get('Predictions')[1:95], 6 )
    S2 = round(  df.get('Predictions')[1:95], 6 )
    assert S1.equals( S2 )


def test_simplex_embedded_block_3sp():
    '''embedded = True, E = 3'''
    data = EDM.sampleData['block_3sp']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns  = 'x_t y_t z_t',
                         target   = 'x_t',
                         lib      = [1, 99],
                         pred     = [100, 198],
                         E        = 3,
                         embedded = True ) )
    df  = EDM.Simplex( data, **kwargs )
    dfv = ValidData( 'Smplx_E3_embd_block_3sp_valid.csv' )

    S1 = round( dfv.get('Predictions')[1:98], 6 )
    S2 = round(  df.get('Predictions')[1:98], 6 )
    assert S1.equals( S2 )


def test_simplex_negative_tp():
    '''Tp = -2'''
    data = EDM.sampleData['block_3sp']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = 'x_t',
                         target  = 'y_t',
                         lib     = [1, 100],
                         pred    = [50, 80],
                         E       = 3,
                         Tp      = -2 ) )
    df  = EDM.Simplex( data, **kwargs )
    dfv = ValidData( 'Smplx_negTp_block_3sp_valid.csv' )

    S1 = round( dfv.get('Predictions')[1:98], 6 )
    S2 = round(  df.get('Predictions')[1:98], 6 )
    assert S1.equals( S2 )


def test_simplex_valid_lib():
    '''validLib boolean mask'''
    data = EDM.sampleData['circle']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns  = 'x',
                         target   = 'x',
                         lib      = [1, 200],
                         pred     = [1, 200],
                         E        = 2,
                         Tp       = 1,
                         validLib = data.eval('x > 0.5 | x < -0.5') ) )
    df  = EDM.Simplex( data, **kwargs )
    dfv = ValidData( 'Smplx_validLib_valid.csv' )

    S1 = round( dfv.get('Predictions')[1:195], 6 )
    S2 = round(  df.get('Predictions')[1:195], 6 )
    assert S1.equals( S2 )


def test_simplex_disjoint_lib():
    '''Disjoint lib segments'''
    data = EDM.sampleData['circle']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = 'x',
                         target  = 'x',
                         lib     = [1, 40, 50, 130],
                         pred    = [80, 170],
                         E       = 2,
                         Tp      = 1,
                         tau     = -3 ) )
    df  = EDM.Simplex( data, **kwargs )
    dfv = ValidData( 'Smplx_disjointLib_valid.csv' )

    S1 = round( dfv.get('Predictions')[1:195], 6 )
    S2 = round(  df.get('Predictions')[1:195], 6 )
    assert S1.equals( S2 )


def test_simplex_disjoint_pred_nan():
    '''Disjoint pred segments with nan in data'''
    data = EDM.sampleData['Lorenz5D'].copy()
    data.iloc[ [8, 50, 501], [1, 2] ] = nan

    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = 'V1',
                         target  = 'V2',
                         E       = 5,
                         Tp      = 2,
                         lib     = [1, 50, 101, 200, 251, 500],
                         pred    = [1, 10, 151, 155, 551, 555, 881, 885, 991, 1000] ) )
    df  = EDM.Simplex( data, **kwargs )
    dfv = ValidData( 'Smplx_disjointPred_nan_valid.csv' )

    S1 = round( dfv.get('Predictions')[1:195], 5 )
    S2 = round(  df.get('Predictions')[1:195], 5 )
    assert S1.equals( S2 )


def test_simplex_exclusion_radius():
    '''exclusionRadius = 5'''
    data = EDM.sampleData['circle']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns         = 'x',
                         target          = 'y',
                         lib             = [1, 100],
                         pred            = [21, 81],
                         E               = 2,
                         Tp              = 1,
                         exclusionRadius = 5 ) )
    df  = EDM.Simplex( data, **kwargs )
    dfv = ValidData( 'Smplx_exclRadius_valid.csv' )

    S1 = round( dfv.get('Predictions')[1:60], 6 )
    S2 = round(  df.get('Predictions')[1:60], 6 )
    assert S1.equals( S2 )


def test_simplex_nan_x_to_y():
    '''nan in x column, predicting y'''
    data = EDM.sampleData['circle'].copy()
    data.iloc[ [5,  6, 12], 1 ] = nan
    data.iloc[ [10, 11, 17], 2 ] = nan

    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = 'x',
                         target  = 'y',
                         lib     = [1, 100],
                         pred    = [1, 95],
                         E       = 2,
                         Tp      = 1 ) )
    df  = EDM.Simplex( data, **kwargs )
    dfv = ValidData( 'Smplx_nan_valid.csv' )

    S1 = round( dfv.get('Predictions')[1:90], 6 )
    S2 = round(  df.get('Predictions')[1:90], 6 )
    assert S1.equals( S2 )


def test_simplex_nan_y_to_x():
    '''nan in y column, predicting x'''
    data = EDM.sampleData['circle'].copy()
    data.iloc[ [5,  6, 12], 1 ] = nan
    data.iloc[ [10, 11, 17], 2 ] = nan

    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = 'y',
                         target  = 'x',
                         lib     = [1, 200],
                         pred    = [1, 195],
                         E       = 2,
                         Tp      = 1 ) )
    df  = EDM.Simplex( data, **kwargs )
    dfv = ValidData( 'Smplx_nan2_valid.csv' )

    S1 = round( dfv.get('Predictions')[1:190], 6 )
    S2 = round(  df.get('Predictions')[1:190], 6 )
    assert S1.equals( S2 )


def test_simplex_datetime_index():
    '''Time column contains datetime objects'''
    data = EDM.sampleData['SumFlow_1980-2005']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = 'S12.C.D.S333',
                         target  = 'S12.C.D.S333',
                         lib     = [1, 800],
                         pred    = [801, 1001],
                         E       = 3,
                         Tp      = 1 ) )
    df  = EDM.Simplex( data, **kwargs )

    assert isinstance( df['Time'][0], datetime )

    dfv = ValidData( 'Smplx_DateTime_valid.csv' )
    S1 = round( dfv.get('Predictions')[1:200], 6 )
    S2 = round(  df.get('Predictions')[1:200], 6 )
    assert S1.equals( S2 )


def test_simplex_knn1_neighbors():
    '''knn = 1: verify knn_neighbors array via returnObject'''
    data = EDM.sampleData['Lorenz5D']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns      = 'V5',
                         target       = 'V5',
                         lib          = [301, 400],
                         pred         = [350, 355],
                         knn          = 1,
                         embedded     = True,
                         returnObject = True ) )
    S = EDM.Simplex( data, **kwargs )

    knnValid = array( [322, 334, 362, 387, 356, 355] )[:, None]
    assert array_equal( S.knn_neighbors, knnValid )


def test_simplex_exclusion_radius_knn():
    '''exclusionRadius forces knn_neighbors into expected range'''
    data = EDM.sampleData['Lorenz5D']
    x    = [i + 1 for i in range(1000)]
    data = DataFrame( {'Time': data['Time'], 'X': x, 'V1': data['V1']} )

    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns         = 'X',
                         target          = 'V1',
                         lib             = [1, 100],
                         pred            = [101, 110],
                         E               = 5,
                         exclusionRadius = 10,
                         returnObject    = True ) )
    S = EDM.Simplex( data, **kwargs )

    knnValid = array( [89, 90, 91, 92, 93, 94, 95, 96, 97, 98] )
    assert array_equal( S.knn_neighbors[:, 0], knnValid )
