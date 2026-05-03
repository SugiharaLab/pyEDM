"""
test_smap.py — EDM.SMap tests.
"""

from numpy import nan
import pyEDM as EDM

from conftest import SMapArgs, ValidData


def test_smap_circle_E4():
    '''circle data, E = 4, theta = 3'''
    data = EDM.sampleData['circle']
    kwargs = SMapArgs.copy()
    kwargs.update( dict( columns = 'x',
                         target  = 'x',
                         lib     = [1, 100],
                         pred    = [110, 160],
                         E       = 4,
                         Tp      = 1,
                         tau     = -1,
                         theta   = 3. ) )
    S   = EDM.SMap( data, **kwargs )
    dfv = ValidData( 'SMap_circle_E4_valid.csv' )
    df  = S['predictions']

    S1 = round( dfv.get('Predictions')[1:50], 6 )
    S2 = round(  df.get('Predictions')[1:50], 6 )
    assert S1.equals( S2 )


def test_smap_embedded_circle_E2():
    '''embedded = True; validate predictions and coefficients'''
    data = EDM.sampleData['circle']
    kwargs = SMapArgs.copy()
    kwargs.update( dict( columns  = ['x', 'y'],
                         target   = 'x',
                         lib      = [1, 200],
                         pred     = [1, 200],
                         E        = 2,
                         Tp       = 1,
                         tau      = -1,
                         embedded = True,
                         theta    = 3. ) )
    S   = EDM.SMap( data, **kwargs )
    dfv = ValidData( 'SMap_circle_E2_embd_valid.csv' )
    df  = S['predictions']
    dfc = S['coefficients']

    S1 = round( dfv.get('Predictions')[1:195], 6 )
    S2 = round(  df.get('Predictions')[1:195], 6 )
    assert S1.equals( S2 )

    assert dfc['∂x/∂x'].mean().round(5) == 0.99801
    assert dfc['∂x/∂y'].mean().round(5) == 0.06311


def test_smap_nan():
    '''nan in data'''
    data = EDM.sampleData['circle'].copy()
    data.iloc[ [5,  6, 12], 1 ] = nan
    data.iloc[ [10, 11, 17], 2 ] = nan

    kwargs = SMapArgs.copy()
    kwargs.update( dict( columns = 'x',
                         target  = 'y',
                         lib     = [1, 50],
                         pred    = [1, 50],
                         E       = 2,
                         Tp      = 1,
                         tau     = -1,
                         theta   = 3. ) )
    S   = EDM.SMap( data, **kwargs )
    dfv = ValidData( 'SMap_nan_valid.csv' )
    df  = S['predictions']

    S1 = round( dfv.get('Predictions')[1:50], 6 )
    S2 = round(  df.get('Predictions')[1:50], 6 )
    assert S1.equals( S2 )


def test_smap_no_time():
    '''noTime = True (circle_noTime dataset)'''
    data = EDM.sampleData['circle_noTime']
    kwargs = SMapArgs.copy()
    kwargs.update( dict( columns = 'x',
                         target  = 'y',
                         lib     = [1, 100],
                         pred    = [101, 150],
                         E       = 2,
                         theta   = 3,
                         noTime  = True ) )
    S   = EDM.SMap( data, **kwargs )
    dfv = ValidData( 'SMap_noTime_valid.csv' )
    df  = S['predictions']

    S1 = round( dfv.get('Predictions')[1:50], 6 )
    S2 = round(  df.get('Predictions')[1:50], 6 )
    assert S1.equals( S2 )
