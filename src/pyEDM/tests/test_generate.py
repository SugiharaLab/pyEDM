"""
test_generate.py — Generative mode tests for EDM.Simplex and EDM.SMap.

generateSteps causes the EDM function to iteratively extend predictions
beyond the pred range. generateConcat controls whether the generated
steps are concatenated to the original pred output.
"""

import pyEDM as EDM

from conftest import SimplexArgs, SMapArgs


def test_generate_simplex_concat():
    '''generateSteps = 100, generateConcat = True: output is pred + generated rows'''
    data = EDM.sampleData['circle']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns        = 'x',
                         target         = 'x',
                         lib            = [1, 200],
                         pred           = [1, 2],
                         E              = 2,
                         generateSteps  = 100,
                         generateConcat = True ) )
    df = EDM.Simplex( data, **kwargs )
    assert df.shape == (300, 4)


def test_generate_simplex_no_concat():
    '''generateSteps = 100, generateConcat = False: output contains only generated rows'''
    data = EDM.sampleData['Lorenz5D']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns        = 'V1',
                         target         = 'V1',
                         lib            = [1, 1000],
                         pred           = [1, 2],
                         E              = 5,
                         generateSteps  = 100,
                         generateConcat = False ) )
    df = EDM.Simplex( data, **kwargs )
    assert df.shape == (100, 4)


def test_generate_smap_concat():
    '''SMap generateSteps = 100, generateConcat = True: output is pred + generated rows'''
    data = EDM.sampleData['circle']
    kwargs = SMapArgs.copy()
    kwargs.update( dict( columns        = 'x',
                         target         = 'x',
                         theta          = 3.,
                         lib            = [1, 200],
                         pred           = [1, 2],
                         E              = 2,
                         generateSteps  = 100,
                         generateConcat = True ) )
    S = EDM.SMap( data, **kwargs )
    assert S['predictions'].shape == (300, 4)
