"""
test_predict_nonlinear.py — EDM.PredictNonlinear tests.
"""

import pyEDM as EDM

from conftest import PredictNonlinearArgs, ValidData, GetMP_ContextName


def test_predict_nonlinear_tentmap():
    '''PredictNonlinear on TentMapNoise'''
    data = EDM.sampleData['TentMapNoise']
    kwargs = PredictNonlinearArgs.copy()
    kwargs.update( dict( columns    = 'TentMap',
                         target     = 'TentMap',
                         lib        = [1, 500],
                         pred       = [501, 800],
                         E          = 4,
                         Tp         = 1,
                         tau        = -1,
                         numProcess = 10,
                         theta      = [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5,
                                       2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                         mpMethod   = GetMP_ContextName() ) )  # Remove > 3.13
    df  = EDM.PredictNonlinear( data, **kwargs )
    dfv = round( ValidData( 'PredictNonlinear_valid.csv' ), 6 )

    assert dfv.equals( round( df, 6 ) )
