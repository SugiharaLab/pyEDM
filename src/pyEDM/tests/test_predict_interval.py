"""
test_predict_interval.py — EDM.PredictInterval tests.
"""

import pyEDM as EDM

from conftest import PredictIntervalArgs, ValidData, GetMP_ContextName


def test_predict_interval_block_3sp():
    '''PredictInterval on block_3sp x_t'''
    data = EDM.sampleData['block_3sp']
    kwargs = PredictIntervalArgs.copy()
    kwargs.update( dict( columns    = 'x_t',
                         target     = 'x_t',
                         maxTp      = 15,
                         lib        = [1, 150],
                         pred       = [151, 198],
                         E          = 3,
                         tau        = -1,
                         numProcess = 10,
                         mpMethod   = GetMP_ContextName() ) )  # Remove > 3.13
    df  = EDM.PredictInterval( data, **kwargs )
    dfv = round( ValidData( 'PredictInterval_valid.csv' ), 6 )

    assert dfv.equals( round( df, 6 ) )
