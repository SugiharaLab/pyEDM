"""
test_predict_exclusion_radius.py — EDM.PredictExclusionRadius tests.
"""

import pyEDM as EDM

from conftest import PredictExclusionRadiusArgs, ValidData, GetMP_ContextName


def test_predict_exclusion_radius_sum_flow():
    '''PredictExclusionRadius on Everglades flow'''
    data = EDM.sampleData['SumFlow_1980-2005']
    kwargs = PredictExclusionRadiusArgs.copy()
    kwargs.update( dict( columns    = 'S12.C.D.S333',
                         target     = 'S12.C.D.S333',
                         lib        = [1, 900],
                         pred       = [1, 900],
                         E          = 7,
                         tau        = -3,
                         Tp         = 0,
                         numProcess = 10,
                         mpMethod   = GetMP_ContextName(),
                         showPlot   = False ) )  # Remove > 3.13
    df  = EDM.PredictExclusionRadius( data, **kwargs )
    dfv = round( ValidData( 'PredictExclusionRadius_valid.csv' ), 6 )

    assert dfv.equals( round( df, 6 ) )
