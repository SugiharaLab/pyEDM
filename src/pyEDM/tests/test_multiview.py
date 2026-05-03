"""
test_multiview.py — EDM.Multiview tests.
"""

import pyEDM as EDM

from conftest import MultiviewArgs, ValidData, GetMP_ContextName


def test_multiview_predictions_and_combos():
    '''Validate both the averaged predictions and the combination rankings'''
    data = EDM.sampleData['block_3sp']
    kwargs = MultiviewArgs.copy()
    kwargs.update( dict( columns         = 'x_t y_t z_t',
                         target          = 'x_t',
                         lib             = [1, 100],
                         pred            = [101, 198],
                         D               = 0,
                         E               = 3,
                         Tp              = 1,
                         knn             = 0,
                         tau             = -1,
                         multiview       = 0,
                         exclusionRadius = 0,
                         trainLib        = False,
                         excludeTarget   = False,
                         numProcess      = 4,
                         mpMethod        = GetMP_ContextName() ) )  # Remove > 3.13
    M = EDM.Multiview( data, **kwargs )

    pred  = round( M['Predictions'].get('Predictions'), 4 )
    predv = round( ValidData('Multiview_pred_valid.csv').get('Predictions'), 4 )
    assert predv.equals( pred )

    combo_cols = ['rho', 'MAE', 'RMSE']
    dfvc = round( ValidData('Multiview_combos_valid.csv'), 4 )[ combo_cols ]
    dfmc = round( M['View'][ combo_cols ], 4 )
    assert dfvc.equals( dfmc )
