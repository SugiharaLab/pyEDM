"""
test_embed_dimension.py — EDM.EmbedDimension tests.
"""

import pyEDM as EDM

from conftest import EmbedDimensionArgs, ValidData, GetMP_ContextName


def test_embed_dimension_lorenz():
    '''EmbedDimension on Lorenz5D V1'''
    data = EDM.sampleData['Lorenz5D']
    kwargs = EmbedDimensionArgs.copy()
    kwargs.update( dict( columns         = 'V1',
                         target          = 'V1',
                         maxE            = 12,
                         lib             = [1, 500],
                         pred            = [501, 800],
                         Tp              = 15,
                         tau             = -5,
                         exclusionRadius = 20,
                         numProcess      = 10,
                         mpMethod        = GetMP_ContextName() ) )  # Remove > 3.13
    df  = EDM.EmbedDimension( data, **kwargs )
    dfv = round( ValidData( 'EmbedDim_valid.csv' ), 6 )

    assert dfv.equals( round( df, 6 ) )
