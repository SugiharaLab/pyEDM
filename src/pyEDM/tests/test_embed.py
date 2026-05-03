"""
test_embed.py — EDM.Embed tests.
"""

import pyEDM as EDM
from conftest import EmbedArgs


def test_embed_univariate():
    '''Single column embed'''
    data = EDM.sampleData['circle']
    kwargs = EmbedArgs.copy()
    kwargs.update( dict( E       = 3,
                         tau     = -1,
                         columns = 'x' ) )
    EDM.Embed( data, **kwargs )


def test_embed_multivariate():
    '''Multi-column embed without time'''
    data = EDM.sampleData['circle']
    kwargs = EmbedArgs.copy()
    kwargs.update( dict( E           = 3,
                         tau         = -1,
                         columns     = ['x', 'y'],
                         includeTime = False ) )
    EDM.Embed( data, **kwargs )


def test_embed_include_time():
    '''Multi-column embed with includeTime = True'''
    data = EDM.sampleData['circle']
    kwargs = EmbedArgs.copy()
    kwargs.update( dict( E           = 3,
                         tau         = -1,
                         columns     = ['x', 'y'],
                         includeTime = True ) )
    EDM.Embed( data, **kwargs )


def test_embed_from_file():
    '''Embed reading from file path (must run from pyEDM tests/ directory)'''
    EDM.Embed( pathIn   = '../data/',
               dataFile = 'circle.csv',
               E        = 3,
               tau      = -1,
               columns  = ['x', 'y'] )
