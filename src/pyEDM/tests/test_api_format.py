"""
test_api_format.py — Simplex argument-format acceptance tests.

These tests verify that the API accepts the documented input variants
(string vs list columns/lib/pred, positional args, column names with spaces).
They do not validate numeric output.
"""

import pyEDM as EDM
from conftest import SimplexArgs


def test_api_string_lib_pred():
    '''lib and pred as space-delimited strings'''
    data = EDM.sampleData['Lorenz5D']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = 'V1',
                         target  = 'V5',
                         lib     = '1 300',
                         pred    = '301 310',
                         E       = 5 ) )
    EDM.Simplex( data, **kwargs )


def test_api_list_single_column():
    '''columns as single-element list'''
    data = EDM.sampleData['Lorenz5D']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = ['V1'],
                         target  = 'V5',
                         lib     = [1, 300],
                         pred    = [301, 310],
                         E       = 5 ) )
    EDM.Simplex( data, **kwargs )


def test_api_list_multi_column():
    '''columns as multi-element list'''
    data = EDM.sampleData['Lorenz5D']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = ['V1', 'V3'],
                         target  = 'V5',
                         lib     = [1, 300],
                         pred    = [301, 310],
                         E       = 5 ) )
    EDM.Simplex( data, **kwargs )


def test_api_list_multi_target():
    '''target as list'''
    data = EDM.sampleData['Lorenz5D']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = ['V1', 'V3'],
                         target  = ['V5', 'V2'],
                         lib     = [1, 300],
                         pred    = [301, 310],
                         E       = 5 ) )
    EDM.Simplex( data, **kwargs )


def test_api_knn_zero():
    '''explicit knn = 0 (automatic)'''
    data = EDM.sampleData['Lorenz5D']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = 'V1',
                         target  = 'V5',
                         lib     = [1, 300],
                         pred    = [301, 310],
                         E       = 5,
                         knn     = 0 ) )
    EDM.Simplex( data, **kwargs )


def test_api_negative_tau():
    '''tau = -2'''
    data = EDM.sampleData['Lorenz5D']
    kwargs = SimplexArgs.copy()
    kwargs.update( dict( columns = 'V1',
                         target  = 'V5',
                         lib     = [1, 300],
                         pred    = [301, 310],
                         E       = 5,
                         tau     = -2 ) )
    EDM.Simplex( data, **kwargs )


def test_api_column_names_with_spaces():
    '''Column names containing spaces, all-positional call'''
    data = EDM.sampleData['columnNameSpace']
    EDM.Simplex( data, ['Var 1', 'Var 2'], ['Var 5 1'],
                 [1, 80], [81, 85], 5, 1, 0, -1, 0,
                 False, [], False, 0, False, False, False )
