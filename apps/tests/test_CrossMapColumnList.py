'''Validation tests for pyEDM/apps/SMap_Tp'''
import pytest
from   pyEDM  import sampleData

from conftest import CrossMap_ColumnListArgs, ValidData, GetMP_ContextName
from CrossMap_ColumnList import CrossMap_ColumnList

#------------------------------------------------------------
def test_CrossMap_ColumnList_1():
    '''Lorenz5D'''
    data    = sampleData['Lorenz5D'].copy()
    kwargs  = CrossMap_ColumnListArgs.copy()
    columns = ['V1','V2']
    target  = 'V5'
    kwargs.update( dict( columns = columns,
                         target  = target,
                         E       = 5,
                         Tp      = 1,
                         tau     = -3,
                         lib     = [1,500],
                         pred    = [601,900],
                         mpMethod = GetMP_ContextName() ) )

    D = CrossMap_ColumnList(data, **kwargs)

    # D is dict of Simplex() DataFrame
    df = D[ f'{columns[0]}:{target}' ]

    assert df.columns.to_list() == \
        ['Time','Observations','Predictions','Pred_Variance']
    
#------------------------------------------------------------
def test_CrossMap_ColumnList_2():
    '''Lorenz5D'''
    data    = sampleData['Lorenz5D'].copy()
    kwargs  = CrossMap_ColumnListArgs.copy()
    columns = [['V1','V2'],['V3','V4']]
    target  = 'V5'
    kwargs.update( dict( columns = columns,
                         target  = target,
                         E       = 5,
                         Tp      = 1,
                         tau     = -3,
                         lib     = [1,500],
                         pred    = [601,900],
                         mpMethod = GetMP_ContextName() ) )

    D = CrossMap_ColumnList(data, **kwargs)

    # D is dict of Simplex() DataFrame
    df = D[ f'{columns[0]}:{target}' ]

    assert df.columns.to_list() == \
        ['Time','Observations','Predictions','Pred_Variance']
