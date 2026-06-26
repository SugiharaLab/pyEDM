'''Validation tests for pyEDM/apps/SMap_Tp'''
import pytest
from   pyEDM import sampleData

from conftest import SMap_TpArgs, ValidData, GetMP_ContextName
from SMap_Tp  import SMap_Tp

#------------------------------------------------------------
def test_SMap_Tp_1():
    '''Lorenz5D'''
    data   = sampleData['Lorenz5D'].copy()
    kwargs = SMap_TpArgs.copy()
    TpList = [0,1,2,3,4,5,7,8,9,10]
    kwargs.update( dict( TpList = TpList,
                         target = 'V1',
                         column = 'V3',
                         E      = 5,
                         tau    = -1,
                         theta  = 3.3,
                         lib    = [1,500],
                         pred   = [701,900],
                         mpMethod = GetMP_ContextName() ) )

    D = SMap_Tp(data, **kwargs)

    # SMap_theta() returns dict of dicts:
    # D.keys() = ['theta_0.01', 'theta_0.05',...]
    # D['theta_0.01'].keys() = ['predictions', 'coefficients', 'singularValues']
    keys_ = D[f'Tp{TpList[0]}'].keys()
    
    assert ['predictions', 'coefficients', 'singularValues'] == list(keys_)
