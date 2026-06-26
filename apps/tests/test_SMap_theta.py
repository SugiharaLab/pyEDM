'''Validation tests for pyEDM/apps/SMap_theta'''
import pytest
from   pyEDM import sampleData

from conftest   import SMap_thetaArgs, ValidData, GetMP_ContextName
from SMap_theta import SMap_theta

#------------------------------------------------------------
def test_SMap_theta_1():
    '''Lorenz5D'''
    data   = sampleData['Lorenz5D'].copy()
    kwargs = SMap_thetaArgs.copy()
    thetaList = [0.01,0.05,0.08,0.1,0.5,0.75,1,2,3,4,5,7,8]
    kwargs.update( dict( thetaList = thetaList,
                         target    = 'V1',
                         column    = 'V3',
                         E         = 5,
                         tau       = -1,
                         Tp        = 5,
                         lib       = [1,500],
                         pred      = [701,900],
                         mpMethod  = GetMP_ContextName() ) )

    D = SMap_theta(data, **kwargs)

    # SMap_theta() returns dict of dicts:
    # D.keys() = ['theta_0.01', 'theta_0.05',...]
    # D['theta_0.01'].keys() = ['predictions', 'coefficients', 'singularValues']
    keys_ = D[f'theta_{thetaList[0]}'].keys()
    
    assert ['predictions', 'coefficients', 'singularValues'] == list(keys_)
