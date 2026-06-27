'''Validation tests for pyEDM/apps/Embedding'''
import pytest
from   pyEDM import sampleData

from conftest  import EmbeddingArgs, ValidData
from Embedding import Embedding

#------------------------------------------------------------
def test_Embedding_1():
    '''Lorenz5D'''
    data   = sampleData["Lorenz5D"].copy()
    kwargs = EmbeddingArgs.copy()
    kwargs.update( dict( columns = ['V1','V3'],
                         E       = 5,
                         tau     = -3 ) )

    df = Embedding(data, **kwargs)

    dfv = ValidData("Embedding_Lorenz5D_1_Valid.csv")

    assert df.equals( dfv )
