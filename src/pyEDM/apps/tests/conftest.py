"""
conftest.py — shared resources for pyEDM/apps pytest suite.

pytest loads this file automatically before any test collection.

Contents:
  - GetMP_ContextName()  multiprocessing context helper
  - ValidData()          load a validation CSV by filename
  - *Args dicts          default keyword arguments for each EDM API function
"""

import os, sys
from importlib import resources
from multiprocessing import get_context, get_start_method

from pandas import read_csv, read_feather
from numpy import load

# Monkeypatch sys.path to import non-package apps in ../
sys.path.append('../')

# ---------------------------------------------------------------------------
# Multiprocessing context helper  (remove when > Python 3.13)
# ---------------------------------------------------------------------------

def GetMP_ContextName():
    '''Until > Python 3.14, disallow "fork" multiprocessing context.'''
    allowedContext = ("forkserver", "spawn")
    current = get_start_method( allow_none = True )
    if current in allowedContext:
        return get_context( current )._name
    for method in allowedContext:
        try:
            return get_context( method )._name
        except ValueError:
            continue

# ---------------------------------------------------------------------------
# Validation file helper
# ---------------------------------------------------------------------------

VALID_DIR = os.path.join( os.path.dirname(os.path.abspath(__file__)),
                          "ValidOutput" )

def ValidData( filename, index_col = None ):
    '''Return validation DataFrame from ValidOutput/ directory.'''
    if '.csv' in filename[-4:] :
        df = read_csv( os.path.join(VALID_DIR, filename), index_col = index_col )
    elif '.feather' in filename[-8:] :
        df = read_feather( os.path.join(VALID_DIR, filename) )
    elif '.npy' in filename[-4:] :
        df = load( os.path.join(VALID_DIR, filename) )
    return df

# ---------------------------------------------------------------------------
# data file helper
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join( os.path.dirname(os.path.abspath(__file__)), "data" )

def LoadData( filename ):
    '''Return data DataFrame from data/ directory.'''
    return read_csv( os.path.join( DATA_DIR, filename ) )

# ---------------------------------------------------------------------------
# Default argument dictionaries — one per API function.
#
# Every parameter is listed. Parameters not actively tested carry a comment.
# Tests copy the relevant dict and update only the parameters under test
# ---------------------------------------------------------------------------

EmbeddingArgs = dict( columns    = None,
                      E          = 2,
                      tau        = -1,
                      outputFile = None,
                      plusminus  = False,
                      verbose    = False )

EmbedDim_ColumnsArgs = dict( target          = None,
                             maxE            = 15,
                             minE            = 1,
                             lib             = None,
                             pred            = None,
                             Tp              = 1,
                             tau             = -1,
                             exclusionRadius = 0,
                             firstMax        = False,
                             validLib        = [],
                             noTime          = False,
                             ignoreNan       = True,
                             nWorkers        = None,
                             mpMethod        = None,
                             chunksize       = 1,
                             outputFile      = None,
                             verbose         = False,
                             logPct          = 5,
                             plot            = False )

SMap_thetaArgs = dict( thetaList       = None,
                       target          = None,
                       column          = None,
                       E               = 2,
                       tau             = -1,
                       Tp              = 1,
                       exclusionRadius = 0,
                       lib             = None,
                       pred            = None,
                       cores           = 5,
                       mpMethod        = None,
                       chunksize       = 1,
                       embedded        = False,
                       outputFile      = None,
                       noTime          = False,
                       verbose         = False,
                       plot            = False )

SMap_TpArgs = dict( TpList          = None,
                    target          = None,
                    column          = None,
                    E               = 2,
                    tau             = -1,
                    theta           = 0,
                    exclusionRadius = 0,
                    lib             = None,
                    pred            = None,
                    cores           = 5,
                    mpMethod        = None,
                    chunksize       = 1,
                    embedded        = False,
                    outputFile      = None,
                    noTime          = False,
                    verbose         = False,
                    plot            = False )

CrossMap_ColumnsArgs = dict( target          = None,
                             E               = 0,
                             Evec            = None,
                             Tp              = 1,
                             tau             = -1,
                             exclusionRadius = 0,
                             lib             = None,
                             pred            = None,
                             cores           = 5,
                             mpMethod        = None,
                             chunksize       = 1,
                             returnError     = False,
                             noTime          = False,
                             outputFile      = None,
                             verbose         = False,
                             errorPlot       = 'rho',
                             plot            = False )

CrossMap_ColumnListArgs = dict( columns         = [],
                                target          = None,
                                E               = 0,
                                Tp              = 1,
                                tau             = -1,
                                exclusionRadius = 0,
                                lib             = None,
                                pred            = None,
                                embedded        = False,
                                cores           = 5,
                                mpMethod        = None,
                                chunksize       = 1,
                                outputFile      = None,
                                noTime          = False,
                                verbose         = False,
                                plot            = False )

CrossMap_MatrixArgs = dict( E               = 0,
                            Tp              = 1,
                            tau             = -1,
                            exclusionRadius = 0,
                            lib             = None,
                            pred            = None,
                            threshold       = None,
                            cores           = 5,
                            mpMethod        = None,
                            chunksize       = 1,
                            returnValue     = 'matrix', # or 'dataframe'
                            outputFile      = None,
                            noTime          = False,
                            verbose         = False,
                            plot            = False,
                            title           = None,
                            figsize         = (5,5),
                            dpi             = 150 )

CCM_MatrixArgs = dict( libSizes         = [],
                       pLibSizes        = [10,20,80,90],
                       Tp               = 0,
                       tau              = -1,
                       exclusionRadius  = 0,
                       sample           = 30,
                       seed             = None,
                       noTime           = False,
                       parallel         = True,
                       mpMethod         = None,
                       sharedMB         = 0.01,
                       targetBatchSize  = None,
                       expConverge      = False,
                       progressLog      = None,
                       progressInterval = 5 )
