"""
conftest.py — shared resources for pyEDM pytest suite.

pytest loads this file automatically before any test collection.
Everything defined here is available to all test files in this
directory without an explicit import.

Contents:
  - GetMP_ContextName()  multiprocessing context helper
  - ValidData()          load a validation CSV by filename
  - *Args dicts          default keyword arguments for each EDM API function
"""

import os
from multiprocessing import get_context, get_start_method

from pandas import read_csv
import pyEDM as EDM

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

VALID_DIR = os.path.join( os.path.dirname( EDM.__file__ ), "tests", "validation" )

def ValidData( filename ):
    '''Return DataFrame of validation CSV from the package validation/ directory.'''
    return read_csv( os.path.join( VALID_DIR, filename ) )

# ---------------------------------------------------------------------------
# Default argument dictionaries — one per API function.
#
# Every parameter is listed. Parameters not actively tested carry a comment.
# Tests copy the relevant dict and update only the parameters under test,
# making each test's variation immediately visible.
# ---------------------------------------------------------------------------

EmbedArgs = dict( E           = 0,
                  tau         = -1,
                  columns     = "",
                  includeTime = False,
                  pathIn      = "./",
                  dataFile    = None )

SimplexArgs = dict( columns         = "",
                    target          = "",
                    lib             = "",
                    pred            = "",
                    E               = 0,
                    Tp              = 1,
                    knn             = 0,
                    tau             = -1,
                    exclusionRadius = 0,
                    embedded        = False,
                    validLib        = [],
                    noTime          = False,    # not tested individually
                    generateSteps   = 0,        # tested in test_generate.py
                    generateConcat  = False,    # tested in test_generate.py
                    verbose         = False,    # not tested
                    showPlot        = False,    # not tested
                    ignoreNan       = True,
                    returnObject    = False )

SMapArgs = dict( columns         = "",
                 target          = "",
                 lib             = "",
                 pred            = "",
                 E               = 0,
                 Tp              = 1,
                 knn             = 0,
                 tau             = -1,
                 theta           = 0,
                 exclusionRadius = 0,
                 solver          = None,        # not tested
                 embedded        = False,
                 validLib        = [],
                 noTime          = False,       # tested in test_smap.py
                 generateSteps   = 0,           # tested in test_generate.py
                 generateConcat  = False,       # tested in test_generate.py
                 ignoreNan       = True,
                 showPlot        = False,       # not tested
                 verbose         = False,       # not tested
                 returnObject    = False )

CCMArgs = dict( columns         = "",
                target          = "",
                libSizes        = "",
                sample          = 30,
                E               = 0,
                Tp              = 0,
                knn             = 0,
                tau             = -1,
                exclusionRadius = 0,
                seed            = None,
                embedded        = False,
                validLib        = [],
                includeData     = False,        # not tested
                noTime          = False,        # not tested
                mpMethod        = None,         # not tested
                parallel        = True,         # not tested
                sharedMB        = 0.01,         # not tested
                verbose         = False,        # not tested
                showPlot        = False,        # not tested
                returnObject    = False,
                legacy          = False )       # not tested

MultiviewArgs = dict( columns         = "",
                      target          = "",
                      lib             = "",
                      pred            = "",
                      D               = 0,
                      E               = 1,
                      Tp              = 1,
                      knn             = 0,
                      tau             = -1,
                      multiview       = 0,
                      exclusionRadius = 0,
                      trainLib        = True,
                      excludeTarget   = False,
                      ignoreNan       = True,   # not tested
                      verbose         = False,  # not tested
                      numProcess      = 4,
                      mpMethod        = None,
                      chunksize       = 1,      # not tested
                      showPlot        = False,
                      returnObject    = False )

EmbedDimensionArgs = dict( columns         = "",
                           target          = "",
                           maxE            = 10,
                           lib             = "",
                           pred            = "",
                           Tp              = 1,
                           tau             = -1,
                           exclusionRadius = 0,
                           embedded        = False,  # not tested
                           validLib        = [],     # not tested
                           noTime          = False,  # not tested
                           ignoreNan       = True,   # not tested
                           verbose         = False,  # not tested
                           numProcess      = 4,
                           mpMethod        = None,
                           chunksize       = 1,      # not tested
                           showPlot        = False )

PredictIntervalArgs = dict( columns         = "",
                            target          = "",
                            lib             = "",
                            pred            = "",
                            maxTp           = 10,
                            E               = 1,
                            tau             = -1,
                            exclusionRadius = 0,
                            embedded        = False,  # not tested
                            validLib        = [],     # not tested
                            noTime          = False,  # not tested
                            ignoreNan       = True,   # not tested
                            verbose         = False,  # not tested
                            numProcess      = 4,
                            mpMethod        = None,
                            chunksize       = 1,      # not tested
                            showPlot        = False )

PredictNonlinearArgs = dict( columns         = "",
                             target          = "",
                             theta           = None,
                             lib             = "",
                             pred            = "",
                             E               = 1,
                             Tp              = 1,
                             knn             = 0,
                             tau             = -1,
                             exclusionRadius = 0,
                             solver          = None,   # not tested
                             embedded        = False,  # not tested
                             validLib        = [],     # not tested
                             noTime          = False,  # not tested
                             ignoreNan       = True,   # not tested
                             verbose         = False,  # not tested
                             numProcess      = 4,
                             mpMethod        = None,
                             chunksize       = 1,      # not tested
                             showPlot        = False )
