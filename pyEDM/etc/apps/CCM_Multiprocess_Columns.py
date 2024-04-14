#! /usr/bin/env python3

import time, argparse
from   multiprocessing import Pool
from   itertools       import repeat

from pandas import DataFrame, read_csv, concat
from pyEDM  import CCM, sampleData
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CCM_Columns( data, target = None, E = 2, Tp = 0, tau = -1,
                 libSizes = None, sample = 50,
                 exclusionRadius = 0, cores = 4,
                 noTime = False, outputFile = None,
                 verbose = False, plot = False ):
    
    '''Use multiprocessing Pool to process parallelise CCM.
       The target (-t) argument specifies a column against which all
       other columns will be cross mapped.
    '''

    startTime = time.time()

    if not target :
        raise( RuntimeError( 'target required' ) )

    if not libSizes :
        if data.shape[0] < 50 :
            raise( RuntimeError( 'libSizes required' ) )
        
        # Heuristic sample of library sizes
        libLow   = 20
        libHigh  = int( data.shape[0] - 20 )
        libDelta = int( data.shape[0] / 10 )
        libSizes = [ x for x in range( libLow, libHigh, libDelta ) ]

    # Ignore first column, convert to list
    if not noTime :
        columns = data.columns[ 1 : len(data.columns) ].to_list()
    # Remove target from columns since CCM maps both
    columns.remove( target )

    # Dictionary of arguments for Pool : CrossMapFunc
    argsD = { 'target' : target, 'E' : E, 'Tp' : Tp, 'tau' : tau,
              'libSizes' : libSizes, 'sample' : sample,
              'exclusionRadius' : exclusionRadius, 'noTime' : noTime }

    # Create iterable for Pool.starmap, use repeated copies of argsD, data
    poolArgs = zip( columns, repeat( argsD ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    pool = Pool( processes = cores )
    
    # starmap: elements of the iterable argument are iterables
    #          that are unpacked as arguments
    CMList = pool.starmap( CrossMapFuncColumns, poolArgs )

    df = concat( CMList, axis = 1 )
    df = df.loc[ :, ~df.columns.duplicated() ] # Remove degenerate LibSizes

    print( "Elapsed time:", round( time.time() - startTime, 2 ) )

    if verbose :
        print( df )

    if plot :
        df.plot( 'LibSize', df.columns[1:], linewidth = 3,
                 ylabel = 'CCM ρ' )
        plt.show()

    if outputFile :
        df.to_csv( outputFile )

    return df

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CrossMapFuncColumns( column, argsD, data ):
    '''Call pyEDM CCM using the column, argsD, and data'''
    
    cm = CCM( dataFrame       = data,
              E               = argsD['E'],
              Tp              = argsD['Tp'],
              tau             = argsD['tau'],
              exclusionRadius = argsD['exclusionRadius'],
              columns         = column,
              target          = argsD['target'],
              libSizes        = argsD['libSizes'],
              sample          = argsD['sample'],
              showPlot        = False )

    return cm
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CCM_Columns_CmdLine():
    '''Wrapper for CCM_Columns with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in sampleData
    if args.inputFile:
        dataFrame = read_csv( args.inputFile )
    elif args.inputData:
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call CCM_Columns()
    df = CCM_Columns( data = dataFrame, target = args.target,
                      E = args.E, Tp = args.Tp, tau = args.tau,
                      libSizes = args.libSizes,
                      sample = args.sample,
                      exclusionRadius = args.exclusionRadius,
                      cores = args.cores, noTime = args.noTime,
                      outputFile = args.outputFile,
                      verbose = args.verbose, plot = args.Plot )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = argparse.ArgumentParser( description = 'CCM_Columns' )
    
    parser.add_argument('-i', '--inputFile',
                        dest    = 'inputFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Input data file.')
    
    parser.add_argument('-d', '--inputData',
                        dest    = 'inputData', type = str, 
                        action  = 'store',
                        default = 'Lorenz5D',
                        help    = 'Input data frame name.')
    
    parser.add_argument('-o', '--outputFile',
                        dest    = 'outputFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Output file.')

    parser.add_argument('-E', '--E',
                        dest    = 'E', type = int, 
                        action  = 'store',
                        default = 3,
                        help    = 'Embedding dimension E.')

    parser.add_argument('-T', '--Tp',
                        dest    = 'Tp', type = int, 
                        action  = 'store',
                        default = 0,
                        help    = 'Tp.')

    parser.add_argument('-tau', '--tau',
                        dest    = 'tau', type = int, 
                        action  = 'store',
                        default = -1,
                        help    = 'tau.')

    parser.add_argument('-x', '--exclusionRadius',
                        dest    = 'exclusionRadius', type = int, 
                        action  = 'store',
                        default = 0,
                        help    = 'Exclusion Radius.')

    parser.add_argument('-t', '--target',
                        dest    = 'target', type = str, 
                        action  = 'store',
                        default = 'V1',
                        help    = 'Input file data target name.')
    
    parser.add_argument('-l', '--libSizes',
                        dest    = 'libSizes', type = str, 
                        action  = 'store',
                        default = "40 80 120 200 400 600 800 980",
                        help    = 'CCM library sizes.')

    parser.add_argument('-s', '--sample',
                        dest    = 'sample', type = int, 
                        action  = 'store',
                        default = 50,
                        help    = 'CCM samples.')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int, 
                        action  = 'store',
                        default = 4,
                        help    = 'Multiprocessing cores.')

    parser.add_argument('-n', '--noTime',
                        dest    = 'noTime',
                        action  = 'store_true',
                        default = False,
                        help    = 'noTime.')

    parser.add_argument('-v', '--verbose',
                        dest    = 'verbose',
                        action  = 'store_true',
                        default = False,
                        help    = 'verbose.')

    parser.add_argument('-P', '--Plot',
                        dest    = 'Plot',
                        action  = 'store_true',
                        default = False,
                        help    = 'Plot results.')

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    CCM_Columns_CmdLine()
