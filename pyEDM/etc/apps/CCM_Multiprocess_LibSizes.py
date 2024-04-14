#! /usr/bin/env python3

import time, argparse
from   multiprocessing import Pool
from   itertools       import repeat

from pandas import read_csv, concat
from pyEDM  import CCM, sampleData
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CCM_LibSizes( data, target = None, column = None,
                  E = 2, Tp = 0, tau = -1,
                  libSizesList = None, sample = 50,
                  exclusionRadius = 0, cores = 4,
                  noTime = False, outputFile = None,
                  verbose = False, plot = False ):
    
    '''Use multiprocessing Pool to process parallelise CCM.
       The libSizesList (-l) argument specifies a list of CCM libSizes.
       libSizesList arguments can be any valid CCM libSizes representations.

       For exmaple: ./CCM-Multiprocess.py -l "100 500 100" "750 1000" "1500"
    '''
    
    startTime = time.time()
    
    if not target :
        raise( RuntimeError( 'target required' ) )
    if not  column:
        raise( RuntimeError( 'column required' ) )
    if not libSizesList :
        raise( RuntimeError( 'libSizesList required' ) )

    # Ignore first column, convert to list
    if not noTime :
        columns = data.columns[ 1 : len(data.columns) ].to_list()
    # Remove target from columns since CCM maps both
    columns.remove( target )

    # Dictionary of arguments for Pool : CrossMapFunc
    argsD = { 'target' : target, 'column' : column,
              'E' : E, 'Tp' : Tp, 'tau' : tau,
              'sample' : sample, 'exclusionRadius' : exclusionRadius, 
              'noTime' : noTime }

    # Create iterable for Pool.starmap, use repeated copies of argsD, data
    poolArgs = zip( libSizesList, repeat( argsD ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    pool = Pool( processes = cores )
    
    # starmap: elements of the iterable argument are iterables
    #          that are unpacked as arguments
    CMList = pool.starmap( CrossMapFuncLibSizes, poolArgs )

    df = concat( CMList )

    print( "Elapsed time:", round( time.time() - startTime, 2 ) )

    if verbose :
        print( df )

    if plot :
        df.plot( 'LibSize',
                 [f'{column}:{target}',
                  f'{target}:{column}'], linewidth = 3,
                 ylabel = 'CCM ρ' )
        plt.show()

    if outputFile :
        df.to_csv( outputFile )

    return df

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CrossMapFuncLibSizes( libSizes, argsD, data ):
    '''Call pyEDM CCM using the libSizes, args, and data'''
    
    cm = CCM( dataFrame       = data,
              E               = argsD['E'],
              Tp              = argsD['Tp'],
              tau             = argsD['tau'],
              exclusionRadius = argsD['exclusionRadius'],
              columns         = argsD['column'],
              target          = argsD['target'],
              libSizes        = libSizes, # One of libSizesList
              sample          = argsD['sample'],
              showPlot        = False )

    return cm
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def CCM_LibSizes_CmdLine():
    '''Wrapper for CCM_LibSizes with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in sampleData
    if args.inputFile:
        dataFrame = read_csv( args.inputFile )
    elif args.inputData:
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call CCM_LibSizes()
    df = CCM_LibSizes( data = dataFrame,
                       target = args.target, column = args.column,
                       E = args.E, Tp = args.Tp, tau = args.tau,
                       libSizesList = args.libSizesList,
                       sample = args.sample,
                       exclusionRadius = args.exclusionRadius,
                       cores = args.cores, noTime = args.noTime,
                       outputFile = args.outputFile,
                       verbose = args.verbose, plot = args.Plot )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = argparse.ArgumentParser( description = 'CCM_LibSizes' )
    
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
                        default = 5,
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

    parser.add_argument('-c', '--column',
                        dest    = 'column', type = str, 
                        action  = 'store',
                        default = 'V1',
                        help    = 'Input file data column name.')
    
    parser.add_argument('-t', '--target',
                        dest    = 'target', type = str, 
                        action  = 'store',
                        default = 'V5',
                        help    = 'Input file data target name.')
    
    parser.add_argument('-l', '--libSizesList', nargs = '+',
                        dest    = 'libSizesList', type = str, 
                        action  = 'store',
                        default = [ "50 75 100 200 300", "400 500 600",
                                    "700 800", "900 990" ],
                        help    = 'CCM library sizes.')

    parser.add_argument('-s', '--sample',
                        dest    = 'sample', type = int, 
                        action  = 'store',
                        default = 100,
                        help    = 'CCM samples.')

    parser.add_argument('-n', '--noTime',
                        dest    = 'noTime',
                        action  = 'store_true',
                        default = False,
                        help    = 'noTime.')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int, 
                        action  = 'store',
                        default = 4,
                        help    = 'Multiprocessing cores.')

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
    CCM_LibSizes_CmdLine()
