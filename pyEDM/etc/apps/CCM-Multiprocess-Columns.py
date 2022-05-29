#! /usr/bin/env python3

import time, argparse
from   multiprocessing import Pool
from   itertools       import repeat

from pandas import DataFrame, read_csv, concat

from pyEDM  import CCM, sampleData

#----------------------------------------------------------------------------
# Main module
#----------------------------------------------------------------------------
def main():
    '''Use multiprocessing Pool to process parallelise CCM.
       The target (-t) argument specifies a column against which all
       other columns will be cross mapped.
    '''

    startTime = time.time()

    args = ParseCmdLine()

    Process( args )

    elapsedTime = time.time() - startTime
    print( "Normal Exit elapsed time:", round( elapsedTime, 2 ) )

#----------------------------------------------------------------------------
def Process( args ):

    # If -i input file: load it, else look for inputData in sampleData
    if args.inputFile:
        data = read_csv( args.inputFile )
    elif args.inputData:
        data = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Ignore first column, convert to list
    columns = data.columns[ 1 : len(data.columns) ].to_list() 
    columns.remove( args.target )

    # Create iterable for Pool.starmap, use repeated copies of args, data
    poolArgs = zip( columns, repeat( args ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    pool = Pool( processes = args.cores )
    
    # starmap: elements of the iterable argument are iterables
    #          that are unpacked as arguments
    CMList = pool.starmap( CrossMapFunc, poolArgs )

    df = concat( CMList, 1 )
    df = df.loc[ :, ~df.columns.duplicated() ] # Remove degenerate LibSizes
    print( df )

    if args.outputFile :
        df.to_csv( args.outputFile )

#----------------------------------------------------------------------------
def CrossMapFunc( column, args, data ):
    '''Call pyEDM CCM using the column, args, and data'''
    
    cm = CCM( dataFrame       = data,
              E               = args.E,
              exclusionRadius = args.exclusionRadius,
              columns         = column,
              target          = args.target,
              libSizes        = args.libSizes,
              sample          = args.sample,
              showPlot        = False )

    return cm
    
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = argparse.ArgumentParser( description = 'Test Program' )
    
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
                        default = 100,
                        help    = 'CCM samples.')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int, 
                        action  = 'store',
                        default = 4,
                        help    = 'Multiprocessing cores.')

    parser.add_argument('-p', '--plot',
                        dest    = 'plot',
                        action  = 'store_true',
                        default = False,
                        help    = 'Plot results.')

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    main()
