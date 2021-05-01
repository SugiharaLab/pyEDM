#! /usr/bin/env python3

import time, argparse
from   multiprocessing import Pool
from   itertools       import repeat

from pandas import read_csv, concat

from pyEDM  import CCM

#----------------------------------------------------------------------------
# Main module
#----------------------------------------------------------------------------
def main():
    '''Use multiprocessing Pool to process parallelise CCM.
       The libSizesList (-l) argument specifies a list of CCM libSizes.
       libSizesList arguments can be any valid CCM libSizes representations.

       For exmaple: ./CCM-Multiprocess.py -l "100 500 100" "750 1000" "1500"
    '''
    
    startTime = time.time()
    
    args = ParseCmdLine()
    
    Process( args )

    elapsedTime = time.time() - startTime
    print( "Normal Exit elapsed time:", round( elapsedTime, 2 ) )

#----------------------------------------------------------------------------
def Process( args ):

    data = read_csv( args.inputFile )

    # Create iterable for Pool.starmap, use repeated copies of args, data
    poolArgs = zip( args.libSizesList, repeat( args ), repeat( data ) )

    # Use pool.starmap to distribute among cores
    pool = Pool( processes = args.cores )
    
    # starmap: elements of the iterable argument are iterables
    #          that are unpacked as arguments
    CMList = pool.starmap( CrossMapFunc, poolArgs )

    print( "Result has ", str( len( CMList ) ), " items." )

    df = concat( CMList )
    print( df )

    if args.outputFile :
        df.to_csv( args.outputFile )

#----------------------------------------------------------------------------
def CrossMapFunc( libSizes, args, data ):
    '''Call pyEDM CCM using the libSizes, args, and data'''
    
    cm = CCM( dataFrame       = data,
              E               = args.E,
              exclusionRadius = args.exclusionRadius,
              columns         = args.column,
              target          = args.target,
              libSizes        = libSizes,  # Not from args
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

    parser.add_argument('-c', '--column',
                        dest    = 'column', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Input file data column name.')
    
    parser.add_argument('-t', '--target',
                        dest    = 'target', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Input file data target name.')
    
    parser.add_argument('-l', '--libSizesList', nargs = '+',
                        dest    = 'libSizesList', type = str, 
                        action  = 'store',
                        default = [ "100 300", "500 700",
                                    "1000 1500", "1800 2000" ],
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
