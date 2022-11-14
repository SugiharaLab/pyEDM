#! /usr/bin/env python3

import time, argparse
from   multiprocessing import Pool
from   itertools       import repeat

from   pandas import read_csv
import matplotlib.pyplot as plt

from pyEDM import SMap, sampleData

#----------------------------------------------------------------------------
# Main module
#----------------------------------------------------------------------------
def main():
    '''Use multiprocessing Pool to process parallelise SMap.
       The TpList (-T) argument specifies a list of Tp.
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

    # Create iterable for Pool.starmap, use repeated copies of args, data
    poolArgs = zip( args.TpList, repeat( args ), repeat( data ) )

    # Process pool
    pool = Pool( processes = args.cores )
    
    # Use pool.starmap to distribute among cores
    # starmap: elements of the iterable argument are
    #          iterables unpacked as arguments in SMapFunc()
    SMapList = pool.starmap( SMapFunc, poolArgs )

    # SMapList is a list of SMap dictionaries : create dict with TpX keys
    keys = [ 'Tp' + str( k ) for k in args.TpList ]
    D = dict( zip( keys, SMapList ) )
    
    print( D.keys() )

    # if -P : Plot Tp0 coefficients and predictions
    if args.plot and 'Tp0' in  D.keys() :
        coeff_df = D[ 'Tp0' ][ 'coefficients' ]
        coeff_df.plot()

        pred_df = D[ 'Tp0' ][ 'predictions' ]
        pred_df.plot()

        plt.show()

#----------------------------------------------------------------------------
def SMapFunc( Tp, args, data ):
    '''Call pyEDM SMap using Tp, args, and data'''
    
    sm = SMap( dataFrame       = data,
               lib             = args.lib,
               pred            = args.pred,
               E               = args.E,
               Tp              = Tp,   # Not from args
               exclusionRadius = args.exclusionRadius,
               columns         = args.column,
               target          = args.target,
               embedded        = args.embedded,
               showPlot        = False )

    return sm

#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = argparse.ArgumentParser( description = 'SMap multiprocess Tp' )
    
    parser.add_argument('-T', '--TpList', nargs = '+',
                        dest    = 'TpList', type = int, 
                        action  = 'store',
                        default = [-5,-4,-3,-2,-1,0,1,2,3,4,5],
                        help    = 'Prediction interval.')

    parser.add_argument('-i', '--inputFile',
                        dest    = 'inputFile', type = str, 
                        action  = 'store',
                        default = None,
                        help    = 'Input data file.')
    
    parser.add_argument('-d', '--inputData',
                        dest    = 'inputData', type = str, 
                        action  = 'store',
                        default = 'Lorenz5D',
                        help    = 'Input data frame in sampleData.')
    
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
                        default = 'V3',
                        help    = 'Input file data target name.')
    
    parser.add_argument('-l', '--lib',
                        dest    = 'lib', type = str, 
                        action  = 'store',
                        default = '1 990',
                        help    = 'lib.')

    parser.add_argument('-p', '--pred',
                        dest    = 'pred', type = str, 
                        action  = 'store',
                        default = '501 990',
                        help    = 'pred.')

    parser.add_argument('-th', '--theta',
                        dest    = 'theta', type = float, 
                        action  = 'store',
                        default = 3.3,
                        help    = 'theta.')

    parser.add_argument('-e', '--embedded',
                        dest    = 'embedded',
                        action  = 'store_true',
                        default = False,
                        help    = 'embedded flag.')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int, 
                        action  = 'store',
                        default = 4,
                        help    = 'Multiprocessing cores.')

    parser.add_argument('-P', '--plot',
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
