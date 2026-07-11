#! /usr/bin/env python3

from datetime        import datetime
from argparse        import ArgumentParser
from multiprocessing import get_context
from itertools       import repeat

from   pandas import read_csv
import matplotlib.pyplot as plt

from pyEDM import SMap, sampleData

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def SMap_Tp( data, TpList = None, target = None, column = None,
             E = 2, tau = -1, theta = 0, exclusionRadius = 0,
             lib = None, pred = None, cores = 5, mpMethod = None,
             chunksize = 1, embedded = False,
             outputFile = None, noTime = False,
             verbose = False, plot = False ):
    
    '''Use multiprocessing Pool to process parallelise SMap.
       The TpList (-T) argument specifies a list of Tp.
       Return dict of Tp{Tp} : SMap dict
    '''
    
    startTime = datetime.now()
    
    if not target :
        raise( RuntimeError( 'target required' ) )
    if not  column:
        raise( RuntimeError( 'column required' ) )
    if not  TpList:
        raise( RuntimeError( 'TpList required' ) )

    # Dictionary of arguments for Pool : SMapTpFunc
    argsD = { 'target' : target, 'column' : column,
              'lib' : lib, 'pred' : pred, 'theta' : theta,
              'E' : E, 'tau' : tau, 'exclusionRadius' : exclusionRadius, 
              'embedded' : embedded, 'noTime' : noTime }

    # Create iterable for Pool.starmap, use repeated copies of args, data
    poolArgs = zip( TpList, repeat( argsD ), repeat( data ) )

    # Process pool
    mpContext = get_context( mpMethod )
    with mpContext.Pool( processes = cores ) as pool :
        SMapList = pool.starmap( SMapTpFunc, poolArgs, chunksize = chunksize )

    # SMapList is a list of SMap dictionaries : create dict with TpX keys
    keys = [ 'Tp' + str( k ) for k in TpList ]
    D = dict( zip( keys, SMapList ) )

    if verbose :
        print( f'Elapsed time:  {datetime.now() - startTime}' )

        print( D.keys() )

    # if -P : Plot Tp0 coefficients and predictions
    if plot and 'Tp0' in  D.keys() :
        coeff_df = D[ 'Tp0' ][ 'coefficients' ]
        coeff_df.plot( 'Time', coeff_df.columns[1:] )

        pred_df = D[ 'Tp0' ][ 'predictions' ]
        pred_df.plot( 'Time', pred_df.columns[1:] )

        plt.show()

    return D

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def SMapTpFunc( Tp, argsD, data ):
    '''Call pyEDM SMap using Tp, args, and data'''
    
    sm = SMap( dataFrame       = data,
               columns         = argsD['column'],
               target          = argsD['target'],
               lib             = argsD['lib'],
               pred            = argsD['pred'],
               E               = argsD['E'],
               Tp              = Tp, # Single value of TpList
               theta           = argsD['theta'],
               exclusionRadius = argsD['exclusionRadius'],
               embedded        = argsD['embedded'],
               noTime          = argsD['noTime'],
               showPlot        = False )

    return sm

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def SMap_Tp_CmdLine():
    '''Wrapper for SMap_Tp with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in sampleData
    if args.inputFile:
        dataFrame = read_csv( args.inputFile )
    elif args.inputData:
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call SMap_Tp()
    D = SMap_Tp( data = dataFrame, TpList = args.TpList, 
                 target = args.target, column = args.column,
                 E = args.E, tau = args.tau, theta = args.theta,
                 exclusionRadius = args.exclusionRadius,
                 lib = args.lib, pred = args.pred,
                 embedded = args.embedded, cores = args.cores,
                 mpMethod = args.mpMethod, chunksize = args.chunksize,
                 outputFile = args.outputFile, noTime = args.noTime,
                 verbose = args.verbose, plot = args.Plot )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = ArgumentParser( description = 'SMap multiprocess Tp' )
    
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

    parser.add_argument('-nT', '--noTime',
                        dest    = 'noTime',
                        action  = 'store_true',
                        default = False,
                        help    = 'noTime.')

    parser.add_argument('-v', '--verbose',
                        dest    = 'verbose',
                        action  = 'store_true',
                        default = False,
                        help    = 'verbose.')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int, 
                        action  = 'store',
                        default = 4,
                        help    = 'Multiprocessing cores.')

    parser.add_argument('-mp', '--mpMethod',
                        dest    = 'mpMethod', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Multiprocessing start method')

    parser.add_argument('-cz', '--chunksize',
                        dest   = 'chunksize', type = int,
                        action = 'store', default = 1,
                        help = 'ProcessPool chunksize')

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
    SMap_Tp_CmdLine()
