#! /usr/bin/env python3

# Python distribution modules
import argparse
from time               import time
from pickle             import dump
from itertools          import repeat
from concurrent.futures import ProcessPoolExecutor

# Community modules
from pandas import DataFrame
from pyEDM  import EmbedDimension, sampleData
from matplotlib import pyplot as plt

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def EmbedDim_Columns( data, target = None, maxE = 15,
                      lib = None, pred = None, Tp = 1, tau = -1,
                      exclusionRadius = 0, validLib = [], noTime = False,
                      ignoreNan = True, cores = 2, EDimCores = 5,
                      outputFile = None, verbose = False, plot = False ):

    '''Use multiprocessing Pool to process parallelise EmbedDimension.
       Note EmbedDimension is already multiprocessing but usually will
       not require more than a few processes, maxE at most. On platforms
       with many cores this application multiprocesses instances of
       EmbedDimension. There are two arguments to contol the number
       of cores used for each:
         -C  args.cores is number of processors running EmbedDimFunc().
         -EC args.EDimCores is number of processors for EmbedDimension().
       The total should not exceed available capacity

       If target is not specified EmbedDimension is univariate where each
       embedding evaluation is of the data column itself. It target (-t)
       is specified it is a single column against which all other columns
       are cross mapped.
    '''

    startTime = time()

    # If no lib and pred, create from full data span
    if lib is None :
        lib = [ 1, data.shape[0] ]
    if pred is None :
        pred = [ 1, data.shape[0] ]

    # Ignore first column, convert to list
    if not noTime :
        columns = data.columns[ 1 : len(data.columns) ].to_list()

    N = len( columns )

    # Dictionary of arguments for PoolExecutor : EmbedDimFunc
    argsD = { 'target'          : target,
              'maxE'            : maxE,
              'lib'             : lib,
              'pred'            : pred,
              'Tp'              : Tp,
              'tau'             : tau,
              'exclusionRadius' : exclusionRadius,
              'validLib'        : validLib,
              'noTime'          : noTime,
              'ignoreNan'       : ignoreNan,
              'EDimCores'       : EDimCores }

    # ProcessPoolExecutor has no starmap(). Pass argument lists directly.
    with ProcessPoolExecutor( max_workers = cores ) as exe :
        EDim = exe.map(EmbedDimFunc, columns, repeat(data,N), repeat(argsD,N))

    # EDim is a generator of dictionaries from EmbedDimFunc
    # Fill maxCols, targets and maxE arrays, dict
    maxCols = [None] * N
    targets = [None] * N
    maxE    = [0]    * N
    i       = 0
    for edim in EDim :
        maxCols[i] = edim['column']
        targets[i] = edim['target']
        maxE[i]    = edim['maxE']
        i = i + 1

    D = { 'column':maxCols, 'target':targets, 'E':maxE }

    if outputFile :
        with open( outputFile, 'wb' ) as fileObj:
            dump( D, fileObj )

    ET = round( time() - startTime, 2 )
    print( f'Elapsed Time {ET}' )

    if verbose :
        print( DataFrame( D ) )

    if plot :
        DataFrame( D ).plot( 'column', 'E' )
        plt.show()

    return D

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def EmbedDimFunc( column, df, args ):
    '''Estimate optimal embedding dimension [1:maxE]
       If no target specified : univariate embedding of column'''

    if args['target'] :
        target = args['target']
    else :
        target = column

    ed = EmbedDimension( dataFrame       = df,
                         columns         = column,
                         target          = target,
                         maxE            = args['maxE'],
                         lib             = args['lib'],
                         pred            = args['pred'],
                         Tp              = args['Tp'],
                         tau             = args['tau'],
                         exclusionRadius = args['exclusionRadius'],
                         validLib        = args['validLib'],
                         noTime          = args['noTime'],
                         ignoreNan       = args['ignoreNan'],
                         verbose         = False,
                         numProcess      = args['EDimCores'],
                         showPlot        = False )

    # Find max E(rho)
    iMax = ed['rho'].round(2).argmax()
    maxE = ed['E'].iloc[ iMax ]

    return { 'column':column, 'target':target, 'maxE':maxE } #, 'EDim':ed }

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def EmbedDim_Columns_CmdLine():
    '''Wrapper for EmbedDim_Columns with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in sampleData
    if args.inputFile:
        dataFrame = read_csv( args.inputFile )
    elif args.inputData:
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call EmbedDim_Columns()
    D = EmbedDim_Columns( data = dataFrame,
                          target = args.target, maxE = args.maxE,
                          lib = args.lib, pred = args.pred,
                          Tp = args.Tp, tau = args.tau,
                          exclusionRadius = args.exclusionRadius,
                          validLib = args.validLib, noTime = args.noTime,
                          ignoreNan = args.ignoreNan, cores = args.cores,
                          EDimCores = args.EDimCores,
                          outputFile = args.outputFile,
                          verbose = args.verbose, plot = args.Plot )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():

    parser = argparse.ArgumentParser( description = 'CrossMap Columns' )

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

    parser.add_argument('-t', '--target',
                        dest    = 'target', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Data target name.')

    parser.add_argument('-maxE', '--maxE',
                        dest    = 'maxE', type = int,
                        action  = 'store',
                        default = 15,
                        help    = 'maxE.')

    parser.add_argument('-l', '--lib', nargs = 2,
                        dest   = 'lib', type = int,
                        action = 'store', default = None,
                        help = 'lib')

    parser.add_argument('-pred', '--pred', nargs = 2,
                        dest   = 'pred', type = int,
                        action = 'store', default = None,
                        help = 'pred')

    parser.add_argument('-T', '--Tp',
                        dest    = 'Tp', type = int,
                        action  = 'store',
                        default = 1,
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

    parser.add_argument('-VL', '--validLib',
                        dest   = 'validLib', type = str,
                        action = 'store', default = '',
                        help = 'expression to eval for validLib')

    parser.add_argument('-noTime', '--noTime',
                        dest   = 'noTime',
                        action = 'store_true', default = False,
                        help = 'noTime')

    parser.add_argument('-in', '--ignoreNan',
                        dest   = 'ignoreNan',
                        action = 'store_false', default = True,
                        help = 'ignoreNan')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int,
                        action  = 'store',
                        default = 2,
                        help    = 'Multiprocessing cores.')

    parser.add_argument('-EC', '--EDimCores',
                        dest   = 'EDimCores', type = int,
                        action = 'store', default = 5,
                        help = 'EmbedDim multiprocessing cores.')

    parser.add_argument('-v', '--verbose',
                        dest    = 'verbose',
                        action  = 'store_true',
                        default = False,
                        help    = 'verbose.')

    parser.add_argument('-P', '--Plot',
                        dest    = 'Plot',
                        action  = 'store_true',
                        default = False,
                        help    = 'Plot.')

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    EmbedDim_Columns_CmdLine()
