#! /usr/bin/env python3

# Python distribution modules
from argparse  import ArgumentParser
from warnings  import filterwarnings
from datetime  import datetime
from itertools import islice

# Community modules
from pandas     import concat, read_feather, read_csv
from matplotlib import pyplot as plt

# Local modules
import sys
sys.path.append( '../../../EDM/pyEDM/apps' )
from EmbedDim_Columns import EmbedDim_Columns

# numpy/lib/_function_base_impl.py:3000:
# RuntimeWarning: invalid value encountered in divide  c /= stddev[None, :]
filterwarnings( "ignore", category = RuntimeWarning )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def EDimColsBlock( data, blockSize, target = None, maxE = 15, minE = 1,
                   lib = None, pred = None, Tp = 1, tau = -1,
                   exclusionRadius = 0, firstMax = False,
                   validLib = [], noTime = False, ignoreNan = True,
                   cores = 5, EDimCores = 2,
                   mpMethod = None, chunksize = 1, 
                   outputFile = None, verbose = False, plot = False ):

    '''Wrapper for pyEDM/apps/EmbedDim_Columns()
    EmbedDim_Columns() uses concurrent.futures.ProcessPoolExecutor()
    to parallelize pyEDM EmbedDimension() for all columns of a matrix/
    DataFrame. EmbedDimension() uses multiprocessing.pool.Pool() to
    parallelize the embedding dimension evaluation.

    On big data the multiple levels of concurrence can strongly contend
    for CPU resources when the number of columns is large (>1000). One way
    to get around this is to break the columns into blocks and process
    each block serially.

    blockSize : size of the block

    With exception of the blockSize parameter, parameters are the same as
    EmbedDim_Columns():
       Two arguments contol use of cores:
         -C  args.cores is number of processors running EmbedDim_Columns().
         -EC args.EDimCores is number of processors for EmbedDimension().
       The product should not exceed available capacity.
       The multiprocessing context is set to mpMethod. 

       If target is not specified EmbedDimension is univariate where each
       embedding evaluation is of the data column itself. It target (-t)
       is specified it is a single column against which all other columns
       are cross mapped and evaluated against embedding dimension.

       If firstMax is True, return the intial (lowest E) maximum.

       Return : DataFrame[ 'column', 'target', 'maxE', 'maxRho' ]
    '''

    startTime = datetime.now()
    N_cols    = len( data.columns )

    if verbose :
        print( f'EDimColsBlock(): {startTime}' )
        print( f'data shape {data.shape} N_cols={N_cols}' )

    if blockSize > N_cols :
        raise RuntimeError( f'blockSize must be less than num columns {N_cols}' )

    if outputFile :
        if not outputFile[-8:] in '.feather'             and \
           not outputFile[-4:] in ['.pkl','.csv','.zip'] and \
           not outputFile[-3:] in ['.gz','.xz'] :
            err = 'EDimColsBlock() ' +\
                f'unrecognized outputFile format: {outputFile}'
            raise RuntimeError( err )

    # generator of islice iterators of columns into N blocks
    block_i = blockGenerate( range( N_cols ), blockSize )

    # List of blockSize iterators with indices for each block
    g = [_ for _ in block_i]

    N_blocks = len( g )

    if verbose :
        print( datetime.now() )
        print( f'blockSize {blockSize}  N_blocks {N_blocks }' )

    # Container for DataFrame of each block
    L = [None] * N_blocks

    for k in range( N_blocks ) :
        b = g[k]            # k_th iterator
        i = [_ for _ in b]  # block indices from g[k]

        if not noTime and k == 0 :
            # Remove index 0 of time column
            i = i[1:]

        dataBlock = data.iloc[:,i] # subset data[ :, block ]

        # DataFrame{'column':maxCols,'target':targets,'E':maxE,'rho':maxRho}
        L[k] = EmbedDim_Columns( dataBlock,
                                 target          = target,
                                 maxE            = maxE,
                                 minE            = minE,
                                 lib             = lib,
                                 pred            = pred,
                                 Tp              = Tp,
                                 tau             = tau,
                                 exclusionRadius = exclusionRadius,
                                 firstMax        = firstMax,
                                 validLib        = validLib,
                                 noTime          = True,
                                 ignoreNan       = ignoreNan,
                                 cores           = cores,
                                 EDimCores       = EDimCores,
                                 mpMethod        = mpMethod,
                                 chunksize       = chunksize,
                                 outputFile      = None,
                                 verbose         = False,
                                 plot            = False )

    DF = concat( L )

    if outputFile :
        if '.csv' in outputFile[-4:] :
            DF.to_csv( outputFile, index = False )
        elif '.feather' in outputFile[-8:] :
            DF.to_feather( outputFile )
        elif any([_ in outputFile[-4:] for _ in ['.pkl','.gz','.xz','.zip'] ]):
            DF.to_pickle( outputFile )
        else :
            err = 'EDimColsBlock() ' +\
                f' Failed to write outputFile: {outputFile}'
            print( err )

    if verbose :
        print( f'Elapsed Time {datetime.now() - startTime}' )
        print( DF.head(3) )

    if plot :
        DF.plot( 'column', 'E' )
        plt.show()

    return DF

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def blockGenerate(iterable, N):
    '''Given iterable of NxN items, generate slices corresponding
       to N rows of the NxN product matrix'''
    N_  = len(iterable)
    k_s = 0 # block start index
    while k_s < N_ :
        yield islice( iterable, k_s, k_s + N )
        k_s = k_s + N

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def EDimColsBlock_CmdLine():
    '''Wrapper for EmbedDim_Columns with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in pyEDM sampleData
    if args.inputFile:
        if '.csv' in args.inputFile[-4:] :
            dataFrame = read_csv( args.inputFile )
        elif '.feather' in args.inputFile[-8:] :
            dataFrame = read_feather( args.inputFile )
        else :
            msg = f'Input file {args.inputFile} must be csv or feather'
            raise( RuntimeError( msg ) )
    elif args.inputData:
        from pyEDM import sampleData
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    # Call EmbedDim_Columns()
    DF = EDimColsBlock( data = dataFrame, blockSize = args.blockSize,
                        target = args.target,
                        maxE = args.maxE, minE = args.minE,
                        lib = args.lib, pred = args.pred,
                        Tp = args.Tp, tau = args.tau,
                        exclusionRadius = args.exclusionRadius,
                        firstMax = args.firstMax,
                        validLib = args.validLib, noTime = args.noTime,
                        ignoreNan = args.ignoreNan,
                        cores = args.cores, EDimCores = args.EDimCores,
                        mpMethod = args.mpMethod, chunksize = args.chunksize,
                        outputFile = args.outputFile,
                        verbose = args.verbose, plot = args.Plot )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():

    parser = ArgumentParser( description = 'EDim Cols Block' )

    parser.add_argument('-i', '--inputFile',
                        dest    = 'inputFile', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Input data file.')

    parser.add_argument('-d', '--inputData',
                        dest    = 'inputData', type = str,
                        action  = 'store',
                        default = 'Lorenz5D',
                        help    = 'pyEDM sample data name.')

    parser.add_argument('-o', '--outputFile',
                        dest    = 'outputFile', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Output file: csv feather gz xz pkl.')

    parser.add_argument('-b', '--blockSize', required = True,
                        dest    = 'blockSize', type = int,
                        action  = 'store',
                        #default = 1,
                        help    = 'blockSize')

    parser.add_argument('-t', '--target',
                        dest    = 'target', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Target column')

    parser.add_argument('-maxE', '--maxE',
                        dest    = 'maxE', type = int,
                        action  = 'store',
                        default = 15,
                        help    = 'maxE')

    parser.add_argument('-minE', '--minE',
                        dest    = 'minE', type = int,
                        action  = 'store',
                        default = 1,
                        help    = 'minE')

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
                        help    = 'Tp')

    parser.add_argument('-tau', '--tau',
                        dest    = 'tau', type = int,
                        action  = 'store',
                        default = -1,
                        help    = 'tau')

    parser.add_argument('-x', '--exclusionRadius',
                        dest    = 'exclusionRadius', type = int,
                        action  = 'store',
                        default = 0,
                        help    = 'Exclusion Radius')

    parser.add_argument('-f', '--firstMax',
                        dest   = 'firstMax',
                        action = 'store_true', default = False,
                        help = 'Choose first in rho(E)')

    parser.add_argument('-VL', '--validLib',
                        dest   = 'validLib', type = str,
                        action = 'store', default = '',
                        help = 'expression to eval for validLib')

    parser.add_argument('-noTime', '--noTime',
                        dest   = 'noTime',
                        action = 'store_true', default = False,
                        help = 'Set noTime True')

    parser.add_argument('-in', '--ignoreNan',
                        dest   = 'ignoreNan',
                        action = 'store_false', default = True,
                        help = 'Set ignoreNan False')

    parser.add_argument('-mp', '--mpMethod',
                        dest    = 'mpMethod', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Multiprocessing start method')

    parser.add_argument('-C', '--cores',
                        dest    = 'cores', type = int,
                        action  = 'store',
                        default = 2,
                        help    = 'Multiprocessing cores')

    parser.add_argument('-EC', '--EDimCores',
                        dest   = 'EDimCores', type = int,
                        action = 'store', default = 5,
                        help = 'EmbedDim multiprocessing cores')

    parser.add_argument('-cz', '--chunksize',
                        dest   = 'chunksize', type = int,
                        action = 'store', default = 1,
                        help = 'ProcessPoolExecutor.map chunksize')

    parser.add_argument('-v', '--verbose',
                        dest    = 'verbose',
                        action  = 'store_true',
                        default = False,
                        help    = 'verbose')

    parser.add_argument('-P', '--Plot',
                        dest    = 'Plot',
                        action  = 'store_true',
                        default = False,
                        help    = 'Plot')

    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    EDimColsBlock_CmdLine()
