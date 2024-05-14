#! /usr/bin/env python3

import argparse

from pyEDM  import Embed
from pandas import concat, read_csv

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def Embedding( data, columns = None, E = 2, tau = -1,
               outputFile = None, plusminus = False, verbose = False ):
    '''
    EDM Embed wrapper
    Create time-delay embedding with time column for EDM. 
    Useful to create mixed multivariate embeddings for SMap and
    embeddings with time-advanced vectors. 
 
    Add Time column. Replace V(t+0) and V(t-0) with V.
    If args.plusminus create both time-delay and time-advanced columns,
    require args.tau < 0.
    '''

    # Presume time is first column
    timeName   = data.columns[0]
    timeSeries = data[ timeName ]

    # If no columns specified, use all except first
    if not columns :
        columns = data.columns[ 1: ]

    if verbose :
        print( "Input time column: ", timeName )
        print( "Input columns: ", columns )

    # Create embeddings of columns
    # There will be redundancies vis V1(t-0), V1(t+0)
    if plusminus :
        embed_minus = Embed( dataFrame = data, E = E, tau = tau,
                             columns = columns )
        embed_plus = Embed( dataFrame = data, E = E, tau = abs( tau ),
                            columns = columns )
        df = concat( [ timeSeries, embed_minus, embed_plus ], axis = 1 )

        # Remove *(t+0) columns redundant with *(t-0)
        cols_tplus0 = [ c for c in df.columns if '(t+0)' in c ]
        df = df.drop( columns = cols_tplus0 )

    else :
        embed_ = Embed( dataFrame = data, E = E, tau = tau,
                        columns = columns )
        df = concat( [ timeSeries, embed_ ], axis = 1 )

    # Rename *(t+0) to original column names
    cols_tplus0 = [ c for c in df.columns if '(t+0)' in c ]
    cols_orig   = [ c.replace( '(t+0)', '', 1 ) for c in cols_tplus0 ]
    df.rename( columns = dict( zip( cols_tplus0, cols_orig ) ), inplace = True )

    # Rename *(t-0) to original column names
    cols_tminus0 = [ c for c in df.columns if '(t-0)' in c ]
    cols_orig    = [ c.replace( '(t-0)', '', 1 ) for c in cols_tminus0 ]
    df.rename( columns = dict( zip( cols_tminus0, cols_orig ) ), inplace = True )

    # Rename first column to original time column name
    df.rename( columns = { df.columns[0] : timeName }, inplace = True )

    if verbose :
        print( "Columns:", df.columns, "\n-------------------------------" )
        print( df.head( 4 ) )
        print( df.tail( 4 ) )

    if outputFile :
        df.to_csv( outputFile, index = False )

    return df

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def Embedding_CmdLine():
    '''Wrapper for Embedding with command line parsing'''

    args = ParseCmdLine()

    # Read data
    if not args.inputFile :
        raise( RuntimeError( '-i inputFile argument required to read .csv' ) )

    dataFrame = read_csv( args.inputFile )

    # Call Embedding()
    df = Embedding( data = dataFrame, columns = args.columns,
                    E = args.E, tau = args.tau, plusminus = args.plusminus, 
                    verbose = args.verbose )

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():

    parser = argparse.ArgumentParser( description = 'Time Delay Embed' )

    parser.add_argument('-i', '--inputFile',
                        dest   = 'inputFile', type = str, 
                        action = 'store',     default = '',
                        help = 'Input file.')

    parser.add_argument('-o', '--outputFile',
                        dest   = 'outputFile', type = str, 
                        action = 'store',      default = None,
                        help = 'Output file name.')

    parser.add_argument('-p', '--plusminus',
                        dest   = 'plusminus',
                        action = 'store_true', default = False,
                        help = 'Both time-delay & time-advanced columns.')

    parser.add_argument('-c', '--columns', nargs = '*',
                        dest   = 'columns', type = str, 
                        action = 'store',   default = [],
                        help = 'columns.')

    parser.add_argument('-E', '--E',
                        dest   = 'E',     type = int, 
                        action = 'store', default = 2,
                        help = 'E.')

    parser.add_argument('-tau', '--tau',
                        dest   = 'tau',   type = int, 
                        action = 'store', default = -1,
                        help = 'tau.')

    parser.add_argument('-v', '--verbose',
                        dest   = 'verbose',
                        action = 'store_true', default = False,
                        help = 'verbose.')

    args = parser.parse_args()

    if args.plusminus and args.tau > 0 :
        print( 'tau changed to {} with --plusminus'.format( -args.tau ) )
        args.tau = -args.tau

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    Embedding_CmdLine() # Call CLI wrapper for Embedding()
