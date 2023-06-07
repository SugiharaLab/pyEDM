#! /usr/bin/env python3

import argparse

from pyEDM  import Embed
from pandas import concat, read_csv

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def main():
    '''
    EDM Embed wrapper
    Create time-delay embedding with time column for EDM. 
    Useful to create mixed multivariate embeddings for SMap and
    embeddings with time-advanced vectors.  
    Presume negative tau. Rename V(t-0) to V. Add Time column.
    If args.forward create time-advanced columns, remove V(t+0).
    '''

    args = ParseCmdLine()

    data = read_csv( args.inputFile )

    # Presume time is first column
    timeName   = data.columns[0]
    timeSeries = data[ timeName ]

    # If no columns specified, use all except first
    if not args.columns :
        args.columns = data.columns[ 1: ]

    if args.verbose :
        print( "Input time column: ", timeName )
        print( "Input columns: ", args.columns )

    # Create embeddings of columns
    # There will be redundancies vis V1(t-0), V1(t+0)
    if args.forward :
        embed_minus = Embed( dataFrame = data, E = args.E, tau = args.tau,
                             columns = args.columns )
        embed_plus = Embed( dataFrame = data, E = args.E, tau = abs( args.tau ),
                            columns = args.columns )
        df = concat( [ timeSeries, embed_minus, embed_plus ], axis = 1 )

        # Remove *(t+0) columns redundant with *(t-0)
        cols_tplus0 = [ c for c in df.columns if '(t+0)' in c ]
        df = df.drop( columns = cols_tplus0 )

    else :
        embed_ = Embed( dataFrame = data, E = args.E, tau = args.tau,
                        columns = args.columns )
        df = concat( [ timeSeries, embed_ ], axis = 1 )

    # Rename *(t-0) to original column names
    cols_tminus0 = [ c for c in df.columns if '(t-0)' in c ]
    cols_orig    = [ c.replace( '(t-0)', '', 1 ) for c in cols_tminus0 ]
    df.rename( columns = dict( zip( cols_tminus0, cols_orig ) ), inplace = True )

    # Rename first column to original time column name
    df.rename( columns = { df.columns[0] : timeName }, inplace = True )

    if args.verbose :
        print( "Columns:", df.columns, "\n-------------------------------" )
        print( df.head( 4 ) )

    if args.outputFile :
        df.to_csv( args.outputFile, index = False )

#----------------------------------------------------------------------------
def ParseCmdLine():

    parser = argparse.ArgumentParser( description = 'Time Delay Embed' )

    parser.add_argument('-i', '--inputFile',
                        dest   = 'inputFile', type = str, 
                        action = 'store',     default = '',
                        help = 'Input file.')

    parser.add_argument('-o', '--outputFile',
                        dest   = 'outputFile', type = str, 
                        action = 'store',      default = 'out.csv',
                        help = 'Output file name.')

    parser.add_argument('-f', '--forward',
                        dest   = 'forward',
                        action = 'store_true', default = False,
                        help = 'Add time-advanced columns.')

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

    if args.tau > 0 :
        # Presume time-delay embedding
        args.tau = -args.tau
        print( "tau changed to {}".format( args.tau ) )

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    main()
