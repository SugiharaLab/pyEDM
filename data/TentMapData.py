#! /usr/bin/env python3

#import sys, string, time
from array    import array
from argparse import ArgumentParser
from random   import seed, random, uniform

import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Main module
#----------------------------------------------------------------------------
def main():
    
    args = ParseCmdLine()
    
    TentMap( args )

#----------------------------------------------------------------------------
# Tent Map data according to Sugihara, Nature 344, p 734 (1990).
#   x_t+1 = 2 x_t      : 0   < x_t < 0.5
#   x_t+1 = 2 - 2 x_t  : 0.5 < x_t < 1
#   Δt    = x_t+1 - x_t
#
# An issue with this sequence is that it quickly converges to 0.
# Once it reaches 0, the sequence is stable.  To avoid this, the -n
# option adds a small noise input to the sequence values. 
#----------------------------------------------------------------------------
def TentMap( args ):

    x = array( 'd', [0] * args.N ) # Type 'd' is double (min 8 bytes)

    seed()
    x[0] = random() # float in [0,1]

    print( 'x[0] = {0:6.4f}'.format( x[0] ) )

    # Compute Tent Map values
    for i in range( args.N - 1 ) :
        if x[ i ] < 0.5 :
            x[ i + 1 ] = 2 * x[ i ]
        elif x[ i ] > 0.5 :
            x[ i + 1 ] = 2 - 2 * x[ i ]
        else :
            # Sequence has iterated to 0.5 : 1 : 0. reseed the value
            #x[ i + 1 ] = 0.5 + uniform(-100, 100) / 10000
            x[ i + 1 ] = random()
            print( "End value (", x[i], ") at i=", i )

        if args.noise :
            x[ i + 1 ] = x[ i + 1 ] + random() / 1000

    if args.difference :
        for i in range( 1, args.N - 1 ) :
            x[ i ] = x[ i + 1 ] - x[ i ]

    with open( args.outputPath + args.outputFile, 'w' ) as fob:
        fob.write( 'Time, ' + str( args.N ) + '\n' )
        for i in range( args.N ) :
            fob.write( '{0:d}, {1:8.6f}\n'.format(i+1, x[i]) )
            
    #---------------------------------------------------
    if args.plot :
        fig, ax = plt.subplots()
        ax.plot( range( args.N ), x )
    
        ax.set( xlabel = 'index ()',
                ylabel = 'amplitude ()',
                title  = 'Tent Map' )
        plt.show()

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = ArgumentParser( description = 'Tent Map Data' )
    
    parser.add_argument('-N', '--N',
                        dest   = 'N', type = int, 
                        action = 'store',     default = 500,
                        help = 'Number of points.')
    
    parser.add_argument('-d', '--difference',
                        dest   = 'difference',
                        action = 'store_false', default = True,
                        help = 'Disable perform first difference.')
    
    parser.add_argument('-n', '--noise',
                        dest   = 'noise',
                        action = 'store_false', default = True,
                        help = 'Disable additive noise.')
    
    parser.add_argument('-p', '--outputPath',
                        dest   = 'outputPath', type = str, 
                        action = 'store',      default = './',
                        help = 'outputPath')

    parser.add_argument('-o', '--outputFile',
                        dest   = 'outputFile', type = str, 
                        action = 'store',      default = 'TentMapData.csv',
                        help = 'outputFile')

    parser.add_argument('-P', '--plot',
                        dest   = 'plot',
                        action = 'store_true', default = False,
                        help = 'Show plot.')
    
    args = parser.parse_args()

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation
# main() will only be called 'automatically' when the script is passed
# as the argument to the Python interpreter, not when imported as a module.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
