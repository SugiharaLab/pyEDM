#! /usr/bin/env python3

from argparse import ArgumentParser

from   scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

#----------------------------------------------------------------------------
# Main module
#----------------------------------------------------------------------------
def main():
    '''
    Create, plot and write out multidimensional data from the Rossler
    dynamical model.

    '''
    
    args = ParseCmdLine()
    
    Rossler( args )
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def Rossler( args ):
    '''
    Use scipy.integrate.odeint() to integrate the differential equation.

    Note: scipy docs say to use solve_ivp instead of odeint, but,
    it's interface doesn't support additional arguments to the 
    function to be integrated requiring a wrapper or lambda function,
    and even with that: I could not get it to work... Nice. 
    '''
    
    # initial state
    v0 = np.array( args.initial )

    t = np.arange( 0.0, args.T, args.dT )

    # odeint requires "extra" variables to be in a tuple with name matching
    # inside the derivative function, so make N, F explicit
    a, b, c = args.constants
    V = odeint( dRossler, v0, t, args = (a,b,c) )

    # Compute and store the derivatives
    dVdt      = np.zeros( V.shape )
    dVdt_cols = range( dVdt.shape[1] )

    for row in range( 0, dVdt.shape[0] ):
        # State variable derivatives dV/dt directly from dRossler()
        # using the integrated value at each timestep as the v0
        dVdt[ row, dVdt_cols ] = dRossler( V[ row, dVdt_cols ], 0, a,b,c )

    if len( args.jacobians ) :
        # Note that args.jacobians is a list of pairs (tuples)
        jacobians = np.zeros( ( V.shape[0], len( args.jacobians ) ) )
        
        for pair, col in zip( args.jacobians, range( len(args.jacobians) ) ) :
            d1 = np.gradient( V[ :, pair[0] ] )
            d2 = np.gradient( V[ :, pair[1] ] )
            jacobians[ :, col ] = d1 / d2

    # Create output matrix
    # t is an array, first cast as matrix and transpose to merge with x & dVdt
    output = np.concatenate( (np.asmatrix( t ).T, V, dVdt), 1 )
    if len( args.jacobians ) :
        output = np.concatenate( (output, jacobians), 1 )

    if args.outputFile:
        # Create the header
        header = 'Time'
        for dim in range( len( v0 ) ) :
            header = header + ',V' + str( dim + 1 )
        for dim in dVdt_cols :
            header = header + ',dV' + str( dim + 1 ) + '/dt'
        if len( args.jacobians ) :
            for pair in args.jacobians :
                header = header + ',∂V' + str(pair[0] + 1) +\
                                  '/∂V' + str(pair[1] + 1)

        np.savetxt( args.outputFile, output, fmt = '%.4f', delimiter = ',',
                    header = header, comments = '' )

    #------------------------------------------------------------
    # 3-D plot first three variables in args.dimensions
    if '3D' in args.plot :
      from mpl_toolkits.mplot3d import Axes3D
      fig3D = plt.figure()
      ax3D  = fig3D.gca(projection='3d')
      
      ax3D.plot( V[ :, args.dimensions[0] ],
                 V[ :, args.dimensions[1] ],
                 V[ :, args.dimensions[2] ] )
      
      ax3D.set_xlabel( '$x_{0:d}$'.format( args.dimensions[0] ) )
      ax3D.set_ylabel( '$x_{0:d}$'.format( args.dimensions[1] ) )
      ax3D.set_zlabel( '$x_{0:d}$'.format( args.dimensions[2] ) )
      plt.show()
    
    #------------------------------------------------------------
    # 2-D plot all variables in args.dimensions
    elif '2D' in args.plot :
      plotColors = [ 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 
                     'blue', 'green', 'red', 'cyan', 'magenta', 'black' ]
      fig, ax = plt.subplots()
      
      for d in args.dimensions:
        # output has Time inserted to column 0, add +1
        ax.plot( output[:,0], output[:,d+1],
                 color = plotColors[d], linewidth = 3 )

      ax.set( xlabel = 'index ()',
              ylabel = 'amplitude ()',
              title  = 'Rossler' )
      plt.show()

#----------------------------------------------------------------------------
#  State derivatives
#----------------------------------------------------------------------------
def dRossler( v, t, a, b, c ):
    '''
    dx/dt = -y - z
    dy/dt =  x + ay
    dz/dt =  b + z(x-c)
    '''
    dv = np.zeros( len( v ) )
    x,y,z = v
    
    # 
    dv[0] = -y - z
    dv[1] =  x + a*y  
    dv[2] =  b + z*(x-c)
    
    return dv

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = ArgumentParser( description = 'Rossler' )
    
    parser.add_argument('-i', '--initial', nargs = '+',
                        dest   = 'initial', type = float,
                        action = 'store', default = [1, 0, 1],
                        help = 'Initial state values.')

    parser.add_argument('-c', '--constants', nargs = '+',
                        dest   = 'constants', type = float,
                        action = 'store', default = [0.2, 0.2, 5.7],
                        help = 'Constants a, b, c.')

    parser.add_argument('-j', '--jacobians', nargs = '+',
                        dest   = 'jacobians',
                        action = 'store', default = [],
                        help = 'Variable Jacobian columns, list of pairs.')

    parser.add_argument('-T', '--T',
                        dest   = 'T', type = float, 
                        action = 'store',      default = 100.,
                        help = 'Max time.')

    parser.add_argument('-t', '--dT',
                        dest   = 'dT', type = float, 
                        action = 'store',      default = 0.025,
                        help = 'Time increment.')

    parser.add_argument('-o', '--outputFile',
                        dest   = 'outputFile', type = str, 
                        action = 'store', default = None,
                        help = 'Output file.')

    parser.add_argument('-P', '--plot',
                        dest   = 'plot', type = str,
                        action = 'store',      default = '3D',
                        help = '2D or 3D plot')

    parser.add_argument('-d', '--dimensions', nargs='+',
                        dest   = 'dimensions', type = int, 
                        action = 'store', default = [1, 2, 3],
                        help = 'Dimensions to 2D plot.')

    args = parser.parse_args()

    # Zero offset dimensions
    args.dimensions = [ d-1 for d in args.dimensions ]

    if len( args.jacobians ) :
        # Must be pairs of coefficient columns
        if len( args.jacobians ) % 2 :
            raise RuntimeError( "ParseCmdLine() column numbers " +\
                                "for jacobians must be in pairs." )

        # Convert args.jacobians into a list of int pairs (tuples)
        # Offset columns to zero-offset to use in matrix x
        args.jacobians = [ ( int( args.jacobians[i]     ) - 1,
                             int( args.jacobians[i + 1] ) - 1 )
                           for i in range( 0, len( args.jacobians ), 2 ) ]

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation
# main() will only be called 'automatically' when the script is passed
# as the argument to the Python interpreter, not when imported as a module.
#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
