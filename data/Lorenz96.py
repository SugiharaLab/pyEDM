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
    Create, plot and write out multidimensional data from the Lorenz '96
    dynamical model.

    From Lorenz '96:

    N variables k=1, ... N, governed by N equations:
           d X_k / dt = -X_k-2 * X_k-1 + X_k-1 * X_k+1 - X_k + F
                      = (X_k+1 - X_k-2) * X_k-1 - X_k + F

    We assume N > 3; the dynamics are of little interest otherwise.

    For very small values of F, all solutions converge to the steady state
    solution X = F. For larger F most solutions are periodic, but for still
    larger values of F (dependent on K) chaos ensues. For N = 36 and F = 8
    λ_1 corresponds to a doubling time of 2.1 days. If F is 10 the time drops
    to 1.5 days. ... this scaling makes time unit equal to 5 days. With a
    time step of Δt = 0.05 units, or 6 hours. 

    --
    Lorenz, Edward (1996). Predictability – A problem partly solved,
    Seminar on Predictability, Vol. I, ECMWF

    http://www.math.colostate.edu/~gerhard/MATH540/FILES/
    Karimi2010_ChaosInLorenz96.pdf
    '''
    
    args = ParseCmdLine()
    
    Lorenz96( args )
    
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def Lorenz96( args ):
    '''
    Use scipy.integrate.odeint() to integrate the differential equation.

    Note: scipy docs say to use solve_ivp instead of odeint, but,
    it's interface doesn't support additional arguments to the 
    function to be integrated requiring a wrapper or lambda function,
    and even with that: I could not get it to work... Nice. 
    '''
    
    # initial state (equilibrium)
    v0 = args.forceConstant * np.ones( args.nVariables )

    v0[ args.i_perturb ] += args.perturb # add small perturbation

    t = np.arange( 0.0, args.T, args.dT )

    # odeint requires "extra" variables to be in a tuple with name matching
    # inside the derivative function, so make N, F explicit
    N = args.nVariables
    F = args.forceConstant
    V = odeint( dLorenz96, v0, t, args = (N, F) )

    # Compute and store the derivatives
    dVdt      = np.zeros( V.shape )
    dVdt_cols = range( dVdt.shape[1] )

    for row in range( 0, dVdt.shape[0] ):
        # State variable derivatives dV/dt directly from Lorenz96()
        # using the integrated value at each timestep as the v0
        dVdt[ row, dVdt_cols ] = dLorenz96( V[ row, dVdt_cols ], 0, N, F )

    if len( args.jacobians ) :
        # Note that args.jacobians is a list of pairs (tuples)
        jacobians = np.zeros( ( V.shape[0], len( args.jacobians ) ) )
        
        for pair, col in zip( args.jacobians, range( len(args.jacobians) ) ) :
            d1 = np.gradient( V[ :, pair[0] ] )
            d2 = np.gradient( V[ :, pair[1] ] )
            jacobians[ :, col ] = d1 / d2

    # Create output matrix
    # get index of starting point to exclude transient start
    exclude = int( np.where( t == args.exclude )[0] )
    
    # t is an array, first cast as matrix and transpose to merge with x & dVdt
    output = np.concatenate( (np.asmatrix( t ).T, V, dVdt), 1 )
    if len( args.jacobians ) :
        output = np.concatenate( (output, jacobians), 1 )
    output = output[ exclude::, ]

    if args.outputFile:
        # Create the header
        header = 'Time'
        for dim in range( args.nVariables ) :
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
      
      ax3D.plot( V[ exclude::, args.dimensions[0] ],
                 V[ exclude::, args.dimensions[1] ],
                 V[ exclude::, args.dimensions[2] ] )
      
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
              title  = 'Lorenz 96' )
      plt.show()

#----------------------------------------------------------------------------
#  State derivatives
#  Relies on Python array wrapping for negative/overflow indicies
#----------------------------------------------------------------------------
def dLorenz96( v, t, N, F ):
    '''
    N variables k=1, ... N, governed by N equations:
           d X_k / dt = (X_k+1 - X_k-2) * X_k-1 - X_k + F
    Assume N > 3
    '''
    dv = np.zeros(N)

    # first the 3 edge cases: i=1,2,N
    dv[0]   = (v[1] - v[N-2]) * v[N-1] - v[0]
    dv[1]   = (v[2] - v[N-1]) * v[0]   - v[1]
    dv[N-1] = (v[0] - v[N-3]) * v[N-2] - v[N-1]
    
    # then the general case
    for i in range(2, N-1):
        dv[i] = (v[i+1] - v[i-2]) * v[i-1] - v[i]

    # add the forcing term
    dv = dv + F

    return dv

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    
    parser = ArgumentParser( description = 'Lorenz 96' )
    
    parser.add_argument('-D', '--nVariables',
                        dest   = 'nVariables', type = int, 
                        action = 'store',     default = 5,
                        help = 'Number of variables.')
    
    parser.add_argument('-j', '--jacobians', nargs = '+',
                        dest   = 'jacobians',
                        action = 'store', default = [],
                        help = 'Variable Jacobian columns, list of pairs.')

    parser.add_argument('-f', '--forceConstant',
                        dest   = 'forceConstant', type = int, 
                        action = 'store',      default = 8,
                        help = 'Forcing constant.')

    parser.add_argument('-p', '--perturb',
                        dest   = 'perturb', type = float, 
                        action = 'store',      default = 0.01,
                        help = 'Pertubation value.')

    parser.add_argument('-i', '--iPerturb',
                        dest   = 'i_perturb', type = int, 
                        action = 'store',      default = 3,
                        help = 'Pertubation index.')

    parser.add_argument('-T', '--T',
                        dest   = 'T', type = float, 
                        action = 'store',      default = 60.,
                        help = 'Max time.')

    parser.add_argument('-t', '--dT',
                        dest   = 'dT', type = float, 
                        action = 'store',      default = 0.05,
                        help = 'Time increment.')

    parser.add_argument('-x', '--exclude',
                        dest   = 'exclude', type = float, 
                        action = 'store',      default = 10.,
                        help = 'Initial transient time to exclude.')

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

    if args.i_perturb >= args.nVariables :
        print( "i_perturb > D, setting to 1" )
        args.i_perturb = 1

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
