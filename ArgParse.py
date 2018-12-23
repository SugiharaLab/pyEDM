
from argparse import ArgumentParser

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    '''
    nargs = '*' All command-line args are gathered into a list.
    nargs = '+' All command-line args are gathered into a list, and,
                an error message generated if not at least one argument.
    '''
    
    parser = ArgumentParser( description = 'EDM' )
    
    parser.add_argument('-m', '--method',
                        dest   = 'method', type = str, 
                        action = 'store', default = 'Simplex',
                        help = 'Type of projection Simplex or SMap.')

    parser.add_argument('-p', '--prediction', nargs = '+',
                        dest   = 'prediction', type = int, 
                        action = 'store', default = [1, 10],
                        help = 'Prediction start/stop indices.')

    parser.add_argument('-l', '--library', nargs = '+',
                        dest   = 'library', type = int, 
                        action = 'store', default = [1, 10],
                        help = 'Library start/stop indices.')

    parser.add_argument('-E', '--EmbedDimension',
                        dest   = 'E', type = int, 
                        action = 'store', default = -1,
                        help = 'Embedding dimension.')

    parser.add_argument('-k', '--knn',
                        dest   = 'k_NN', type = int, 
                        action = 'store', default = -1,
                        help = 'Number of nearest neighbors.')

    parser.add_argument('-N', '--noNeighborLimit',
                        dest   = 'noNeighborLimit',
                        action = 'store_true', default = False,
                        help = "Don't limit neighbors based on Tp." )

    parser.add_argument('-T', '--Tp',
                        dest   = 'Tp', type = int, 
                        action = 'store', default = 0,
                        help = 'Forecast interval (0 default).')

    parser.add_argument('-t', '--theta',
                        dest   = 'theta', type = float, 
                        action = 'store', default = 0,
                        help = 'S-Map local weighting exponent (0 default).')

    parser.add_argument('-j', '--jacobians', nargs = '+',
                        dest   = 'jacobians',
                        action = 'store', default = [],
                        help = 'S-Map Jacobian columns, list of pairs.')

    parser.add_argument('-svd', '--SVDLeastSquares',
                        dest   = 'SVDLeastSquares',
                        action = 'store_true', default = False,
                        help = 'Use SVD least squares in S-Map.')
    
    parser.add_argument('-sig', '--SVDSignificance',
                        dest   = 'SVDSignificance', type = float, 
                        action = 'store', default = 0.00001,
                        help = 'S-Map SVD significance (10^-5 default).')

    parser.add_argument('-tr', '--TikhonovAlpha', 
                        dest   = 'TikhonovAlpha', type = float,
                        action = 'store', default = None,
                        help = 'Tikhonov regularisation initial alpha in ' +\
                               'S-Map SVD.')
    
    parser.add_argument('-en', '--ElasticNetAlpha', # ElasticNetCV l1_ratio
                        dest   = 'ElasticNetAlpha', type = float,
                        action = 'store', default = None,
                        help = 'Elastic Net alpha in S-Map.')
    
    parser.add_argument('-M', '--multiview',
                        dest   = 'multiview', type = int, 
                        action = 'store', default = 0,
                        help = 'Multiview ensemble size (sqrt(m) default).')

    parser.add_argument('-u', '--tau',
                        dest   = 'tau', type = int, 
                        action = 'store', default = 1,
                        help = 'Time delay (tau).')

    parser.add_argument('-f', '--forwardTau',
                        dest   = 'forwardTau', 
                        action = 'store_true', default = False,
                        help = 'Embed as t + tau instead of t - tau.')

    parser.add_argument('-c', '--columns', nargs = '*',
                        dest   = 'columns', type = str,
                        action = 'store', default = '',
                        help = 'Data or embedded data column names.')

    parser.add_argument('-r', '--target',
                        dest   = 'target', type = str,
                        action = 'store', default = None,
                        help = 'Data library target column name.')

    parser.add_argument('-e', '--embedded',
                        dest   = 'embedded',
                        action = 'store_true', default = False,
                        help = 'Input data is an embedding.')

    parser.add_argument('-L', '--libsize', nargs = '*',
                        dest   = 'libsize', type = int,
                        action = 'store',
                        default = [ 10, 80, 10 ],
                        help = 'CCM Library size range [start, stop, incr].')

    parser.add_argument('-s', '--subsample',
                        dest   = 'subsample', type = int, 
                        action = 'store',      default = 100,
                        help = 'Number subsamples generated at each library.')

    parser.add_argument('-R', '--randomLib',
                        dest   = 'randomLib', 
                        action = 'store_true', default = False,
                        help = 'CCM random library samples enabled.')

    parser.add_argument('-S', '--seed',
                        dest   = 'seed', type = int, 
                        action = 'store',      default = None,
                        help = 'Random number generator seed: (None default)')

    parser.add_argument('-pa', '--path',
                        dest   = 'path', type = str, 
                        action = 'store',      default = './data/',
                        help = 'Input & Output file path.')

    parser.add_argument('-i', '--inputFile',
                        dest   = 'inputFile', type = str, 
                        action = 'store',     default = None,
                        help = 'Input observation file.')

    parser.add_argument('-o', '--outputFile',
                        dest   = 'outputFile', type = str, 
                        action = 'store',      default = None,
                        help = 'Output prediction file.')
    
    parser.add_argument('-os', '--outputSmapFile',
                        dest   = 'outputSmapFile', type = str, 
                        action = 'store',      default = None,
                        help = 'S-map Output file.')
    
    parser.add_argument('-oe', '--outputEmbed',
                        dest   = 'outputEmbed', type = str, 
                        action = 'store',      default = None,
                        help = 'Output embedded data file.')
    
    parser.add_argument('-fs', '--figureSize', nargs = 2,
                        dest   = 'figureSize', type = float,
                        action = 'store', default = [ 5, 3 ],
                        help = 'Figure size (default [5, 3]).')
    
    parser.add_argument('-P', '--plot',
                        dest   = 'plot',
                        action = 'store_true', default = False,
                        help = 'Show plot.')
    
    parser.add_argument('-PX', '--plotXLabel',
                        dest   = 'plotXLabel', type = str,
                        action = 'store', default = 'Time ()',
                        help = 'Plot x-axis label.')
    
    parser.add_argument('-PY', '--plotYLabel',
                        dest   = 'plotYLabel', type = str,
                        action = 'store', default = 'Amplitude ()',
                        help = 'Plot y-axis label.')
    
    parser.add_argument('-PD', '--plotDate',  # Set automatically 
                        dest   = 'plotDate',
                        action = 'store_true', default = False,
                        help = 'Time values are pyplot datetime numbers.')
    
    parser.add_argument('-v', '--verbose',
                        dest   = 'verbose',
                        action = 'store_true', default = False,
                        help = 'Print status messages.')
    
    parser.add_argument('-w', '--warnings',
                        dest   = 'warnings',
                        action = 'store_true', default = False,
                        help = 'Show warnings.')
    
    parser.add_argument('-D', '--Debug',
                        dest   = 'Debug',
                        action = 'store_true', default = False,
                        help = 'Activate Debug messsages.')
    
    args = parser.parse_args()

    AdjustArgs( args )

    if args.Debug:
        print( 'ParseCmdLine()' )
        print( args )

    return args

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def AdjustArgs( args ):
    '''
    Index offsets and validation. 
    '''
    
    # If S-Map prediction, require k_NN > E + 1, default is all neighbors.
    # If Simplex and k_NN not specified, k_NN is set to E+1 in Prediction()
    if "smap" in args.method.lower() :
        if args.k_NN > 0:
            if args.k_NN <= args.E :
                raise RuntimeError( "ParseCmdLine() k_NN must be at least " +\
                                    " E+1 with method S-Map." )
        if  not args.embedded and len( args.columns ) > 1 :
            print( "ParseCmdLine() WARNING:  Multivariable S-Map should use" +\
                   " -e (embedded) data input to ensure data/dimension" +\
                   " correspondance." )

        # S-Map coefficient columns for jacobians start at 1 since the 0th
        # column is the S-Map linear prediction bias term
        if len( args.jacobians ) :
            if "0" in args.jacobians :            
                raise RuntimeError( "ParseCmdLine() S-Map coefficient columns"+\
                                    " for jacobians can not use column 0." )

        # Must be pairs of coefficient columns
        if len( args.jacobians ) % 2 :
            raise RuntimeError( "ParseCmdLine() S-Map coefficient columns " +\
                                "for jacobians must be in pairs." )

        # Convert args.jacobians into a list of int pairs (tuples)
        args.jacobians = [ ( int( args.jacobians[i]     ),
                             int( args.jacobians[i + 1] ) )
                           for i in range( 0, len( args.jacobians ), 2 ) ]

        # SVD, Tikhonov and ElasticNet are mutually exclusive
        if ( args.TikhonovAlpha and args.ElasticNetAlpha ) or \
           ( args.SVDLeastSquares and args.TikhonovAlpha ) or \
           ( args.SVDLeastSquares and args.ElasticNetAlpha ) :
            raise RuntimeError( "ParseCmdLine() Multiple S-Map solve methods "+\
                                "specified.  Use one of: Tikhonov (-tr)  " +\
                                "Elastic Net (-en)  SVD (-svd)." )

        if args.ElasticNetAlpha :
            # Since ArgParse.py a standalone module, check here for sklearn
            try:
                from sklearn.linear_model import ElasticNetCV
            except ImportError:
                raise RuntimeError( "ParseCmdLine() Failed to import " +\
                                    "ElasticNetCV from sklearn.linear_model."+\
                                    " See scikit-learn.org" )
            # sklearn.linear_model.ElasticNet documentation states:
            # l1_ratio <= 0.01 is not reliable
            if args.ElasticNetAlpha <= 0.01 :
                print( "ParseCmdLine() Setting ElasticNetAlpha to 0.01." )
                args.ElasticNetAlpha = 0.01
            if args.ElasticNetAlpha > 1 :
                print( "ParseCmdLine() Setting ElasticNetAlpha to 1." )
                args.ElasticNetAlpha = 1
            

    # Convert library and prediction indices to zero-offset
    args.prediction = [ x-1 for x in args.prediction ]
    args.library    = [ x-1 for x in args.library    ]
    
    # Python slice indexing start:stop:increment means < stop
    # so [0:9:1] returns 9 elements from 0 to 8, not 10 from 0 to 9
    args.prediction[-1] = args.prediction[-1] + 1
    args.library   [-1] = args.library   [-1] + 1

    # Convert figureSize to a tuple
    args.figureSize = tuple( args.figureSize )
