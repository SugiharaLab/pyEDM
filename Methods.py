# Python distribution modules
from multiprocessing import Pool
from copy            import deepcopy
from collections     import OrderedDict
from itertools       import combinations

# Community modules
import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.dates import num2date
from   numpy.random     import randint, seed

# Local modules
from ArgParse import ParseCmdLine
from EDM      import EmbedData, ReadEmbeddedData, Prediction,    \
                     SimplexProjection, Distance, DistanceMetric,\
                     ComputeError, nCol, nRow

# Global enum as a class
class Source :
    Python, Jupyter = range( 2 )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def Predict( args, data = None, colNames = None, target = None,
             source = Source.Python ):
    '''
    Data input/embedding wrapper for EDM.Prediction() to compute:

      Simplex projection of observational data (Sugihara, 1990), or
      SMap    projection of observational data (Sugihara, 1994).

    There are two options for data file input, or an embedding can be
    passed in directly (data, colNames, target).

    If --embedding (-e) is specified, it is assumed that the data file
    or data input is already an embedding or multivariable data matrix.
    Otherwise, the data is embedded by EmbedData(). 

    If --embedding (-e) is specified and the data input parameter is None, 
    then the -i (inputFile) is processed by ReadEmbeddedData() which assumes
    the files consists of a .csv file formatted as:

       [ Time, Dim_1, Dim_2, ... ] 

    where Dim_1 is observed data, Dim_2 data offset by τ, Dim_3 by 2τ...
    The user can specify the desired embedding dimension E, which
    can be less than the total number of columns in the inputFile. 
    The first E + 1 columns (Time, D1, D2, ... D_E) will be returned.

    Alternatively, the data can be a .csv file with multiple simultaneous
    observations or delay embeddings (columns) where the columns to 
    embed and target to project are specified with the -c (columns)
    and -r (target) options. In all cases 'time' is required in column 0. 
 
    Embedding can be done with EDM.EmbedData() via the wrapper Embed.py. 
    Note: The embedded data .csv file will have fewer rows (observations)
    than the data input to EmbedData() by E - 1. 
    '''

    if args.embedded :
        if data is None :
            # args.inputFile is an embedding or multivariable data frame.
            # ReadEmbeddedData() sets args.E to the number of columns
            # if the -c (columns) and -t (target) options are used.
            embedding, colNames, target = ReadEmbeddedData( args )
        else:
            # Data matrix is passed in as parameter, no embedding needed
            embedding = data
            # target taken as-is from input parameters
    else :
        # args.inputFile are timeseries data to be embedded by EmbedData
        embedding, colNames, target = EmbedData( args, data, colNames )

    rho, rmse, mae, header, output, smap_output = Prediction( embedding,
                                                              colNames,
                                                              target, args )
    if source == Source.Jupyter :
        return { 'rho':rho, 'RMSE':rmse, 'MAE':mae, 'header':header,
                 'prediction':output, 'S-map':smap_output }
    else:
        return
    
#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def Embed( args, data = None, colNames = None, source = Source.Python ):
    '''
    Wrapper for EDM.EmbedData()

    Time-delay embedd data vector(s) from args.inputFile into 
    args.Dimensions at multiples of args.tau.  Note that if the 
    -f --forwardTau option is specified, then the embedding is 
    x(t) + τ instead of x(t) - τ.

    The -e (embedColumns) option specifies the zero-offset column
    numbers or column names to embed from args.inputFile.

    Writes a .csv file with header [Time, Dim_1, Dim_2...] if -o specified.

    Note: The output .csv file will have fewer rows (observations)
    than the input data by args.Dimension - 1 (E-1). 
    '''
    
    embedding, header, target = EmbedData( args, data, colNames )

    if args.Debug:
        print( "Embed() " + ' '.join( args.embedColumns ) +\
               " from " + args.inputFile +\
               " E=" + str( args.E ) + " " +\
               str( embedding.shape[0] ) + " rows,  " +\
               str( embedding.shape[1] ) + " columns." )
        
        print( header )
        print( embedding[ 0:3, : ] )
        
    if source == Source.Jupyter :
        return { 'header':header, 'embedding':embedding, 'target':target }
    else:
        return

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def EmbedDimensions( args, source = Source.Python ):
    '''
    Using ParseCmdLine() arguments, override E and k_NN to evaluate 
    embeddings for E = 1 to 10.

    There are two options for data input. One is to use the -c (columns)
    argument so that the -i (inputFile) will be considered a timeseries with 
    embeddings dynamically performed by EmbedData() for each evaluation.  
    The other is to specify -e (embedded) so that -i (inputFile) specifies 
    a .csv file with an embedding or multivariables already in place.  The
    vector in the second column (j=1) will be considered the observed data.
    This will be read by ReadEmbeddedData().

    Prediction() sets k_NN equal to E + 1 if -k not specified and method 
    is Simplex.
    '''
    
    # Save args.plot flag, but disable so Prediction() does not plot
    showPlot  = args.plot
    args.plot = False

    # Process pool
    pool = Pool()
    
    # Create iterable with args variants for E = 1 to 10
    argsList = []
    for E in range( 1, 11 ) :
        newArgs      = deepcopy( args )
        newArgs.E    = E
        newArgs.k_NN = E + 1
        argsList.append( newArgs )

    # Submit EmbedPredict jobs to the process pool
    results = pool.map( EmbedPredict, argsList )
    
    E_rho = {} # Dict to hold E : rho pairs from EmbedPredict() tuple

    for result in results :
        if result == None:
            continue
        
        E_rho[ result[ 0 ] ] = result[ 1 ]

    # Console output
    print( "{:<5} {:<10}".format('E','ρ') )
    for E_, rho_ in E_rho.items():
        print( "{0:<5} {1:<10}".format( E_, rho_ ) )

    #----------------------------------------------------------
    if showPlot:
        
        fig, ax = plt.subplots( 1, 1, figsize = args.figureSize, dpi = 150 )
        
        ax.plot( E_rho.keys(), E_rho.values(),
                 label = 'Predictions_t(+{0:d})'.format( args.Tp ),
                 color='blue', linewidth = 3 )
        
        ax.set( xlabel = 'Embedding Dimension',
                ylabel = 'Prediction Skill' + r' $\rho$',
                title  = args.inputFile +\
                         ' Tp=' + str( args.Tp ) )
        plt.show()

    if source == Source.Jupyter :
        return E_rho
    else:
        return

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def EmbedPredict( args ):
    '''
    Pool worker function called from EmbedDimensions()
    '''
    
    # if -e has been specified: use ReadEmbeddedData()
    # ReadEmbeddedData() sets args.E to the number of columns specified
    # if the -c (columns) and -t (target) options are used, otherwise
    # it uses args.E to read E columns.
    if args.embedded :
        # Set args.E so at least 10 dimensions are read.
        E      = args.E
        args.E = 10
        embedding, colNames, target = ReadEmbeddedData( args )
        # Reset args.E for Prediction
        args.E = E
    
    else :
        # -e not specified, embed on each iteration
        embedding, colNames, target = EmbedData( args )
        
    rho, rmse, mae, header, output, smap_output = Prediction( embedding,
                                                              colNames,
                                                              target, args )
    return tuple( ( args.E, round( rho, 3 ) ) )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PredictDecays( args, source = Source.Python ):
    '''
    Using ParseCmdLine() arguments, override Tp to evaluate Tp = 1 to 10.

    There are two options for data input. One is to use the -c (columns)
    argument so that the -i (inputFile) will be considered a timeseries with 
    embeddings dynamically performed by EmbedData() for each evaluation.  
    The other is to specify -e (embedded) so that -i (inputFile) specifies 
    a .csv file with an embedding or multivariables already in place.  The
    vector in the second column (j=1) will be considered the observed data.
    This will be read by ReadEmbeddedData().

    Prediction() sets k_NN equal to E+1 if -k not specified and method 
    is Simplex.
    '''
    
    # Save args.plot flag, but disable so Prediction() does not plot
    showPlot  = args.plot
    args.plot = False

    # if -e has not been specified: use EmbedData()
    if not args.embedded :
        embedding, colNames, target = EmbedData( args )
    else :
        # ReadEmbeddedData() sets args.E to the number of columns specified
        # if the -c (columns) and -t (target) options are used, otherwise
        # it uses args.E to read E columns. 
        embedding, colNames, target = ReadEmbeddedData( args )

    # Process pool
    pool = Pool()
    
    # Create iterable with args variants for Tp = 1 to 10
    argsEmbeddingList = []
    for T in range( 1, 11 ) :
        newArgs    = deepcopy( args )
        newArgs.Tp = T
        # Add the embedding, colNames, target in a tuple
        argsEmbeddingList.append( ( newArgs, embedding, \
                                    colNames, target, 'Tp' ) )

    # Submit PredictFunc jobs to the process pool
    results = pool.map( PredictFunc, argsEmbeddingList )
    
    Tp_rho = {} # Dict to hold Tp : rho pairs from PredictFunc() tuple

    for result in results :
        if result == None:
            continue
        
        Tp_rho[ result[ 0 ] ] = result[ 1 ]

    # Console output
    print( "{:<5} {:<10}".format('Tp','ρ') )
    for T_, rho_ in Tp_rho.items():
        print( "{0:<5} {1:<10}".format( T_, rho_ ) )

    #----------------------------------------------------------
    if showPlot:
        
        fig, ax = plt.subplots( 1, 1, figsize = args.figureSize, dpi = 150 )
        
        ax.plot( Tp_rho.keys(), Tp_rho.values(),
                 label = 'Predictions_t(+{0:d})'.format( args.Tp ),
                 color='blue', linewidth = 3 )
        
        ax.set( xlabel = 'Forecast time Tp',
                ylabel = 'Prediction Skill' + r' $\rho$',
                title  = args.inputFile +\
                         ' E=' + str( args.E ) )
        plt.show()

    if source == Source.Jupyter :
        return Tp_rho
    else:
        return
    
#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PredictFunc( argsEmbedding ) :
    '''
    Pool worker function called from PredictDecays() or SMapNL()
    '''

    args       = argsEmbedding[ 0 ]
    embedding  = argsEmbedding[ 1 ]
    colNames   = argsEmbedding[ 2 ]
    target     = argsEmbedding[ 3 ]
    outputType = argsEmbedding[ 4 ]

    # Prediction does not currently use colNames
    rho, rmse, mae, header, output, smap_output = Prediction( embedding,
                                                              colNames,
                                                              target, args )
    if 'Tp' in outputType:
        return tuple( ( args.Tp, round( rho, 3 ) ) )
    elif 'theta' in outputType:
        return tuple( ( args.theta, round( rho, 3 ) ) )
    else:
        raise Exception( 'PredictFunc() Invalid output specified.' )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def SMapNL( args, data = None, colNames = None, target = None, thetas = None,
            source = Source.Python ):
    '''
    Using ParseCmdLine() arguments, override the -t (theta) to evaluate 
    theta = 0.01 to 9.

    There are two options for data file input. One is to use the -c (columns)
    argument so that the -i (inputFile) will be considered a timeseries with 
    embeddings dynamically performed by EmbedData() for each evaluation.  
    The other is to specify -e (embedded) so that -i (inputFile) specifies 
    a .csv file with an embedding or multivariables already in place.  The
    vector in the second column (j=1) will be considered the observed data.
    This will be read by ReadEmbeddedData().

    Data can also be passed in (data, colNames, target) instead of read
    from a file. 
    '''
    
    args.method = 'smap'

    # Save args.plot flag, but disable so Prediction() does not plot
    showPlot  = args.plot
    args.plot = False

    if args.embedded :
        if data is None :
            # args.inputFile is an embedding or multivariable data frame.
            # ReadEmbeddedData() sets args.E to the number of columns
            # if the -c (columns) and -t (target) options are used.
            embedding, colNames, target = ReadEmbeddedData( args )
        else:
            # Data matrix is passed in as parameter, no embedding needed
            embedding = data
            # target taken as-is from input parameters
    else :
        # args.inputFile are timeseries data to be embedded by EmbedData
        embedding, colNames, target = EmbedData( args, data, colNames )

    if thetas is None :
        # Evaluate theta localization parameter from 0.01 to 9
        Theta = [ 0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9 ]
    else :
        if len( thetas ) < 1 :
            raise Exception( 'SMapNL() theta must have at least one value.' )
        Theta = thetas

    # Process pool
    pool = Pool()
    
    # Create iterable with args variants for theta
    argsEmbeddingList = []
    for theta in Theta :
        newArgs       = deepcopy( args )
        newArgs.theta = theta
        # Add the embedding, colNames, target in a tuple
        argsEmbeddingList.append( ( newArgs, embedding,
                                    colNames, target, 'theta' ) )

    # Submit PredictFunc jobs to the process pool
    results = pool.map( PredictFunc, argsEmbeddingList )
    
    # Dict to hold theta : rho pairs from PredictFunc() tuple
    theta_rho = OrderedDict()

    for result in results :
        if result == None:
            continue
        theta_rho[ result[ 0 ] ] = result[ 1 ]

    # Console output
    print( "{:<5} {:<10}".format('θ','ρ') )
    for theta_, rho_ in theta_rho.items():
        print( "{0:<5} {1:<10}".format( theta_, rho_ ) )

    #----------------------------------------------------------
    if showPlot:
        
        fig, ax = plt.subplots( 1, 1, figsize = args.figureSize, dpi = 150 )
        
        ax.plot( theta_rho.keys(), theta_rho.values(),
                 label = 'Predictions_t(+{0:d})'.format( args.Tp ),
                 color='blue', linewidth = 3 )
        
        ax.set( xlabel = 'S Map Localization θ',
                ylabel = 'Prediction Skill' + r' $\rho$',
                title  = args.inputFile + ' Tp=' + str( args.Tp ) +\
                         ' E=' + str( args.E ) )
        plt.show()

    if source == Source.Jupyter :
        return theta_rho
    else:
        return

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def Multiview( args, source = Source.Python ):
    '''
    Data input requires -c (columns) to specify timeseries columns
    in inputFile (-i) that will be embedded by EmbedData(), and the 
    -r (target) specifying the data target column in inputFile.

    args.E represents the number of variables to combine for each
    assessment, as well as the number of time delays to create in 
    EmbedData() for each variable. 

    Prediction() with Simplex sets k_NN equal to E+1 if -k not specified.

    --
    Ye H., and G. Sugihara, 2016. Information leverage in interconnected 
    ecosystems: Overcoming the curse of dimensionality. 
    Science 353:922–925.
    '''

    if not len( args.columns ) :
        raise RuntimeError( 'Multiview() requires -c to specify data.' )
    if not args.target :
        raise RuntimeError( 'Multiview() requires -r to specify target.' )
    if args.E < 0 :
        raise RuntimeError( 'Multiview() E is required.' )

    # Save args.plot flag, and disable so Prediction() does not plot
    showPlot  = args.plot
    args.plot = False

    # Save args.outputFile and reset so Prediction() does not write 
    outputFile      = args.outputFile
    args.outputFile = None

    # Embed data from inputFile
    embedding, colNames, target = EmbedData( args )

    # Combinations of possible embedding variables (columns), E at-a-time
    # Column 0 is time. Coerce the iterable into a list of E-tuples
    nVar   = len( args.columns )
    combos = list( combinations( range( 1, nVar * args.E + 1 ), args.E ) )

    # Require that each embedding has at least one coordinate with
    # observed data (zero time lag). This corresponds to combo tuples
    # with modulo E == 1.
    # Note: this only works if the data (unlagged) are in columns
    # 1, 1 + E, 1 + 2E, ... which is consistent with EmbedData() output.
    combo_i = []
    for i in range( len( combos ) ) :
        c = combos[i] # a tuple of combination indices
        for x in c:
            if x % args.E == 1:
                combo_i.append( i )
                break

    combos = [ combos[i] for i in combo_i ]

    if not args.multiview :
        # Ye & Sugihara suggest sqrt( m ) as the number of embeddings to avg
        args.multiview = max( 2, int( np.sqrt( len( combos ) ) ) )
        
        print( 'Multiview() Set view sample size to ' + str( args.multiview ))
        
    #---------------------------------------------------------------
    # Evaluate variable combinations.
    # Note that this is done within the library itself (in-sample).
    # Save a copy of the specified prediction observations.
    prediction = args.prediction

    # Override the args.prediction for in-sample forecast skill evaluation
    args.prediction = args.library
    
    # Process pool to evaluate combos
    pool = Pool()

    # Iterable list of arguments for EvalLib()
    argList = []
    for combo in combos :
        argList.append( ( args, combo, embedding, colNames, target ) )
    
    # Submit EvalLib jobs to the process pool
    results = pool.map( EvalLib, argList )
    
    # Dict to hold combos : rho pairs from EvalLib() tuple
    Combo_rho = {}

    for result in results :
        if result == None:
            continue
        Combo_rho[ result[ 0 ] ] = result[ 1 ]

    #---------------------------------------------------------------
    # Rank the in-sample forecasts, zip returns an iterator of 1-tuples
    rho_sort, combo_sort = zip( *sorted( zip( Combo_rho.values(),
                                              Combo_rho.keys() ),
                                         reverse = True ) )
    
    if args.Debug:
        print( "Multiview()  In sample sorted embeddings:" )
        print( 'Columns         ρ' )
        for i in range( min( args.multiview, len( combo_sort ) ) ):
            print(str( combo_sort[i] ) + "    " + str( round( rho_sort[i],4)))
    
    #---------------------------------------------------------------
    # Perform predictions with the top args.multiview embeddings
    # Reset the user specified prediction vector
    args.prediction = prediction
    
    argList.clear() # Iterable list of arguments for EvalPred()

    # Take the top args.multiview combos
    for combo in combo_sort[ 0: args.multiview ] :
        argList.append( ( args, combo, embedding, colNames, target ) )
    
    # Submit EvalPred jobs to the process pool
    results = pool.map( EvalPred, argList )
    
    Results = OrderedDict() # Dictionary of dictionaries results each combo

    for result in results :
        if result == None:
            continue 
        Results[ result[ 0 ] ] = result[ 1 ]

    # Console output
    print( "Multiview()  Prediction Embeddings:" )
    print( "Columns       Names                       ρ       mae   rmse" )
    for key in Results.keys() :
        result = Results[ key ]
        print( str( key ) + "   " + ' '.join( result[ 'names' ] ) +\
               "  " + str( round( result[ 'rho'  ], 4 ) ) +\
               "  " + str( round( result[ 'mae'  ], 4 ) ) +\
               "  " + str( round( result[ 'rmse' ], 4 ) ) )

    #----------------------------------------------------------
    # Compute Multiview averaged prediction
    # The output item of Results dictionary is a matrix with three
    # columns [ Time, Data, Prediction_t() ]
    # Collect the Predictions into a single matrix
    aresult = Results[ combo_sort[0] ]
    nrows   = nRow( aresult['output'] )
    time    = aresult['output'][:,0]
    data    = aresult['output'][:,1]
    
    M = np.zeros( ( nrows, len( Results ) ) )

    col_i = 0
    for result in Results.values() :
        output = result[ 'output' ]
        M[ :, col_i ] = output[ :, 2 ] # Prediction is in col j=2
        col_i = col_i + 1

    prediction    = np.mean( M, axis = 1 )
    multiview_out = np.column_stack( ( time, data, prediction ) )

    # Write output
    header = 'Time,Data,Prediction_t(+{0:d})'.format( args.Tp )
    if outputFile:
        np.savetxt( args.path + outputFile, multiview_out, fmt = '%.4f',
                    delimiter = ',', header = header, comments = '' )

    # Estimate correlation coefficient on observed : predicted data
    rho, r, rmse, mae = ComputeError( data, prediction )

    print( ("Multiview()  ρ {0:5.3f}  r {1:5.3f}  RMSE {2:5.3f}  "
            "MAE {3:5.3f}").format( rho, r, rmse, mae ) )
    
    #----------------------------------------------------------
    if showPlot:
        
        Time = multiview_out[ :, 0 ] # Required to be first (j=0) column

        if args.plotDate :
            Time = num2date( Time )
        
        fig, ax = plt.subplots( 1, 1, figsize = args.figureSize, dpi = 150 )
        
        ax.plot( Time, multiview_out[ :, 1 ],
                 label = 'Observations',
                 color='blue', linewidth = 2 )
        
        ax.plot( Time, multiview_out[ :, 2 ],
                 label = 'Predictions_t(+{0:d})'.format( args.Tp ),
                 color='red', linewidth = 2 )

        if args.verbose :  # Plot all projections
            for col in range( nCol( M ) ) :
                ax.plot( multiview_out[ :, 0 ], M[ :, col ],
                         label = combo_sort[col], linewidth = 2 )
        
        ax.legend()
        ax.set( xlabel = args.plotXLabel,
                ylabel = args.plotYLabel,
                title  = "Multiview  " + args.inputFile +\
                         ' Tp=' + str( args.Tp ) +\
                         ' E='  + str( args.E ) + r' $\rho$=' +\
                str( round( rho, 2 ) ) )
        plt.show()

    if source == Source.Jupyter :
        return { 'header':header, 'multiview':multiview_out,
                 'rho':rho, 'r':r, 'RMSE':rmse, 'MAE':mae }
    else:
        return

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def EvalLib( argsList ) :
    '''
    Function for multiprocessing of combo evaluation within library
    argsList : ( args, combo, embedding, colNames, target )
    '''
    args      = argsList[ 0 ]
    combo     = argsList[ 1 ]
    embedding = argsList[ 2 ]
    colNames  = argsList[ 3 ]
    target    = argsList[ 4 ]
    
    # Extract the variable combination
    # Note that we prepend the time column (0,) as Prediction() requires
    embed = np.take( embedding, (0,) + combo, axis = 1 )
    
    # Evaluate prediction skill
    rho, rmse, mae, header, output, smap_output = Prediction( embed,
                                                              colNames,
                                                              target, args )
    
    return tuple( ( combo, round( rho, 5 ) ) )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def EvalPred( argsList ) :
    '''
    Function for multiprocessing of combo evaluation outside library
    argsList : ( args, combo, embedding, colNames, target )
    '''
    args      = argsList[ 0 ]
    combo     = argsList[ 1 ]
    embedding = argsList[ 2 ]
    colNames  = argsList[ 3 ]
    target    = argsList[ 4 ]
        
    # Extract the variable combination
    # Note that we prepend the time column (0,) as Prediction() requires
    embed = np.take( embedding, (0,) + combo, axis = 1 )
    Names = [ colNames[i] for i in combo ]
    
    # Evaluate prediction skill
    rho, rmse, mae, header, output, smap_output = Prediction( embed,
                                                              colNames,
                                                              target, args )
    Result = { 'names' : Names,  'rho'    : rho,
               'rmse'  : rmse,   'mae'    : mae,
               'header': header, 'output' : output }

    return tuple( ( combo, Result ) )

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def CCM( args, source = Source.Python ):
    '''
    Compute simplex cross-map skill over (random) subsamples of a time series. 

    Data are a .csv file with multiple simultaneous observations (columns)
    and "time" in the first column.  The -r (target) column is used for
    the cross prediction, -c (column) is embedded to dimension E.
    CCM is performed simultaneously between both target and column with
    the use of a process Pool. 
    
    Arguments: 
    -L (libsize) specifies a list of library sizes [start, stop, increment]
    -s (subsample) number of subsamples generated at each library size, if:
    -R (replacement) subsample with replacement. 

    Simplex "Predictions" are made over the same data/embedding slices as
    the library so that -l and -p parameters have no meaning. 
    '''

    if not len( args.columns ) :
        raise RuntimeError( "CCM() -c must specify the column to embed." )
    if not args.target :
        raise RuntimeError( "CCM() -r must specify the target column." )
    
    if args.seed :
        seed( int( args.seed ) )
    
    args.method = 'simplex'
    
    # Save args.plot flag, but disable so Prediction() does not plot
    showPlot  = args.plot
    args.plot = False

    # Process pool
    pool = Pool()
    
    # Create iterable with args for the two CrossMap() calls
    argsList = []
    # Cross mapping from -c (columns) to -r (target):
    argsList.append( deepcopy( args ) )

    # Switch the columns and target in args, assume only one column
    target       = args.target
    columns      = args.columns
    args.target  = columns[0]
    args.columns = [target]
                                     
    # Cross mapping from -r (target) to -c (columns) :
    argsList.append( deepcopy( args ) )

    # Switch back for plotting
    args.target  = target
    args.columns = columns
    
    # Submit the CrossMap jobs to the process pool
    results = pool.map( CrossMap, argsList )

    R0 = results[ 0 ] # tuple ( ID, PredictLibStats{} )
    R1 = results[ 1 ] # tuple ( ID, PredictLibStats{} )

    # Extract results based on the "columns to target" ID
    if R0[ 0 ] in str( args.columns ) + " to " + args.target :
        Col_Targ = R0[ 1 ]
        Targ_Col = R1[ 1 ]
    else :
        Col_Targ = R1[ 1 ]
        Targ_Col = R0[ 1 ]
    
    start, stop, increment = args.libsize

    print( "lib_size  ρ " + args.columns[0] + "   ρ " + args.target )
    print( "            " + args.target     + "     " + args.columns[0] )
    
    for lib_size in range( start, stop + 1, increment ) :
        col_targ_rho = Col_Targ[ lib_size ][0] # ρ in element [0]
        targ_col_rho = Targ_Col[ lib_size ][0] # ρ in element [0]

        print( "{:>8} {:>10} {:>10}".format( lib_size,
                                             np.round( col_targ_rho, 2 ),
                                             np.round( targ_col_rho, 2 ) ) )
    if showPlot :
        #-------------------------------------------------------
        # Plot rho at each subsample
        lib_sizes = np.arange( start, stop + 1, increment )
        col_targ_rho = [ Col_Targ[ lib_size ][0] \
                         for lib_size in range( start, stop + 1, increment ) ]
        
        targ_col_rho = [ Targ_Col[ lib_size ][0] \
                         for lib_size in range( start, stop + 1, increment ) ]
        
        fig, ax = plt.subplots( 1, 1, figsize = args.figureSize, dpi = 150 )
        ax.plot( lib_sizes, col_targ_rho,
                 linewidth = 3, color = 'blue',
                 label = args.columns[0] + " to " + args.target )
        ax.plot( lib_sizes, targ_col_rho,
                 linewidth = 3, color = 'red',
                 label = args.target + " to " + args.columns[0] )
        plt.axhline( y = 0, linewidth = 1 )
        ax.legend()
        
        ax.set( xlabel = 'Library Size',
                ylabel = "Cross map correlation " + r' $\rho$',
                title  = args.inputFile +\
                '  E=' + str( args.E  ) )
        plt.show()
        
    if source == Source.Jupyter :
        return { 'lib_sizes':lib_sizes, 'column_target_rho':col_targ_rho,
                 'target_column_rho':targ_col_rho }
    else:
        return

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def CrossMap( args ) :
    '''
    Pool worker function called from CCM()
    '''
    
    # Generate embedding on the data to be cross mapped (-c column)
    embedding, colNames, target = EmbedData( args )

    # Use entire library and prediction from embedding matrix
    libraryMatrix = predictionMatrix = embedding
    N_row = nRow( libraryMatrix )

    # Range of CCM library indices
    start, stop, increment = args.libsize
    
    if args.randomLib :
        # Random samples from library with replacement
        maxSamples = args.subsample
    else:
        # Contiguous samples up to the size of the library
        maxSamples = stop
        
    # Simplex: if k_NN not specified, set k_NN to E + 1
    if args.k_NN < 0 :
        args.k_NN = args.E + 1
        if args.verbose:
            print( "CCM() Set k_NN to E + 1 = " + str( args.k_NN ) +\
                   " for SimplexProjection." )

    #-----------------------------------------------------------------
    print( "CCM(): Simplex cross mapping from " + str( args.columns ) +\
           " to " + args.target +  "  E=" + str( args.E ) +\
           " k_nn=" + str( args.k_NN ) +\
           "  Library range: [{}, {}, {}]".format( start, stop, increment ))

    #-----------------------------------------------------------------
    # Distance for all possible pred : lib E-dimensional vector pairs
    # Distances is a Matrix of all row to to row distances
    #-----------------------------------------------------------------
    Distances = CCMGetDistances( libraryMatrix, args )
    
    #----------------------------------------------------------
    # Predictions
    #----------------------------------------------------------
    PredictLibStats = {} # { lib_size : ( rho, r, rmse, mae ) }
    # Loop for library sizes
    for lib_size in range( start, stop + 1, increment ) :

        if args.Debug :
            print( "CCM(): lib_size " + str( lib_size ) )

        prediction_rho = np.zeros( ( maxSamples, 4 ) )
        # Loop for subsamples
        for n in range( maxSamples ) :

            if args.randomLib :
                # Uniform random sample of rows, with replacement
                lib_i = randint( low  = 0,
                                 high = N_row,
                                 size = lib_size )
            else :
                if lib_size >= N_row :
                    # library size exceeded, back down
                    lib_i = np.arange( 0, N_row )
                    
                    if args.warnings or args.verbose :
                        print( "CCM(): max lib_size is {}, "
                               "lib_size has been limited.".format( N_row ) )
                else :
                    # Contiguous blocks up to N_rows = maxSamples
                    if n + lib_size < N_row :
                        lib_i = np.arange( n, n + lib_size )
                    else:
                        # n + lib_size exceeds N_row, wrap around to data origin
                        lib_start = np.arange( n, N_row )
                        max_i     = min( lib_size - (N_row - n), N_row )
                        lib_wrap  = np.arange( 0, max_i )
                        lib_i     = np.concatenate((lib_start,lib_wrap),axis=0)

            #----------------------------------------------------------
            # k_NN nearest neighbors : Local CCMGetNeighbors() function
            #----------------------------------------------------------
            neighbors, distances = CCMGetNeighbors( Distances, lib_i, args )

            predictions = SimplexProjection( libraryMatrix[ lib_i, : ],
                                             target       [ lib_i ],
                                             neighbors,
                                             distances,
                                             args )

            rho, r, rmse, mae = ComputeError( target[ lib_i ], predictions )

            prediction_rho[ n, : ] = [ rho, r, rmse, mae ]

        rho_  = np.mean( prediction_rho[ :, 0 ] )
        r_    = np.mean( prediction_rho[ :, 1 ] )
        rmse_ = np.mean( prediction_rho[ :, 2 ] )
        mae_  = np.mean( prediction_rho[ :, 3 ] )
        
        PredictLibStats[ lib_size ] = ( rho_, r_, rmse_, mae_  )

    # Return tuple with ( ID, PredictLibStats{} )
    return ( str( args.columns ) + " to " + args.target, PredictLibStats )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def CCMGetDistances( libraryMatrix, args ) :
    '''
    Note that for CCM the libraryMatrix and predictionMatrix are the same.

    Return Distances: a square matrix with distances.
    Matrix elements D[i,j] hold the distance between the E-dimensional
    phase space point (vector) between rows (observations) i and j.
    '''
    
    N_row = nRow( libraryMatrix )
    
    D = np.full( (N_row, N_row), 1E30 ) # Distance matrix init to 1E30
    E = args.E + 1
    
    for row in range( N_row ) :
        # Get E-dimensional vector from this library row
        # Exclude the 1st column (j=0) of times
        y = libraryMatrix[ row, 1:E: ]

        for col in range( N_row ) :
            # Ignore the diagonal (row == col)
            if row == col :
                continue
            
            # Find distance between vector (y) and other library vector
            # Exclude the 1st column (j=0) of Time
            D[ row, col ] = Distance( libraryMatrix[ col, 1:E: ], y )
            # Insert degenerate values since D[i,j] = D[j,i]
            D[ col, row ] = D[ row, col ]

    return ( D )
    
#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def CCMGetNeighbors( Distances, lib_i, args ) :
    '''
    Return a tuple of ( neighbors, distances ). neighbors is a matrix of 
    row indices in the library matrix. Each neighbors row represents one 
    prediction vector. Columns are the indices of k_NN nearest neighbors 
    for the prediction vector (phase-space point) in the library matrix.
    distances is a matrix with the same shape as neighbors holding the 
    corresponding distance values in each row.

    Note that the indices in neighbors are not the original indices in
    the libraryMatrix rows (observations), but are with respect to the
    distances subset defined by the list of rows lib_i, and so have values
    from 0 to len(lib_i)-1.
    '''
    
    N_row = len( lib_i )           # Subset of libraryMatrix and Distances
    col_i = np.arange( len( lib_i ) )#Vector of col indices [0,...len(lib_i)-1]
    k_NN  = args.k_NN

    if args.Debug :
        print( 'GetNeighbors() Distances:' )
        print( round( Distances[ 0:5, 0:5 ], 4 ) )
        print( 'N_row = ' + str( N_row ) )

    # Matrix to hold libraryMatrix row indices
    # One row for each prediction vector, k_NN columns for each index
    neighbors = np.zeros( (N_row, k_NN), dtype = int )

    # Matrix to hold libraryMatrix k_NN distance values
    # One row for each prediction vector, k_NN columns for each index
    distances = np.zeros( (N_row, k_NN) )

    # For each prediction vector (row in predictionMatrix) find the list
    # of library indices that are within k_NN points
    row = 0
    for row_i in lib_i :
        # Take D[ row, col ] a row at a time, col represent other row distance
        # Sort based on Distance with paired column indices
        # D_row_i is a list of tuples sorted by increasing distance
        D_row_i = sorted( zip( Distances[ row_i, lib_i ], col_i ) )

        # Take the first k_NN distances and column indices
        k_NN_distances = [ x[0] for x in D_row_i ][ 0:k_NN ] # distance
        k_NN_neighbors = [ x[1] for x in D_row_i ][ 0:k_NN ] # index

        if args.Debug :
            if np.amax( k_NN_distances ) > 1E29 :
                raise RuntimeError( "GetNeighbors() Library is too small to " +\
                                    "resolve " + str( k_NN ) + " k_NN "   +\
                                    "neighbors." )
        
            # Check for ties.  JP: haven't found any so far...
            if len( k_NN_neighbors ) != len( np.unique( k_NN_neighbors ) ) :
                raise RuntimeError( "GetNeighbors() Degenerate neighbors" )
        
        neighbors[ row, ] = k_NN_neighbors
        distances[ row, ] = k_NN_distances

        row = row + 1
    
    if args.Debug :
        print( 'GetNeighbors()  neighbors' )
        print( neighbors[ 0:5, ] )

    return ( neighbors, distances )
