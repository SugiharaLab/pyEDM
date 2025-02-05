# python modules

# package modules
from numpy  import apply_along_axis, insert, isnan, isfinite, exp
from numpy  import full, integer, linspace, mean, nan, power, sum
from pandas import DataFrame, Series, concat

from numpy.linalg import lstsq # from scipy.linalg import lstsq

# local modules
from .EDM import EDM as EDMClass

#-----------------------------------------------------------
class SMap( EDMClass ):
    '''SMap class : child of EDM'''

    def __init__( self,
                  dataFrame       = None,
                  columns         = "",
                  target          = "",
                  lib             = "",
                  pred            = "",
                  E               = 0,
                  Tp              = 1,
                  knn             = 0,
                  tau             = -1,
                  theta           = 0.,
                  exclusionRadius = 0,
                  solver          = None,
                  embedded        = False,
                  validLib        = [],
                  noTime          = False,
                  generateSteps   = 0,
                  generateConcat  = False,
                  ignoreNan       = True,
                  verbose         = False ):
        '''Initialize SMap as child of EDM.
           Set data object to dataFrame.
           Setup : Validate(), CreateIndices(), get targetVec, time'''

        # Instantiate EDM class: inheret all members to self
        super(SMap, self).__init__( dataFrame, 'SMap' )

        # Assign parameters from API arguments
        self.columns         = columns
        self.target          = target
        self.lib             = lib
        self.pred            = pred
        self.E               = E
        self.Tp              = Tp
        self.knn             = knn
        self.tau             = tau
        self.theta           = theta
        self.exclusionRadius = exclusionRadius
        self.solver          = solver
        self.embedded        = embedded
        self.validLib        = validLib
        self.noTime          = noTime
        self.generateSteps   = generateSteps
        self.generateConcat  = generateConcat
        self.ignoreNan       = ignoreNan
        self.verbose         = verbose

        # SMap storage
        self.Coefficients   = None # DataFrame SMap API output
        self.SingularValues = None # DataFrame SMap API output
        self.coefficients   = None # ndarray SMap output (N_pred, E+1)
        self.singularValues = None # ndarray SMap output (N_pred, E+1)

        # Setup
        self.Validate()      # EDM Method
        self.CreateIndices() # Generate lib_i & pred_i, validLib

        self.targetVec = self.Data[ [ self.target[0] ] ].to_numpy()

        if self.noTime :
            # Generate a time/index vector, store as ndarray
            timeIndex = [ i for i in range( 1, self.Data.shape[0] + 1 ) ]
            self.time = Series( timeIndex, dtype = int ).to_numpy()
        else :
            # 1st data column is time
            self.time = self.Data.iloc[ :, 0 ].to_numpy()

        if self.solver is None :
            self.solver = lstsq

    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------
    def Run( self ) :
    #-------------------------------------------------------------------
        self.EmbedData()
        self.RemoveNan()
        self.FindNeighbors()
        self.Project()
        self.FormatProjection()

    #-------------------------------------------------------------------
    def Project( self ) :
    #-------------------------------------------------------------------
        '''For each prediction row compute projection as the linear
           combination of regression coefficients (C) of weighted
           embedding vectors (A) against target vector (B) : AC = B.

           Weights reflect the SMap theta localization of the knn
           for each prediction. Default knn = len( lib_i ). 

           Matrix A has (weighted) constant (1) first column
           to enable a linear intercept/bias term.

           Sugihara (1994) doi.org/10.1098/rsta.1994.0106
        '''

        if self.verbose:
            print( f'{self.name}: Project()' )

        N_pred = len( self.pred_i )
        N_dim  = self.E + 1

        self.projection     = full( N_pred, nan, dtype = float )
        self.variance       = full( N_pred, nan, dtype = float )
        self.coefficients   = full( (N_pred, N_dim), nan, dtype = float )
        self.singularValues = full( (N_pred, N_dim), nan, dtype = float )

        embedding = self.Embedding.to_numpy() # reference to ndarray

        # Compute average distance for knn pred rows into a vector
        distRowMean = mean( self.knn_distances, axis = 1 )

        # Weight matrix of row vectors
        if self.theta == 0 :
            W = full( self.knn_distances.shape, 1., dtype = float )
        else :
            distRowScale = self.theta / distRowMean
            W = exp( -distRowScale[:,None] * self.knn_distances )

        # knn_neighbors + Tp
        knn_neighbors_Tp = self.knn_neighbors + self.Tp # N_pred x knn

        # Function to select targetVec for rows of Boundary condition matrix
        def GetTargetRow( knn_neighbor_row ) :
            return self.targetVec[ knn_neighbor_row ][:,0]

        # Boundary condition matrix of knn + Tp targets : N_pred x knn
        B = apply_along_axis( GetTargetRow, 1, knn_neighbors_Tp )

        if self.targetVecNan :
            # If there are nan in the targetVec need to remove them
            # from B since Solver returns nan. B_valid is matrix of
            # B row booleans of valid data for pred rows
            # Function to apply isfinite to rows
            def FiniteRow( B_row ) :
                return isfinite( B_row )

            B_valid = apply_along_axis( FiniteRow, 1, B )

        # Weighted boundary condition matrix of targets : N_pred x knn
        wB = W * B

        # Process each prediction row
        for row in range( N_pred ) :
            # Allocate array
            A = full( ( self.knn, N_dim ), nan, dtype = float )

            A[:,0] = W[row,:] # Intercept bias terms in column 0 (weighted)

            libRows = self.knn_neighbors[ row, : ] # 1 x knn

            for j in range( 1, N_dim ) :
                A[ :, j ] = W[ row, : ] * embedding[ libRows, j-1 ]

            wB_ = wB[row,:]

            if self.targetVecNan :
                # Redefine A, wB_ to remove targetVec nan
                valid_i = B_valid[ row, : ]
                A       = A [ valid_i, : ]
                wB_     = wB[ row, valid_i ]

            # Linear mapping of theta weighted embedding A onto weighted target B
            C, SV = self.Solver( A, wB_ )

            self.coefficients  [ row, : ] = C
            self.singularValues[ row, : ] = SV

            # Prediction is local linear projection.
            if isnan( C[0] ) :
                projection_ = 0
            else :
                projection_ = C[0]

            for e in range( 1, N_dim ) :
                projection_ = projection_ + \
                    C[e] * embedding[ self.pred_i[ row ], e-1 ]

            self.projection[ row ] = projection_

            # "Variance" estimate assuming weights are probabilities
            if self.targetVecNan :
                deltaSqr = power( B[ row, valid_i ] - projection_, 2 )
                self.variance[ row ] = \
                    sum( W[ row, valid_i ] * deltaSqr ) \
                    / sum( W[ row, valid_i ] )
            else :
                deltaSqr = power( B[row,:] - projection_, 2 )
                self.variance[ row ] = sum(W[row]*deltaSqr) / sum(W[row])

    #-------------------------------------------------------------------
    def Solver( self, A, wB ) :
    #-------------------------------------------------------------------
        '''Call SMap solver. Default is numpy.lstsq'''

        if self.solver.__class__.__name__ in \
           [ 'function', '_ArrayFunctionDispatcher' ] and \
           self.solver.__name__ == 'lstsq' :
            # numpy default lstsq or scipy lstsq
            C, residuals, rank, SV = self.solver( A, wB, rcond = None )
            return C, SV

        # Otherwise, sklearn.linear_model passed as solver
        # Coefficient matrix A has weighted unity vector in the first
        # column to create a bias (intercept) term. sklearn.linear_model's
        # include an intercept term by default. Ignore first column of A.
        LM = self.solver.fit( A[:,1:], wB )
        C  = LM.coef_
        if hasattr( LM, 'intercept_' ) :
            C = insert( C, 0, LM.intercept_ ) 
        else :
            C = insert( C, 0, nan ) # Insert nan for intercept term

        if self.solver.__class__.__name__ == 'LinearRegression' :
            SV = LM.singular_ # Only LinearRegression has singular_
            SV = insert( SV, 0, nan )
        else :
            SV = None # full( A.shape[0], nan )

        return C, SV

    #-------------------------------------------------------------------
    def Generate( self ) :
    #-------------------------------------------------------------------
        '''SMap Generation
           Given lib: override pred to be single prediction at end of lib
           Replace self.Projection with G.Projection

           Note: Generation with datetime time values fails: incompatible
                 numpy.datetime64, timedelta64 and python datetime, timedelta
        '''
        if self.verbose:
            print( f'{self.name}: Generate()' )

        # Local references for convenience
        N      = self.Data.shape[0]
        column = self.columns[0]
        target = self.target[0]
        lib    = self.lib

        # Override pred for single prediction at end of lib
        pred = [ lib[-1] - 1, lib[-1] ]
        if self.verbose:
            print(f'{self.name}: Generate(): pred overriden to {pred}')

        # Output DataFrames to replace self.Projection, self.Coefficients...
        if self.noTime :
            time_dtype = float # numpy int cannot represent nan, use float
        else :
            self.ConvertTime()

            time0 = self.time[0]
            if isinstance( time0, int ) or isinstance( time0, integer ) :
                time_dtype = float # numpy int cannot represent nan, use float
            else :
                time_dtype = type( time0 )

        nOutRows  = self.generateSteps
        generated = DataFrame({'Time' : full(nOutRows, nan, dtype = time_dtype),
                               'Observations'  : full(nOutRows, nan),
                               'Predictions'   : full(nOutRows, nan),
                               'Pred_Variance' : full(nOutRows, nan)})

        coeff_ = full( (nOutRows, self.E + 2), nan )
        if self.tau < 0 :
            coefNames = [f'∂{target}/∂{target}(t-{e})' for e in range(self.E)]
        else :
            coefNames = [f'∂{target}/∂{target}(t+{e})' for e in range(self.E)]
        colNames = [ 'Time', 'C0' ] + coefNames
        genCoeff = DataFrame( coeff_, columns = colNames )

        sv_      = full( (nOutRows, self.E + 2), nan )
        colNames = [ 'Time' ] + [ f'C{i}' for i in range( self.E + 1 ) ]
        genSV    = DataFrame( sv_, columns = colNames )

        # Allocate vector for univariate column data
        # At each iteration the prediction is stored in columnData
        # timeData and columnData are copied to newData for next iteration
        columnData     = full( N + nOutRows, nan )
        columnData[:N] = self.Data.loc[ :, column ] # First col only

        # Allocate output time vector & newData DataFrame
        timeData = full( N + nOutRows, nan )
        if self.noTime :
            # If noTime create a time vector and join into self.Data
            timeData[:N] = linspace( 1, N, N )
            timeDF       = DataFrame( {'Time' : timeData[:N]} )
            self.Data    = timeDF.join( self.Data, lsuffix = '_' )
        else :
            timeData[:N] = self.time # Presume column 0 is time

        newData = self.Data.copy()

        #-------------------------------------------------------------------
        # Loop for each feedback generation step
        #-------------------------------------------------------------------
        for step in range( self.generateSteps ) :
            if self.verbose :
                print( f'{self.name}: Generate(): step {step} {"="*50}')

            # Local SMapClass for generation
            G = SMap( dataFrame       = newData,
                      columns         = column,
                      target          = target,
                      lib             = lib,
                      pred            = pred,
                      E               = self.E,
                      Tp              = self.Tp,
                      knn             = self.knn,
                      tau             = self.tau,
                      theta           = self.theta,
                      exclusionRadius = self.exclusionRadius,
                      solver          = self.solver,
                      embedded        = self.embedded,
                      validLib        = self.validLib,
                      noTime          = self.noTime,
                      generateSteps   = self.generateSteps,
                      generateConcat  = self.generateConcat,
                      ignoreNan       = self.ignoreNan,
                      verbose         = self.verbose )

            # 1) Generate prediction ----------------------------------
            G.Run()

            if self.verbose :
                print( 'G.Projection' )
                print( G.Projection ); print()

            newPrediction = G.Projection['Predictions'].iat[-1]
            newTime       = G.Projection.iloc[-1, 0] # Presume col 0 is time

            # 2) Save prediction in generated --------------------------
            generated.iloc[ step, : ] = G.Projection.iloc    [-1, :]
            genCoeff.iloc [ step, : ] = G.Coefficients.iloc  [-1, :]
            genSV.iloc    [ step, : ] = G.SingularValues.iloc[-1, :]

            if self.verbose :
                print( f'2) generated\n{generated}\n' )

            # 3) Increment library by adding another row index ---------
            # Dynamic library not implemented

            # 4) Increment prediction indices --------------------------
            pred = [ p + 1 for p in pred ]

            if self.verbose:
                print(f'4) pred {pred}')

            # 5) Add 1-step ahead projection to newData for next Project()
            columnData[ N + step ] = newPrediction
            timeData  [ N + step ] = newTime

            # JP : for big data this is likely not efficient
            newData = DataFrame( { 'Time'      : timeData  [:(N + step + 1)],
                                   f'{column}' : columnData[:(N + step + 1)] } )

            if self.verbose:
                print(f'5) newData: {newData.shape}  newData.tail(4):')
                print( newData.tail(4) )
        #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Loop for each feedback generation step
        #----------------------------------------------------------

        # Replace self.Projection with generated
        if self.generateConcat :
            timeName = self.Data.columns[0]
            dataDF   = self.Data.loc[ :, [timeName, column] ]
            dataDF.columns = [ 'Time', 'Observations' ]
            self.Projection = concat( [ dataDF, generated ], axis = 0 )
        else :
            self.Projection = generated

        self.Coefficients   = genCoeff
        self.SingularValues = genSV
