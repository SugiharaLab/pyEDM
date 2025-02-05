
# python modules

# package modules
from numpy  import array, divide, exp, fmax, full, integer, nan
from numpy  import linspace, power, subtract, sum, zeros
from pandas import DataFrame, Series, concat

# local modules
from .EDM import EDM as EDMClass

#-----------------------------------------------------------
class Simplex( EDMClass ):
    '''Simplex class : child of EDM
       CCM & Multiview inhereted from Simplex
       To Do : Neighbor ties'''

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
                  exclusionRadius = 0,
                  embedded        = False,
                  validLib        = [],
                  noTime          = False,
                  generateSteps   = 0,
                  generateConcat  = False,
                  ignoreNan       = True,
                  verbose         = False ):
        '''Initialize Simplex as child of EDM.
           Set data object to dataFrame.
           Setup : Validate(), CreateIndices(), get targetVec, time'''

        # Instantiate EDM class: inheret EDM members to self
        super(Simplex, self).__init__( dataFrame, 'Simplex' )

        # Assign parameters from API arguments
        self.columns         = columns
        self.target          = target
        self.lib             = lib
        self.pred            = pred
        self.E               = E
        self.Tp              = Tp
        self.knn             = knn
        self.tau             = tau
        self.exclusionRadius = exclusionRadius
        self.embedded        = embedded
        self.validLib        = validLib
        self.noTime          = noTime
        self.generateSteps   = generateSteps
        self.generateConcat  = generateConcat
        self.ignoreNan       = ignoreNan
        self.verbose         = verbose

        # Prediction row accounting of library neighbor ties
        # self.anyTies       = False
        # self.ties          = None  # (bool) true/false each prediction row
        # self.tieFirstIndex = None  # (int) index in knn of first tie
        # self.tiePairs      = None  # vector of 2-tuples

        # Setup
        self.Validate()      # EDM Method
        self.CreateIndices() # Generate lib_i & pred_i, validLib : EDM Method

        self.targetVec = self.Data[ [ self.target[0] ] ].to_numpy()

        if self.noTime :
            # Generate a time/index vector, store as ndarray
            timeIndex = [ i for i in range( 1, self.Data.shape[0] + 1 ) ]
            self.time = Series( timeIndex, dtype = int ).to_numpy()
        else :
            # 1st data column is time
            self.time = self.Data.iloc[ :, 0 ].to_numpy()

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
        '''Simplex Projection
           Sugihara & May (1990) doi.org/10.1038/344734a0'''
        if self.verbose:
            print( f'{self.name}: Project()' )

        # First column of knn_distances is minimum distance of all N pred rows
        minDistances = self.knn_distances[:,0]
        # In case there is 0 in minDistances: minWeight = 1E-6
        minDistances = fmax( minDistances, 1E-6 )

        # Divide each column of the N x k knn_distances matrix by N row
        # column vector minDistances
        scaledDistances = divide( self.knn_distances, minDistances[:,None] )

        weights      = exp( -scaledDistances )  # N x k
        weightRowSum = sum( weights, axis = 1 ) # N x 1

        # Matrix of knn_neighbors + Tp defines library target values
        # JP : Find optimal way to fill libTargetValues, for now:
        #   Since number of knn is usually less than number of pred rows
        #   loop over knn_neighbors_Tp columns to get target value column
        #   vectors from the knn_neighbors_Tp row indices
        knn_neighbors_Tp = self.knn_neighbors + self.Tp     # N x k
        libTargetValues  = zeros( knn_neighbors_Tp.shape )  # N x k

        for j in range( knn_neighbors_Tp.shape[1] ) : # for each column j of k   
            libTargetValues[ :, j ][ :, None ] = \
                self.targetVec[ knn_neighbors_Tp[ :, j ] ]

        # Projection is average of weighted knn library target values
        self.projection = sum(weights * libTargetValues, axis=1) / weightRowSum

        # "Variance" estimate assuming weights are probabilities
        libTargetPredDiff = subtract( libTargetValues, self.projection[:,None] )
        deltaSqr          = power( libTargetPredDiff, 2 )
        self.variance     = sum( weights * deltaSqr, axis = 1 ) / weightRowSum

    #-------------------------------------------------------------------
    def Generate( self ) :
    #-------------------------------------------------------------------
        '''Simplex Generation
           Given lib: override pred for single prediction at end of lib
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

        if self.verbose:
            print(f'\tData shape: {self.Data.shape} : ' +\
                  f'{self.Data.columns.to_list()}')
            print(self.Data.head(3))
            print(f'\tlib: {lib}')

        # Override pred for single prediction at end of lib
        pred = [ lib[-1] - 1, lib[-1] ]
        if self.verbose:
            print(f'\tGenerate(): pred overriden to {pred}')

        # Output DataFrame to replace self.Projection
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

        # Allocate vector for univariate column data
        # At each iteration the prediction is stored in columnData
        # timeData and columnData are copied to newData for next iteration
        columnData     = full( N + nOutRows, nan )
        columnData[:N] = self.Data.loc[ :, column ] # First col only

        # Allocate output time vector & newData DataFrame
        timeData = full( N + nOutRows, nan, dtype = time_dtype )
        if self.noTime :
            # If noTime create a time vector and join into self.Data
            timeData[:N] = linspace( 1, N, N )
            timeDF       = DataFrame( {'Time' : timeData[:N]} )
            self.Data    = timeDF.join( self.Data, lsuffix = '_' )
        else :
            timeData[:N] = self.time

        newData = self.Data.copy()

        #-------------------------------------------------------------------
        # Loop for each feedback generation step
        #-------------------------------------------------------------------
        for step in range( self.generateSteps ) :
            if self.verbose :
                print( f'{self.name}: Generate(): step {step} {"="*50}')

            # Local SimplexClass for generation
            G = Simplex( dataFrame       = newData,
                         columns         = column,
                         target          = target,
                         lib             = lib,
                         pred            = pred,
                         E               = self.E,
                         Tp              = self.Tp,
                         knn             = self.knn,
                         tau             = self.tau,
                         exclusionRadius = self.exclusionRadius,
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
                print( '1) G.Projection' )
                print( G.Projection ); print()

            newPrediction = G.Projection['Predictions'].iat[-1]
            newTime       = G.Projection.iloc[-1, 0] # Presume col 0 is time

            # 2) Save prediction in generated --------------------------
            generated.iloc[ step, : ] = G.Projection.iloc[-1, :]

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

        # Replace self.Projection with Generated
        if self.generateConcat :
            # Join original data observations with generated predictions
            timeName = self.Data.columns[0]
            dataDF   = self.Data.loc[ :, [timeName, column] ]
            dataDF.columns = [ 'Time', 'Observations' ]
            self.Projection = concat( [ dataDF, generated ], axis = 0 )

        else :
            self.Projection = generated
