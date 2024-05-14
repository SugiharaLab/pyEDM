# python modules

# package modules
from numpy  import apply_along_axis, insert, isnan, isfinite, exp
from numpy  import full, mean, nan, power, sum
from pandas import Series

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
                  ignoreNan       = True,
                  verbose         = False ):
        '''Initialize SMap as child of EDM. 
           Set data object to dataFrame.
           Setup : Validate(), CreateIndices(), get targetVec, allTime'''

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
            # Generate a time/index vector
            timeIndex = [ i for i in range( 1, self.Data.shape[0] + 1 ) ]
            self.allTime = Series( timeIndex, dtype = int )
        else :
            # 1st data column is time
            self.allTime = self.Data.iloc[ :, 0 ]

        if self.solver is None :
            self.solver = lstsq

    #-------------------------------------------------------------------
    # Methods
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
        '''

        if self.verbose:
            print( f'{self.name}: Project()' )

        N_pred = len( self.pred_i )
        N_dim  = self.E + 1

        self.projection_    = full( N_pred, nan, dtype = float )
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
                projection = 0
            else :
                projection = C[0]

            for e in range( 1, N_dim ) :
                projection = projection + \
                    C[e] * embedding[ self.pred_i[ row ], e-1 ]

            self.projection_[ row ] = projection

            # "Variance" estimate assuming weights are probabilities
            if self.targetVecNan :
                deltaSqr = power( B[ row, valid_i ] - projection, 2 )
                self.variance[ row ] = \
                    sum( W[ row, valid_i ] * deltaSqr ) \
                    / sum( W[ row, valid_i ] )
            else :
                deltaSqr = power( B[row,:] - projection, 2 )
                self.variance[ row ] = sum(W[row]*deltaSqr) / sum(W[row])

    #-------------------------------------------------------------------
    def Solver( self, A, wB ) :
    #-------------------------------------------------------------------
        # print( f'{self.name}: Solver.' )
        '''Call SMap solver. Default is numpy.lstsq'''

        if self.solver.__class__.__name__ == 'function' and \
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
        print( f'{self.name}: Generate() Not implemented.' )
