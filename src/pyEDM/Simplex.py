
# python modules

# package modules
from numpy  import divide, exp, fmax, power, subtract, sum, zeros
from pandas import Series

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
    def Project( self ) :
    #-------------------------------------------------------------------
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
        # JP : Need to find optimal way to fill libTargetValues
        # Since number of knn is usually less than number of pred rows
        # loop over knn_neighbors_Tp columns to get target value column
        # vectors from the knn_neighbors_Tp row indices
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
        print( f'{self.name}: Generate() Not implemented.' )
