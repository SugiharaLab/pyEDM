
# python modules
from multiprocessing import Pool
from math import floor, sqrt
from warnings import warn
from itertools import combinations, repeat

# package modules
from numpy import argsort, array

# local modules
from .Simplex import Simplex as SimplexClass
from .AuxFunc import ComputeError, IsIterable

import pyEDM.API      as API
import pyEDM.PoolFunc as PoolFunc

#------------------------------------------------------------------
class Multiview:
    '''Multiview class : Base class. Contains a Simplex instance

       D represents the number of variables to combine for each
       assessment, if not specified, it is the number of columns.

       E is the embedding dimension of each variable. 
       If E = 1, no time delay embedding is done, but the variables
       in the embedding are named X(t-0), Y(t-0)...

       Simplex.Validate() sets knn equal to E+1 if knn not specified, 
       so we need to explicitly set knn to D + 1.

       Parameter 'multiview' is the number of top-ranked D-dimensional
       predictions for the final prediction. Corresponds to parameter k
       in Ye & Sugihara with default k = sqrt(m) where m is the number 
       of combinations C(n,D) available from the n = D * E columns 
       taken D at-a-time.

       Ye H., and G. Sugihara, 2016. Information leverage in
       interconnected ecosystems: Overcoming the curse of dimensionality
       Science 353:922-925.

       NOTE: Multiview evaluates the top projections using in-sample
             library predictions. It can be shown that highly accurate
             in-sample predictions can be made from arbitrary non-
             constant, non-oscillatory vectors. Therefore, some attention
             may be warranted to filter prospective embedding vectors.
             The trainLib flag disables this default behavior (pred == lib)
             so that the top k rankings are done using the specified
             lib and pred. 
    '''

    def __init__( self,
                  dataFrame       = None,
                  columns         = "",
                  target          = "", 
                  lib             = "",
                  pred            = "",
                  D               = 0, 
                  E               = 1, 
                  Tp              = 1,
                  knn             = 0,
                  tau             = -1,
                  multiview       = 0,
                  exclusionRadius = 0,
                  trainLib        = True,
                  excludeTarget   = False,
                  ignoreNan       = True,
                  verbose         = False,
                  numProcess      = 4,
                  returnObject    = False ):
        '''Initialize Multiview.'''

        # Assign parameters from API arguments
        self.name            = 'Multiview'
        self.Data            = dataFrame
        self.columns         = columns
        self.target          = target
        self.lib             = lib
        self.pred            = pred
        self.D               = D
        self.E               = E
        self.Tp              = Tp
        self.knn             = knn
        self.tau             = tau
        self.multiview       = multiview
        self.exclusionRadius = exclusionRadius
        self.trainLib        = trainLib
        self.excludeTarget   = excludeTarget
        self.ignoreNan       = ignoreNan
        self.verbose         = verbose
        self.numProcess      = numProcess

        self.Embedding  = None # DataFrame 
        self.View       = None # DataFrame
        self.Projection = None # DataFrame

        self.combos             = None # List of column combinations (tuples)
        self.topRankCombos      = None # List of top columns (tuples)
        self.topRankProjections = None # dict of columns : DataFrame
        self.topRankStats       = None # dict of columns : dict of stats

        # Setup
        self.Validate() # Multiview Method
        self.Setup()    # Embed Data

    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------
    def Rank( self ) :
        '''Multiprocess to rank top multiview vectors'''

        if self.verbose:
            print( f'{self.name}: Rank()' )

        args = { 'target'          : self.target, 
                 'lib'             : self.lib,
                 'pred'            : self.pred,
                 'E'               : self.D,
                 'Tp'              : self.Tp,
                 'tau'             : self.tau,
                 'exclusionRadius' : self.exclusionRadius,
                 'embedded'        : True,
                 'noTime'          : True,
                 'ignoreNan'       : self.ignoreNan }

        if self.trainLib :
            # Set pred = lib for in-sample training 
            args['pred'] = self.lib

        # Create iterable for Pool.starmap, repeated copies of data, args
        poolArgs = zip( self.combos, repeat( self.Embedding ), repeat( args ) )

        # Multiargument starmap : MultiviewSimplexRho in PoolFunc
        with Pool( processes = self.numProcess ) as pool :
            rhoList = pool.starmap( PoolFunc.MultiviewSimplexRho, poolArgs )

        rhoVec    = array( rhoList, dtype = float )
        rank_i    = argsort( rhoVec )[::-1] # Reverse results 
        topRank_i = rank_i[ :self.multiview ]

        self.topRankCombos = [ self.combos[i] for i in topRank_i ]

    #-------------------------------------------------------------------
    # 
    #-------------------------------------------------------------------
    def Project( self ) :
        '''Projection with top multiview vectors'''

        if self.verbose:
            print( f'{self.name}: Project()' )

        args = { 'target'          : self.target, 
                 'lib'             : self.lib,
                 'pred'            : self.pred,
                 'E'               : self.D,
                 'Tp'              : self.Tp,
                 'tau'             : self.tau,
                 'exclusionRadius' : self.exclusionRadius,
                 'embedded'        : True,
                 'noTime'          : True,
                 'ignoreNan'       : self.ignoreNan }

        # Create iterable for Pool.starmap, repeated copies of data, args
        poolArgs = zip( self.topRankCombos, repeat( self.Embedding ),
                        repeat( args ) )

        # Multiargument starmap : MultiviewSimplexPred in PoolFunc
        with Pool( processes = self.numProcess ) as pool :
            dfList = pool.starmap( PoolFunc.MultiviewSimplexPred, poolArgs )

        self.topRankProjections = dict( zip( self.topRankCombos, dfList ) )

    #--------------------------------------------------------------------
    def Setup( self ):
    #--------------------------------------------------------------------
        '''Set D, lib, pred, combos. Embed Data. 
        '''
        if self.verbose:
            print( f'{self.name}: Setup()' )

        # Set default lib & pred if not provided
        if self.trainLib :
            if not len( self.pred ) and not len( self.lib ) :
                # Set lib & pred for ranking : lib, pred = 1/2 data
                self.lib  = [ 1, floor( self.Data.shape[0]/2 ) ]
                self.pred = [ floor( self.Data.shape[0]/2 ) + 1,
                              self.Data.shape[0]]

        # Establish state-space dimension D
        # default to number of input columns (not embedding columns)
        if self.D == 0 :
            self.D = len( self.columns )

        # Check D is not greater than number of embedding columns
        if self.D > len( self.columns ) * self.E :
            newD = len( self.columns ) * self.E
            msg = f'Validate() {self.name}: D = {self.D}'      +\
                ' exceeds number of columns in the embedding: {newD}.' +\
                f' D set to {newD}'
            warn( msg )

            self.D = newD

        # Remove target columns from potential combos
        if self.excludeTarget :
            comboCols = [col for col in self.columns if col not in self.target]
        else :
            comboCols = self.columns

        # Embed Data
        self.Embedding = API.Embed( self.Data, E = self.E, tau = self.tau,
                                    columns = comboCols )

        # Combinations of possible embedding vectors, D at-a-time
        self.combos = list( combinations( self.Embedding.columns, self.D ) )

        # Establish number of ensembles if not specified
        if self.multiview < 1 :
            # Ye & Sugihara suggest sqrt( m ) as number of embeddings to avg
            self.multiview = floor( sqrt( len( self.combos ) ) )

            if self.verbose :
                msg = 'Validate() {self.name}:' +\
                    f' Set view sample size to {self.multiview}'
                print( msg, flush = True )

        if self.multiview > len( self.combos ) :
            msg = 'Validate() {self.name}: multiview ensembles ' +\
                f' {self.multiview} exceeds the number of available' +\
                f' combinations: {len(combos)}. Set to {len(combos)}.'
            warn( msg )

            self.multiview = len( self.combos )

    #--------------------------------------------------------------------
    def Validate( self ):
    #--------------------------------------------------------------------
        if self.verbose:
            print( f'{self.name}: Validate()' )

        if not self.columns :
            raise RuntimeError( f'Validate() {self.name}: columns required.' )
        if not IsIterable( self.columns ) :
            self.columns = self.columns.split()

        if not self.target :
            raise RuntimeError( f'Validate() {self.name}: target required.' )
        if not IsIterable( self.target ) :
            self.target = self.target.split()
        # Add (t-0) to target since Embedding columns are mapped
        self.target[0] = self.target[0] + '(t-0)'

        if not self.trainLib :
            if not len( self.lib ) :
                msg = f'{self.name}: Validate(): trainLib False requires' +\
                       ' lib specification.'
                raise RuntimeError( msg )

            if not len( self.pred ) :
                msg = f'{self.name}: Validate(): trainLib False requires' +\
                       ' pred specification.'
                raise RuntimeError( msg )
