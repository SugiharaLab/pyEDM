
# python modules
from multiprocessing import Pool

# package modules
from pandas import DataFrame, concat
from numpy  import array, exp, fmax, divide, mean, nan, roll, sum, zeros
from numpy.random import default_rng

# local modules
from .Simplex import Simplex as SimplexClass
from .AuxFunc import ComputeError, IsIterable

#------------------------------------------------------------
class CCM:
    '''CCM class : Base class. Contains two Simplex instances'''

    def __init__( self,
                  dataFrame       = None,
                  columns         = "",
                  target          = "",
                  E               = 0,
                  Tp              = 0,
                  knn             = 0,
                  tau             = -1,
                  exclusionRadius = 0,
                  libSizes        = [],
                  sample          = 0,
                  seed            = None,
                  includeData     = False,
                  embedded        = False,
                  validLib        = [],
                  noTime          = False,
                  ignoreNan       = True,
                  verbose         = False ):
        '''Initialize CCM.'''

        # Assign parameters from API arguments
        self.name            = 'CCM'
        self.Data            = dataFrame
        self.columns         = columns
        self.target          = target
        self.E               = E
        self.Tp              = Tp
        self.knn             = knn
        self.tau             = tau
        self.exclusionRadius = exclusionRadius
        self.libSizes        = libSizes
        self.sample          = sample
        self.seed            = seed
        self.includeData     = includeData
        self.embedded        = embedded
        self.validLib        = validLib
        self.noTime          = noTime
        self.ignoreNan       = ignoreNan
        self.verbose         = verbose

        # Set full lib & pred
        self.lib = self.pred = [ 1, self.Data.shape[0] ]

        self.CrossMapList  = None # List of CrossMap results
        self.libMeans      = None # DataFrame of CrossMap results
        self.PredictStats1 = None # DataFrame of CrossMap stats
        self.PredictStats2 = None # DataFrame of CrossMap stats

        # Setup
        self.Validate() # CCM Method

        # Instantiate Forward and Reverse Mapping objects
        # Each __init__ calls Validate() & CreateIndices()
        # and sets up targetVec, allTime
        self.FwdMap = SimplexClass( dataFrame       = dataFrame,
                                    columns         = columns,
                                    target          = target,
                                    lib             = self.lib,
                                    pred            = self.pred,
                                    E               = E,
                                    Tp              = Tp,
                                    knn             = knn,
                                    tau             = tau,
                                    exclusionRadius = exclusionRadius,
                                    embedded        = embedded,
                                    validLib        = validLib,
                                    noTime          = noTime,
                                    ignoreNan       = ignoreNan,
                                    verbose         = verbose )

        self.RevMap = SimplexClass( dataFrame       = dataFrame,
                                    columns         = target,
                                    target          = columns,
                                    lib             = self.lib,
                                    pred            = self.pred,
                                    E               = E,
                                    Tp              = Tp,
                                    knn             = knn,
                                    tau             = tau,
                                    exclusionRadius = exclusionRadius,
                                    embedded        = embedded,
                                    validLib        = validLib,
                                    noTime          = noTime,
                                    ignoreNan       = ignoreNan,
                                    verbose         = verbose )

    #-------------------------------------------------------------------
    # Methods
    #-------------------------------------------------------------------
    def Project( self, sequential = False ) :
        '''CCM both directions with CrossMap()'''

        if self.verbose:
            print( f'{self.name}: Project()' )

        if sequential : # Sequential alternative to multiprocessing
            FwdCM = self.CrossMap( 'FWD' )
            RevCM = self.CrossMap( 'REV' )
            self.CrossMapList = [ FwdCM, RevCM ]
        else :
            # multiprocessing Pool CrossMap both directions simultaneously
            poolArgs = [ 'FWD', 'REV' ]
            with Pool( processes = 2 ) as pool :
                CrossMapList = pool.map( self.CrossMap, poolArgs )

            self.CrossMapList = CrossMapList

        FwdCM, RevCM = self.CrossMapList

        self.libMeans = \
            DataFrame( {'LibSize' : FwdCM['libRho'].keys(),
                        f"{FwdCM['columns'][0]}:{FwdCM['target'][0]}" :
                        FwdCM['libRho'].values(),
                        f"{RevCM['columns'][0]}:{RevCM['target'][0]}" :
                        RevCM['libRho'].values() } )

        if self.includeData :
            FwdCMStats = FwdCM['predictStats'] # key libSize : list of CE dicts
            RevCMStats = RevCM['predictStats']

            FwdCMDF = []
            for libSize in FwdCMStats.keys() :
                LibSize  = [libSize] * self.sample # this libSize sample times
                libStats = FwdCMStats[libSize]     # sample ComputeError dicts

                libStatsDF = DataFrame( libStats )
                libSizeDF  = DataFrame( { 'LibSize' : LibSize } )
                libDF      = concat( [libSizeDF, libStatsDF], axis = 1 )

                FwdCMDF.append( libDF )

            RevCMDF = []
            for libSize in RevCMStats.keys() :
                LibSize  = [libSize] * self.sample # this libSize sample times
                libStats = RevCMStats[libSize]     # sample ComputeError dicts

                libStatsDF = DataFrame( libStats )
                libSizeDF  = DataFrame( { 'LibSize' : LibSize } )
                libDF      = concat( [libSizeDF, libStatsDF], axis = 1 )

                RevCMDF.append( libDF )

            FwdStatDF = concat( FwdCMDF, axis = 0 )
            RevStatDF = concat( RevCMDF, axis = 0 )

            self.PredictStats1 = FwdStatDF
            self.PredictStats2 = RevStatDF

    #-------------------------------------------------------------------
    # 
    #-------------------------------------------------------------------
    def CrossMap( self, direction ) :
        if self.verbose:
            print( f'{self.name}: CrossMap()' )

        if direction == 'FWD' :
            S = self.FwdMap
        elif direction == 'REV' :
            S = self.RevMap
        else :
            raise RuntimeError( f'{self.name}: CrossMap() Invalid Map' )

        # Create random number generator : None sets random state from OS
        RNG = default_rng( self.seed )

        # Copy S.lib_i since it's replaced every iteration
        lib_i   = S.lib_i.copy()
        N_lib_i = len( lib_i )

        libRhoMap  = {} # Output dict libSize key : mean rho value
        libStatMap = {} # Output dict libSize key : list of ComputeError dicts

        # Loop for library sizes
        for libSize in self.libSizes :
            rhos = zeros( self.sample )
            if self.includeData :
                predictStats = [None] * self.sample

            # Loop for subsamples
            for s in range( self.sample ) :
                # Generate library row indices for this subsample
                rng_i = RNG.choice( lib_i, size = min( libSize, N_lib_i ),
                                    replace = False )

                S.lib_i = rng_i

                S.FindNeighbors() # Depends on S.lib_i

                # Code from Simplex:Project ---------------------------------
                # First column is minimum distance of all N pred rows
                minDistances = S.knn_distances[:,0]
                # In case there is 0 in minDistances: minWeight = 1E-6
                minDistances = fmax( minDistances, 1E-6 )

                # Divide each column of N x k knn_distances by minDistances
                scaledDistances = divide(S.knn_distances, minDistances[:,None])
                weights         = exp( -scaledDistances )  # Npred x k
                weightRowSum    = sum( weights, axis = 1 ) # Npred x 1

                # Matrix of knn_neighbors + Tp defines library target values
                knn_neighbors_Tp = S.knn_neighbors + self.Tp      # Npred x k

                libTargetValues = zeros( knn_neighbors_Tp.shape ) # Npred x k
                for j in range( knn_neighbors_Tp.shape[1] ) :
                    libTargetValues[ :, j ][ :, None ] = \
                        S.targetVec[ knn_neighbors_Tp[ :, j ] ]
                # Code from Simplex:Project ----------------------------------

                # Projection is average of weighted knn library target values
                projection_ = sum( weights * libTargetValues,
                                   axis = 1) / weightRowSum

                # Align observations & predictions as in FormatProjection()
                # Shift projection_ by Tp
                projection_ = roll( projection_, S.Tp )
                if S.Tp > 0 :
                    projection_[ :S.Tp ] = nan
                elif S.Tp < 0 :
                    projection_[ S.Tp: ] = nan

                err = ComputeError( S.targetVec[ S.pred_i, 0 ],
                                    projection_, digits = 5 )

                rhos[ s ] = err['rho']

                if self.includeData :
                    predictStats[s] = err

            libRhoMap[ libSize ] = mean( rhos )

            if self.includeData :
                libStatMap[ libSize ] = predictStats

        # Reset S.lib_i to original
        S.lib_i = lib_i

        if self.includeData :
            return { 'columns' : S.columns, 'target' : S.target,
                     'libRho' : libRhoMap, 'predictStats' : libStatMap }
        else :
            return {'columns':S.columns, 'target':S.target, 'libRho':libRhoMap}

    #--------------------------------------------------------------------
    def Validate( self ):
    #--------------------------------------------------------------------
        if self.verbose:
            print( f'{self.name}: Validate()' )

        if not len( self.libSizes ) :
            raise RuntimeError(f'{self.name} Validate(): LibSizes required.')
        if not IsIterable( self.libSizes ) :
            self.libSizes = [ int(L) for L in self.libSizes.split() ]

        if self.sample == 0:
            raise RuntimeError(f'{self.name} Validate(): ' +\
                               'sample must be non-zero.')

        # libSizes
        #   if 3 arguments presume [start, stop, increment]
        #      if increment < stop generate the library sequence.
        #      if increment > stop presume list of 3 library sizes.
        #   else: Already list of library sizes.
        if len( self.libSizes ) == 3 :
            # Presume ( start, stop, increment ) sequence arguments
            start, stop, increment = [ int( s ) for s in self.libSizes ]

            # If increment < stop, presume start : stop : increment
            # and generate the sequence of library sizes
            if increment < stop :
                if increment < 1 :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes increment {increment} is invalid.'
                    raise RuntimeError( msg )

                if start > stop :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} stop {stop} are invalid.'
                    raise RuntimeError( msg )

                if start < self.E :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} less than E {self.E}'
                    raise RuntimeError( msg )
                elif start < 3 :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} less than 3.'
                    raise RuntimeError( msg )

                # Fill in libSizes sequence
                self.libSizes = [i for i in range(start, stop+1, increment)]

        if self.libSizes[-1] > self.Data.shape[0] :
            msg = f'{self.name} Validate(): ' +\
                  f'Maximum libSize {self.libSizes[-1]}'    +\
                  f' exceeds data size {self.Data.shape[0]}.'
            raise RuntimeError( msg )

        if self.libSizes[0] < self.E + 2 :
            msg = f'{self.name} Validate(): ' +\
                  f'Minimum libSize {self.libSizes[0]}'    +\
                  f' invalid for E={self.E}. Minimum {self.E + 2}.'
            raise RuntimeError( msg )

        if self.Tp < 0 :
            embedShift = abs( self.tau ) * ( self.E - 1 )
            maxLibSize = self.libSizes[-1]
            maxAllowed = self.Data.shape[0] - embedShift + (self.Tp + 1)
            if maxLibSize > maxAllowed :
                msg = f'{self.name} Validate(): Maximum libSize {maxLibSize}'  +\
                    f' too large for Tp {self.Tp}, E {self.E}, tau {self.tau}' +\
                    f' Maximum is {maxAllowed}'
                raise RuntimeError( msg )
