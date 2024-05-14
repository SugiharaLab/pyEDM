# python modules
from warnings import warn

# package modules
from pandas import DataFrame
from numpy  import any, append, array, concatenate, delete, isnan, where, zeros

# local modules
import pyEDM.API
from .AuxFunc import IsIterable

#--------------------------------------------------------------------
class EDM:
#--------------------------------------------------------------------
    '''EDM class : data container
       Simplex, SMap, CCM inherited from EDM'''

    def __init__( self, dataFrame, name = 'EDM' ):
        self.name = name

        self.Data           = dataFrame # DataFrame
        self.Embedding      = None      # DataFrame, includes nan
        self.Projection     = None      # DataFrame Simplex & SMap output

        self.lib_i          = None   # ndarray library indices
        self.pred_i         = None   # ndarray prediction indices
        self.pred_i_all     = None   # ndarray prediction indices, includes nan
        self.libOverlap     = False  # True if lib & pred overlap
        self.disjointLib    = False  # True if disjoint library
        self.ignoreNan      = True   # Remove nan from embedding
        self.xRadKnnFactor  = 5      # exlcusionRadius knn factor

        self.kdTree         = None  # SciPy KDTree (k-dimensional tree)
        self.knn_neighbors  = None  # ndarray (N_pred, knn) sorted
        self.knn_distances  = None  # ndarray (N_pred, knn) sorted

        self.projection_    = None  # ndarray Simplex & SMap output
        self.variance       = None  # ndarray Simplex & SMap output
        self.targetVec      = None  # ndarray entire record
        self.targetVecNan   = False # True if targetVec has nan : SMap only
        self.allTime        = None  # entire record

    #--------------------------------------------------------------------
    # Methods
    #--------------------------------------------------------------------
    from .Neighbors  import FindNeighbors
    from .Formatting import FormatProjection, FillTimes

    #--------------------------------------------------------------------
    def EmbedData( self ) :
    #--------------------------------------------------------------------
        '''Embed data : If not embedded call API.Embed()'''
        if self.verbose:
            print( f'{self.name}: EmbedData()' )

        if not self.embedded :
            self.Embedding = pyEDM.API.Embed( dataFrame = self.Data, E = self.E,
                                        tau = self.tau, columns = self.columns )
        else :
            self.Embedding = self.Data[ self.columns ] # Already an embedding 

    #--------------------------------------------------------------------
    def RemoveNan( self ) :
    #--------------------------------------------------------------------
        '''KDTree in Neighbors does not accept nan
           If ignoreNan remove Embedding rows with nan from lib_i, pred_i'''
        if self.verbose:
            print( f'{self.name}: RemoveNan()' )

        if self.ignoreNan :
            na_lib  = self.Embedding.iloc[self.lib_i,: ].isna().any(axis = 1)
            na_pred = self.Embedding.iloc[self.pred_i,:].isna().any(axis = 1)

            if na_lib.any() :
                if self.name == 'SMap' :
                    original_knn       = self.knn
                    original_lib_i_len = len( self.lib_i )

                # Redefine lib_i excluding nan
                self.lib_i = self.lib_i [ ~na_lib.to_numpy() ]

                # lib_i resized, update SMap self.knn if not user provided
                if self.name == 'SMap' :
                    if original_knn == original_lib_i_len - 1 :
                        self.knn = len( self.lib_i ) - 1

            # Redefine pred_i excluding nan
            if na_pred.any() : self.pred_i = self.pred_i[ ~na_pred.to_numpy() ]

            # If targetVec has nan, set flag for SMap internals
            if self.name == 'SMap' :
                if any( isnan( self.targetVec ) ) :
                    self.targetVecNan = True

    #--------------------------------------------------------------------
    def CreateIndices( self ):
    #--------------------------------------------------------------------
        '''Populate array index vectors lib_i, pred_i
           Indices specified in list of pairs [ 1,10, 31,40... ]
           where each pair is start:stop span of data rows.
        '''
        if self.verbose:
            print( f'{self.name}: CreateIndices() ' )

        #------------------------------------------------
        # lib_i from lib
        #------------------------------------------------
        # libPairs vector of start, stop index pairs
        if len( self.lib ) % 2 :
            # Odd number of lib
            msg = f'{self.name}: CreateIndices() lib must be an even ' +\
                'number of elements. Lib start : stop pairs'
            raise RuntimeError( msg )

        libPairs = [] # List of 2-tuples of lib indices
        for i in range( 0, len( self.lib ), 2 ) :
            libPairs.append( (self.lib[i], self.lib[i+1]) )

        # Validate end > start
        for libPair in libPairs :
            libStart, libEnd = libPair

            if self.name in [ 'Simplex', 'SMap', 'Multiview' ] :
                # Don't check if CCM since default of "1 1" is used.
                if libStart >= libEnd :
                    msg = f'{self.name}: CreateIndices() lib start ' +\
                        f' {libStart} exceeds lib end {libEnd}.'
                    raise RuntimeError( msg )

            # Disallow indices < 1, the user may have specified 0 start
            if libStart < 1 or libEnd < 1 :
                msg = f'{self.name}: CreateIndices() lib indices ' +\
                    f' less than 1 not allowed.'
                raise RuntimeError( msg )

        # Loop over each lib pair
        # Add rows for library segments, disallowing vectors
        # in disjoint library gap accommodating embedding and Tp
        embedShift = abs( self.tau ) * ( self.E - 1 )
        lib_i_list = list()

        for r in range( len( libPairs ) ) :
            start, stop = libPairs[ r ]

            # Adjust start, stop to enforce disjoint library gaps
            if not self.embedded :
                if self.tau < 0 :
                    start = start + embedShift
                else :
                    stop = stop - embedShift

            if self.Tp < 0 :
                if not self.embedded :
                    start = max( start, start + abs( self.Tp ) - 1 )
            else :
                if ( r == len( libPairs ) - 1 ) :
                    stop = stop - self.Tp

            libPair_i = [ i - 1 for i in range( start, stop + 1 ) ]

            lib_i_list.append( array( libPair_i, dtype = int ) )

        # Concatenate lib_i_list into lib_i
        self.lib_i = concatenate( lib_i_list )

        if len( lib_i_list ) > 1 : self.disjointLib = True

        # Validate lib_i: E, tau, Tp combination
        if self.name in [ 'Simplex', 'SMap', 'CCM', 'Multiview' ] :
            vectorStart  = max( (self.E - 1) * self.tau, 0 )
            vectorStart  = max( vectorStart, self.Tp )
            vectorEnd    = min( (self.E - 1) * self.tau, self.Tp )
            vectorEnd    = min( vectorEnd, 0 )
            vectorLength = abs( vectorStart - vectorEnd ) + 1

            if vectorLength > len( self.lib_i ) :
                msg = f'{self.name}: CreateIndices(): Combination of E = ' +\
                f'{self.E}  Tp = {self.Tp}  tau = {self.tau} is invalid '  +\
                'for the library.'
                raise RuntimeError( msg )

        #------------------------------------------------
        # pred_i from pred
        #------------------------------------------------
        # predPairs vector of start, stop index pairs
        if len( self.pred ) % 2 :
            # Odd number of pred
            msg = f'{self.name}: CreateIndices() pred must be an even ' +\
                'number of elements. Pred start : stop pairs'
            raise RuntimeError( msg )

        predPairs = [] # List of 2-tuples of pred indices
        for i in range( 0, len( self.pred ), 2 ) :
            predPairs.append( (self.pred[i], self.pred[i+1]) )

        # Validate end > start
        for predPair in predPairs :
            predStart, predEnd = predPair

            if self.name in [ 'Simplex', 'SMap', 'Multiview' ] :
                # Don't check CCM since default of "1 1" is used.
                if predStart >= predEnd :
                    msg = f'{self.name}: CreateIndices() pred start ' +\
                        f' {predStart} exceeds pred end {predEnd}.'
                    raise RuntimeError( msg )

            # Disallow indices < 1, the user may have specified 0 start
            if predStart < 1 or predEnd < 1 :
                msg = f'{self.name}: CreateIndices() pred indices ' +\
                    ' less than 1 not allowed.'
                raise RuntimeError( msg )

        # Create indices of prediction pred_i
        nPred = 0
        for predPair in predPairs :
            nPred = nPred + (predPair[1] - predPair[0] + 1)
        self.pred_i = zeros( nPred, dtype = int )

        i = 0
        for r in range( len( predPairs ) ) :
            start, stop = predPairs[ r ]

            for j in range( start, stop + 1 ) :
                self.pred_i[ i ] = j - 1  # apply zero-offset
                i = i + 1

        # Remove embedShift nan at ends if needed
        if not self.embedded :
            # If [0, 1, ... embedShift] nan (negative tau) or
            # [N - embedShift, ... N-1, N]  (positive tau) nan
            # are in pred_i delete elements
            if self.tau > 0 :
                nan_i = [ self.Data.shape[0] - 1 - i
                          for i in range( embedShift ) ]
            else :
                nan_i = [ i for i in range( embedShift ) ]

            if nan_i :
                p_i = [ where( self.pred_i == i ) for i in nan_i ]
                self.pred_i = delete( self.pred_i, p_i )

        self.pred_i_all = self.pred_i.copy() # Before internal nan are removed

        # Warn about disjoint prediction sets: not supported
        if len( predPairs ) > 1 :
            msg = f'{self.name}: CreateIndices(): ' +\
                ' Disjoint prediction sets not supported.'
            warn( msg )

        # Validate lib_i pred_i do not exceed data
        if self.lib_i[-1] >= self.Data.shape[0] :
            msg = f'{self.name}: CreateIndices() The prediction index ' +\
                f'{self.lib_i[-1]} exceeds the number of data rows ' +\
                f'{self.Data.shape[0]}'
            raise RuntimeError( msg )

        if self.pred_i[-1] >= self.Data.shape[0] :
            msg = f'{self.name}: CreateIndices() The prediction index ' +\
                f'{self.pred_i[-1]} exceeds the number of data rows ' +\
                f'{self.Data.shape[0]}'
            raise RuntimeError( msg )

        # Check for lib : pred overlap for knn leave-one-out
        if len( set( self.lib_i ).intersection( set( self.pred_i ) ) ) :
                self.libOverlap = True

        if self.name == 'SMap' :
            if self.knn < 1 :  # default knn = 0, set knn value to full lib
                self.knn = len( self.lib_i ) - 1

                if self.verbose :
                    msg = f'{self.name} CreateIndices(): ' +\
                        f'Set knn = {self.knn} for SMap.'
                    print( msg, flush = True )

    #--------------------------------------------------------------------
    def Validate( self ):
    #--------------------------------------------------------------------
        if self.verbose:
            print( f'{self.name}: Validate()' )

        if self.Data is None :
            raise RuntimeError(f'Validate() {self.name}: dataFrame required.')
        else :
            if not isinstance( self.Data, DataFrame ) :
                raise RuntimeError(f'Validate() {self.name}: dataFrame ' +\
                                   'is not a Pandas DataFrame.')

        if not self.columns :
            raise RuntimeError( f'Validate() {self.name}: columns required.' )
        if not IsIterable( self.columns ) :
            self.columns = self.columns.split()

        for column in self.columns :
            if not column in self.Data.columns :
                raise RuntimeError( f'Validate() {self.name}: column ' +\
                                    f'{column} not found in dataFrame.' )

        if not self.target :
            raise RuntimeError( f'Validate() {self.name}: target required.' )
        if not IsIterable( self.target ) :
            self.target = self.target.split()

        for target in self.target :
            if not target in self.Data.columns :
                raise RuntimeError( f'Validate() {self.name}: target ' +\
                                    f'{target} not found in dataFrame.' )

        if not self.embedded :
            if self.tau == 0 :
                raise RuntimeError(f'Validate() {self.name}:' +\
                                   ' tau must be non-zero.')
            if self.E < 1 :
                raise RuntimeError(f'Validate() {self.name}:' +\
                                   f' E = {self.E} is invalid.')

        if self.name != 'CCM' :
            if not self.lib :
                raise RuntimeError( f'Validate() {self.name}: lib required.' )
            if not IsIterable( self.lib ) :
                self.lib = [ int(i) for i in self.lib.split() ]

            if not self.pred :
                raise RuntimeError( f'Validate() {self.name}: pred required.' )
            if not IsIterable( self.pred ) :
                self.pred = [ int(i) for i in self.pred.split() ]

        # Set knn default based on E and lib size, E embedded on num columns
        if self.name in [ 'Simplex', 'CCM', 'Multiview' ] :
            # embedded = true: Set E to number of columns
            if self.embedded :
                self.E = len( self.columns )

            # knn not specified : knn set to E+1
            if self.knn < 1 :
                self.knn = self.E + 1

                if self.verbose :
                    msg = f'{self.name} Validate(): Set knn = {self.knn}'
                    print( msg, flush = True )

        elif self.name == 'SMap' :
            # embedded = true: Set E to number of columns
            if self.embedded and len( self.columns ) :
                self.E = len( self.columns )

            if not self.embedded and len( self.columns ) > 1 :
                msg = f'{self.name} Validate(): Multivariable S-Map ' +\
                'must use embedded = True to ensure data/dimension '  +\
                'correspondance.'
                raise RuntimeError( msg )
