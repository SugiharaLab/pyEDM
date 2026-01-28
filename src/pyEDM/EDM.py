# python modules
from warnings import warn
from datetime import datetime

# package modules
import numpy as np
from pandas import DataFrame
from numpy  import any, append, array, concatenate, isnan, zeros

# local modules
import pyEDM.API
from .AuxFunc import IsIterable
from .NeighborFinder import KDTreeNeighborFinder, PairwiseDistanceNeighborFinder


#--------------------------------------------------------------------
class EDM:
#--------------------------------------------------------------------
    '''EDM class : data container
       Simplex, SMap, CCM inherited from EDM'''

    def __init__(self, dataFrame, neighbor_algorithm = 'kdtree', name = 'EDM'):
        self.name = name

        self.Data          = dataFrame # DataFrame
        self.Embedding     = None      # DataFrame, includes nan
        self.Projection    = None      # DataFrame Simplex & SMap output

        self.lib_i         = None  # ndarray library indices
        self.pred_i        = None  # ndarray prediction indices : nan removed
        self.pred_i_all    = None  # ndarray prediction indices : nan included
        self.predList      = []    # list of disjoint pred_i_all
        self.disjointLib   = False # True if disjoint library
        self.libOverlap    = False # True if lib & pred overlap
        self.ignoreNan     = True  # Remove nan from embedding
        self.xRadKnnFactor = 5     # exlcusionRadius knn factor

        self.kdTree        = None  # SciPy KDTree (k-dimensional tree)
        self.knn_neighbors = None  # ndarray (N_pred, knn) sorted
        self.knn_distances = None  # ndarray (N_pred, knn) sorted

        self.projection    = None  # ndarray Simplex & SMap output
        self.variance      = None  # ndarray Simplex & SMap output
        self.targetVec     = None  # ndarray entire record
        self.targetVecNan  = False # True if targetVec has nan : SMap only
        self.time          = None  # ndarray entire record numerically operable

        self.neighbor_algorithm = neighbor_algorithm
        self.neighbor_finder = None

    #--------------------------------------------------------------------
    # Methods
    #--------------------------------------------------------------------
    from .Formatting import FormatProjection, ConvertTime, AddTime

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
           If ignoreNan remove Embedding rows with nan from lib_i, pred_i
        '''
        if self.verbose:
            print( f'{self.name}: RemoveNan()' )

        if self.ignoreNan :
            # type : <class 'pandas.core.series.Series'>
            # Series of bool for all Embedding columns (axis = 1) of lib_i...
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
            if any( na_pred ) :
                self.pred_i = self.pred_i[ ~na_pred ]

            # If targetVec has nan, set flag for SMap internals
            if self.name == 'SMap' :
                if any( isnan( self.targetVec ) ) :
                    self.targetVecNan = True

        self.PredictionValid()

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

        #------------------------------------------------
        # Validate lib_i: E, tau, Tp combination
        #------------------------------------------------
        if self.name in [ 'Simplex', 'SMap', 'CCM', 'Multiview' ] :
            if self.embedded :
                if len( self.lib_i ) < abs( self.Tp ) :
                    msg = f'{self.name}: CreateIndices(): embbeded True ' +\
                          f'Tp = {self.Tp} is invalid for the library.'
                    raise RuntimeError( msg )
            else :
                vectorStart  = max( [ -embedShift, 0, self.Tp ] )
                vectorEnd    = min( [ -embedShift, 0, self.Tp ] )
                vectorLength = abs( vectorStart - vectorEnd ) + 1

                if vectorLength > len( self.lib_i ) :
                    msg = f'{self.name}: CreateIndices(): Combination of E = '+\
                          f'{self.E}  Tp = {self.Tp}  tau = {self.tau} ' +\
                          'is invalid for the library.'
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

        if len( predPairs ) > 1 : self.disjointPred = True

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

        # Create pred_i indices from predPairs
        for r in range( len( predPairs ) ) :
            start, stop = predPairs[ r ]
            pred_i      = zeros( stop - start + 1, dtype = int )

            i = 0
            for j in range( start, stop + 1 ) :
                pred_i[ i ] = j - 1  # apply zero-offset
                i = i + 1

            self.predList.append( pred_i ) # Append disjoint segment(s)

        # flatten arrays in self.predList for single array self.pred_i
        pred_i_ = []
        for pred_i in self.predList :
            i_      = [i for i in pred_i]
            pred_i_ = pred_i_ + i_

        self.pred_i = array( pred_i_, dtype = int )

        self.PredictionValid()

        self.pred_i_all = self.pred_i.copy() # Before nan are removed

        # Remove embedShift nan from predPairs
        # NOTE : This does NOT redefine self.pred_i, only self.predPairs
        #        self.pred_i is redefined to remove all nan in RemoveNan()
        #        at the API level.
        if not self.embedded :
            # If [0, 1, ... embedShift] nan (negative tau) or
            # [N - embedShift, ... N-1, N]  (positive tau) nan
            # are in pred_i delete elements
            nan_i_start = [ i for i in range( embedShift ) ]
            nan_i_end   = [ self.Data.shape[0]-1-i for i in range(embedShift) ]

            for i in range( len( self.predList ) ) :
                pred_i = self.predList[i]
                nan_i  = None

                if self.tau > 0 :
                    if any( [ i in nan_i_end for i in pred_i ] ) :
                        pred_i_ = [i for i in pred_i if i not in nan_i_end]
                        self.predList[i] = array( pred_i_, dtype = int )
                else :
                    if any( [ i in nan_i_start for i in pred_i ] ) :
                        pred_i_ = [i for i in pred_i if i not in nan_i_start]
                        self.predList[i] = array( pred_i_, dtype = int )

        #------------------------------------------------
        # Validate lib_i pred_i do not exceed data
        #------------------------------------------------
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

        #---------------------------------------------------
        # Check for lib : pred overlap for knn leave-one-out
        #---------------------------------------------------
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
    def PredictionValid( self ) :
    #--------------------------------------------------------------------
        '''Validate there are pred_i to make a prediction
        '''
        if self.verbose:
            print( f'{self.name}: PredictionValid()' )

        if len( self.pred_i ) == 0 :
            msg = f'{self.name}: PredictionValid() No valid prediction ' +\
                'indices. Examine pred, E, tau, Tp parameters and/or nan.'
            warn( msg )

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

        if not len( self.columns ) :
            raise RuntimeError( f'Validate() {self.name}: columns required.' )
        if not IsIterable( self.columns ) :
            self.columns = self.columns.split()

        for column in self.columns :
            if not column in self.Data.columns :
                raise RuntimeError( f'Validate() {self.name}: column ' +\
                                    f'{column} not found in dataFrame.' )

        if not len( self.target ) :
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
            if not len( self.lib ) :
                raise RuntimeError( f'Validate() {self.name}: lib required.' )
            if not IsIterable( self.lib ) :
                self.lib = [ int(i) for i in self.lib.split() ]

            if not len( self.pred ) :
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

        if self.name == 'SMap' :
            # embedded = true: Set E to number of columns
            if self.embedded and len( self.columns ) :
                self.E = len( self.columns )

            if not self.embedded and len( self.columns ) > 1 :
                msg = f'{self.name} Validate(): Multivariable S-Map ' +\
                'must use embedded = True to ensure data/dimension '  +\
                'correspondance.'
                raise RuntimeError( msg )

        if self.generateSteps > 0 :
            # univariate only, embedded must be False
            if self.name in [ 'Simplex', 'SMap' ] :

                if self.embedded :
                    msg = f'{self.name} Validate(): generateSteps > 0 ' +\
                        'must use univariate embedded = False.'
                    raise RuntimeError( msg )

                if self.target[0] != self.columns[0] :
                    msg = f'{self.name} Validate(): generateSteps > 0 ' +\
                          f'must use univariate target ({self.target[0]}) ' +\
                          f' == columns ({self.columns[0]}).'
                    raise RuntimeError( msg )

                # If times are datetime, AddTime() fails
                # EDM.time is ndarray storing python datetime
                # In AddTime() datetime, timedelta operations are not compatible
                # with numpy datetime64, timedelta64 : deltaT fails in conversion
                # If times are datetime: raise exception
                if not self.noTime :
                    try:
                        time0 = self.Data.iloc[ 0, 0 ] # self.time not yet
                        dt0   = datetime.fromisoformat( time0 )
                    except :
                        # dt0 is not a datetime assign for finally to pass
                        dt0 = None
                    finally :
                        # if dt0 is datetime, raise exception for noTime = True
                        if isinstance( dt0, datetime ) :
                            msg = f'{self.name} Validate(): generateSteps ' +\
                                'with datetime needs to use noTime = True.'
                            raise RuntimeError( msg )


    @property
    def exclusionRadius_knn(self):
        """
        Is knn_neighbors exclusionRadius radius adjustment needed?
        Refactored out of FindNeighbors
        """
        out = False
        if self.exclusionRadius > 0:
            if self.libOverlap:
                out = True
            else:
                # If no libOverlap and exclusionRadius is less than the
                # distance in rows between lib : pred, no library neighbor
                # exclusion needed.
                # Find row span between lib & pred
                excludeRow = 0
                if self.pred_i[0] > self.lib_i[-1]:
                    # pred start is beyond lib end
                    excludeRow = self.pred_i[0] - self.lib_i[-1]
                elif self.lib_i[0] > self.pred_i[-1]:
                    # lib start row is beyond pred end
                    excludeRow = self.lib_i[0] - self.pred_i[-1]
                if self.exclusionRadius >= excludeRow:
                   out = True
        return out


    def check_lib_valid(self):
        """
        Refactored out of FindNeighbors
        :return: 
        """
        if len(self.validLib):
            # Convert self.validLib boolean vector to data indices
            data_i = array([i for i in range(self.Data.shape[0])],
                           dtype = int)
            validLib_i = data_i[self.validLib.to_numpy()]
    
            # Filter lib_i to only include valid library points
            lib_i_valid = array([i for i in self.lib_i if i in validLib_i],
                                dtype = int)
    
            if len(lib_i_valid) == 0:
                msg = f'{self.name}: FindNeighbors() : ' + \
                      'No valid library points found. ' + \
                      'All library points excluded by validLib.'
                raise ValueError(msg)
    
            if len(lib_i_valid) < self.knn:
                msg = f'{self.name}: FindNeighbors() : Only {len(lib_i_valid)} ' + \
                      f'valid library points found, but knn={self.knn}. ' + \
                      'Reduce knn or check validLib.'
                warn(msg)
    
            # Replace lib_ with lib_i_valid
            self.lib_i = lib_i_valid

    @property
    def knn_(self):
        """
        K nearest neighbors to actually query for
        Factored out of FindNeighbors()
        """
        knn_ = self.knn
        if self.libOverlap and not self.exclusionRadius_knn :
            # Increase knn +1 if libOverlap
            # Returns one more column in knn_distances, knn_neighbors
            # The first nn degenerate with the prediction vector
            # is replaced with the 2nd to knn+1 neighbors
            knn_ = knn_ + 1

        elif self.exclusionRadius_knn :
            # knn_neighbors exclusionRadius adjustment required
            # Ask for enough knn to discard exclusionRadius neighbors
            # This is controlled by the factor: self.xRadKnnFactor
            # JP : Perhaps easier to just compute all neighbors?
            knn_ = min( knn_ * self.xRadKnnFactor, len( self.lib_i ) )

        if len( self.validLib ) :
            # Have to examine all knn
            knn_ = len( self.lib_i )
        return knn_


    def map_kdtree_knn_indices_to_data(self, raw_distances, raw_neighbors):
        """
        -----------------------------------------------
        Shift KDTree knn_neighbors to lib_i reference
        -----------------------------------------------
        NeighborFinders returns knn referenced to embedding.iloc[self.lib_i,:]
        where returned knn_neighbors are indexed from 0 : len( lib_i ).
        """

            # Step 1: Map KDTree indices to actual data indices
        knn_neighbors = self.map_knn_indices_to_library_indices(raw_neighbors)
        knn_distances = raw_distances

        # Step 2: Remove degenerate neighbors (leave-one-out)
        knn_neighbors, knn_distances = self._remove_degenerate_neighbors(
            knn_neighbors, knn_distances
        )

        # Step 3: Apply exclusion radius filtering
        knn_neighbors, knn_distances = self._apply_exclusion_radius(
            knn_neighbors, knn_distances
        )

        return knn_distances, knn_neighbors


    def map_knn_indices_to_library_indices(self, raw_neighbors):
        """
        Maps KDTree indices to actual library data row indices.

        Args:
            raw_neighbors: KDTree query results with indices relative to library data

        Returns:
            ndarray: Neighbor indices mapped to actual data row numbers
        """
        # Handle contiguous library case
        if not self.disjointLib and \
                self.lib_i[-1] - self.lib_i[0] + 1 == len(self.lib_i):
            return raw_neighbors + self.lib_i[0]
        return np.array(self.lib_i[raw_neighbors]).astype(int)

    def _remove_degenerate_neighbors(self, knn_neighbors, knn_distances):
        """
        Removes self-matching neighbors when library overlaps with prediction set.
        Implements leave-one-out validation by shifting neighbors when first nn is degenerate.

        Args:
            knn_neighbors: Neighbor indices
            knn_distances: Neighbor distances

        Returns:
            tuple: (filtered_neighbors, filtered_distances)
        """
        if not self.libOverlap:
            return knn_neighbors, knn_distances

        # Identify degenerate rows where pred_i == first neighbor
        knn_neighbors_0 = knn_neighbors[:, 0]
        i_overlap = [i == j for i, j in zip(self.pred_i, knn_neighbors_0)]

        # Shift columns: move col[1:knn_] into col[0:(knn_-1)]
        J = knn_distances.shape[1]
        knn_distances[i_overlap, 0:(J - 1)] = knn_distances[i_overlap, 1:J]
        knn_neighbors[i_overlap, 0:(J - 1)] = knn_neighbors[i_overlap, 1:J]

        # Remove extra column if not doing exclusion radius filtering
        if not self.exclusionRadius_knn:
            knn_distances = knn_distances[:, :-1]
            knn_neighbors = knn_neighbors[:, :-1]

        return knn_neighbors, knn_distances


    def _apply_exclusion_radius(self, knn_neighbors, knn_distances):
        """
        Filters neighbors within temporal exclusion radius.
        For each prediction row, selects k neighbors outside exclusionRadius.

        Args:
            knn_neighbors: Neighbor indices
            knn_distances: Neighbor distances

        Returns:
            tuple: (filtered_neighbors, filtered_distances)
        """
        if not self.exclusionRadius_knn:
            return knn_neighbors, knn_distances

        def _SelectValidNeighbors(knnRow, knnDist, excludeRow):
            """Select knn neighbors not in excludeRow"""
            valid_neighbors = np.full(self.knn, -1E6, dtype = int)
            valid_distances = np.full(self.knn, -1E6, dtype = float)

            k = 0
            for r in range(len(knnRow)):
                if knnRow[r] not in excludeRow:
                    valid_neighbors[k] = knnRow[r]
                    valid_distances[k] = knnDist[r]
                    k += 1
                    if k == self.knn:
                        break

            # Warn if couldn't find enough valid neighbors
            if -1E6 in valid_neighbors:
                valid_neighbors = knnRow[:self.knn]
                valid_distances = knnDist[:self.knn]
                warn(f'{self.name}: Failed to find {self.knn} neighbors outside '
                     f'exclusionRadius {self.exclusionRadius}. Consider reducing knn.')

            return valid_neighbors, valid_distances

        # Apply exclusion for each prediction row
        for i in range(len(self.pred_i)):
            pred_i = self.pred_i[i]
            rowLow = max(self.lib_i.min(), pred_i - self.exclusionRadius)
            rowHi = min(self.lib_i.max(), pred_i + self.exclusionRadius)
            excludeRow = list(range(rowLow, rowHi + 1))

            knn_neighbors[i, :self.knn], knn_distances[i, :self.knn] = \
                _SelectValidNeighbors(knn_neighbors[i, :], knn_distances[i, :], excludeRow)

        # Remove extra knn_ columns
        extra_cols = list(range(self.knn, knn_distances.shape[1]))
        knn_distances = np.delete(knn_distances, extra_cols, axis = 1)
        knn_neighbors = np.delete(knn_neighbors, extra_cols, axis = 1)

        return knn_neighbors, knn_distances


    def _build_exclusion_mask(self):
        """
        Pre-compute boolean exclusion mask for pairwise distance neighbor queries.
        Encodes both degenerate neighbors (libOverlap) and exclusionRadius.
    
        :returns: (n_pred, n_lib) where True = exclude neighbor
        """
    
        n_pred = len(self.pred_i)
        n_lib = len(self.lib_i)
        mask = np.zeros((n_lib, n_pred), dtype = bool)
        if not self.libOverlap and not self.exclusionRadius_knn:
            return mask
    
        # Initialize mask: False = include neighbor
    
        # Build index lookup for library
        lib_index_map = {data_index: mask_index for mask_index, data_index in enumerate(self.lib_i)}

        for i, pred_index in enumerate(self.pred_i):
            # remove self from neighbor map
            if pred_index in self.lib_i:
                mask_pos = lib_index_map[pred_index]
                mask[mask_pos, i] = True

            # Handle exclusion radius
            if self.exclusionRadius_knn:
                rowLow = max(np.min(self.lib_i), pred_index - self.exclusionRadius)
                rowHi = min(np.max(self.lib_i), pred_index + self.exclusionRadius)

                # Mark all library indices in exclusion window
                for j, lib_idx in enumerate(self.lib_i):
                    if rowLow <= lib_idx <= rowHi:
                        mask[j, i] = True
    
        return mask


    def FindNeighbors(self):
        # --------------------------------------------------------------------
        '''Find neighbors in the lib set for the pred set

           NeighborFinder objects returns ndarray of knn_neighbors as indices
           with respect to the data array passed in, not with respect to the lib_i
           of embedding[ lib_i ] passed to in. Since lib_i are generally
           not [0..N] the knn_neighbors need to be adjusted to lib_i reference
           for use in projections. If the the library is unitary this is
           a simple shift by lib_i[0]. If the library has disjoint segments
           or unordered indices, a mapping is needed from KDTree to lib_i.

           If there are degenerate lib & pred indices the first nn will
           be the prediction vector itself with distance 0. These are removed
           to implement "leave-one-out" prediction validation. In this case
           self.libOverlap is set True and the value of knn is increased
           by 1 to return an additional nn. The first nn is relplaced by
           shifting the j = 1:knn+1 knn columns into the j = 0:knn columns.

           If exlcusionRadius > 0, and, there are degenerate lib & pred
           indices, or, if there are not degnerate lib & pred but the
           distance in rows between the lib & pred gap is less than
           exlcusionRadius, knn_neighbors have to be selected for each
           pred row to exclude library neighbors within exlcusionRadius.
           This is done by increasing knn by a factor of
           self.xRadKnnFactor, then selecting valid nn.

           Writes to EDM object:
             knn_distances : sorted knn distances
             knn_neighbors : library neighbor rows of knn_distances
        '''
        if self.verbose:
            print(f'{self.name}: FindNeighbors()')

        self.check_lib_valid()

        train = self.Embedding.iloc[self.lib_i, :].to_numpy()
        test = self.Embedding.iloc[self.pred_i, :].to_numpy()

        # make neighbor finder and query
        if self.neighbor_algorithm == 'kdtree':
            self.neighbor_finder = KDTreeNeighborFinder(train,
                                                        leafsize = 20,
                                                        compact_nodes = True,
                                                        balanced_tree = True)
        elif self.neighbor_algorithm == 'pdist':
            self.neighbor_finder = PairwiseDistanceNeighborFinder(train, exclusion = self._build_exclusion_mask())
        else:
            raise ValueError('Unknown neighbor finding algorithm {}'.format(self.neighbor_algorithm))

        # -----------------------------------------------
        # Query prediction set
        # -----------------------------------------------
        numThreads = -1  # Use all CPU threads in kdTree.query

        # exclusions are baked into pdist, and no need to  expand radii
        k = self.knn_ if self.neighbor_algorithm == 'kdtree' or len(self.validLib) > 0 else self.knn
        raw_distances, raw_neighbors = self.neighbor_finder.query(test,
                                                                  k = k,
                                                                  eps = 0,
                                                                  p = 2,
                                                                  workers = numThreads)

        if self.neighbor_algorithm == 'kdtree':
            # KDTree still needs full map-to-index, exclude, and prune
            self.knn_distances, self.knn_neighbors = self.map_kdtree_knn_indices_to_data(
                raw_distances, raw_neighbors
            )
        else:
            if len(self.validLib) > 0:
                raw_neighbors = raw_neighbors[:, :-1]
                raw_distances = raw_distances[:, :-1]

            # pdist only need index remapping
            self.knn_neighbors = self.map_knn_indices_to_library_indices(raw_neighbors)
            self.knn_distances = raw_distances

        # the code expects a 2D array
        if self.knn == 1:
            self.knn_distances = self.knn_distances[:, None]
            self.knn_neighbors = self.knn_neighbors[:, None]