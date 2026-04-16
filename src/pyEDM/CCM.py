"""
Convergent Cross Mapping (CCM)

Strategy: precompute the delay embeddings and validity filtering once
then for each subsample build a small KDTree from L library embedding
vectors and query all M valid points for k nearest neighbors.

At each library size L and each sample iteration:
 - L points are randomly selected as the library.
 - The prediction set is all M valid points (lib and pred span the data).
 - A KDTree is built from the L library embedding vectors.
 - All M points are queried against the L-point tree for k neighbors.
 - Simplex weights and cross-map predictions are computed for all M.
 - Pearson correlation is computed over all M prediction/actual pairs.

Multivariate embedding:
 - Both ``columns`` and ``target`` accept multiple variable names
   (space-delimited string, comma-delimited string, or list of strings).
 - The embedding for a direction is the horizontal stack of per-variable
   E-dimensional time-delay embeddings (mixed-multivariate embedding).
 - The prediction target is always univariate: the first entry of
   ``target`` for the forward map, the first entry of ``columns`` for
   the reverse map.

Tp filters the valid index vectors (points without a valid target at
t+Tp are removed before any computation). exclusionRadius filters
the per-subsample KDTree query results (temporally close neighbors
are replaced by the next-nearest).

Only ``forkserver`` and ``spawn`` multiprocessing contexts allowed. 
Two modes of parallel processing are supported. With small models where
the 2 embedding arrays, 2 target arrays and 2 index arrays are less
than sharedMB Pool initializer is used which pickles the data and
distributes to each worker. If data is larger than sharedMB
multiprocessing.shared_memory is used avoiding per-worker pickle overhead.
On small models Pool initializer can be faster. With exclusionRadius
shared memory may be better.

NaN handling follows pyEDM conventions:
 - Rows with any NaN in the embedding are removed from the valid index.
 - Rows whose Tp-shifted target is NaN or out of bounds are removed.
 - Predictions that resolve to NaN (insufficient neighbors) are dropped
   pairwise with their actuals before computing Pearson correlation,
   mirroring pyEDM's ComputeError NaN removal.

Reference: George Sugihara et al., Detecting Causality in Complex Ecosystems.
           Science338, 496-500(2012). DOI:10.1126/science.1227079
"""

# Distribution
from multiprocessing import cpu_count, shared_memory
from multiprocessing import get_start_method, get_context

# Community
import numpy as np
from pandas import DataFrame
from scipy.spatial import KDTree

# Local
from .AuxFunc import IsIterable


# ====================================================================
# CCM class
# ====================================================================

class CCM:
    """
    Convergent Cross Mapping using per-subsample KDTree construction.

    Construction stores parameters only. Call ``Validate`` and ``Embed``
    to prepare the precomputed state, or call ``Project`` which ensures
    all preparation stages have run.

    Parameters
    ----------
    dataFrame : DataFrame
        Input data with named columns.
    columns : str or [str]
        Column name(s) for the library variable.
    target : str or [str]
        Column name for the target variable. Only target[0] used for Takens
    E : int
        Embedding dimension.
    Tp : int
        Prediction horizon in time steps. Default 0.
    knn : int
        Number of nearest neighbors. Default 0 is set to E+1 if Takens.
    tau : int
        Embedding delay (negative = rows back in time). Default -1.
    exclusionRadius : int
        Temporal exclusion radius for neighbor queries. Default 0.
    validLib : array-like of bool
        Boolean mask of valid library rows. Default: all True.
    embedded : bool
        If True, ``columns`` are already the embedding. Default False.
    libSizes : str or [int]
        Library sizes. Can be overridden in ``Project``.
    sample : int
        Number of subsamples per library size. Default 100.
    seed : int or None
        RNG seed for reproducibility.
    includeData : bool
        Add variance of sample CCM correlation at each libSize. Default False.
    parallel : bool or int
        Worker count. Default True.
    mpMethod : str or None
        Multiprocessing start method. Default None.
    sharedMB : float
        Total worker data size threshold in MB. Below this, worker
        arrays are passed via Pool initargs (pickle); above, they are
        placed in OS shared memory for zero-copy access.

    Attributes (available after corresponding stage)
    -------------------------------------------------
    After ``Validate``:
        tgt_vec, col_pred_vec, shifts, N

    After ``Embed``:
        embedding_col, embedding_tgt, valid_col, valid_tgt,
        idx_col, idx_tgt, embed_valid_col, embed_valid_tgt,
        pred_vals_col, pred_vals_tgt, k

    After ``Project``:
        Data — DataFrame passed in
        libMeans — DataFrame returned by the most recent Project call.
    """

    _validated = False
    _embedded  = False

    def __init__(self,
                 dataFrame,
                 columns,
                 target,
                 E,
                 Tp=0,
                 knn=0,
                 tau=-1,
                 exclusionRadius=0,
                 libSizes=[],
                 sample=30,
                 seed=None,
                 validLib=[],
                 embedded=False,
                 includeData=False,
                 parallel=True,
                 mpMethod=None,
                 sharedMB=0.01,
                 verbose=False):

        self.name            = 'CCM'
        self.Data            = dataFrame
        self.columns         = columns # -> [] Validate()
        self.target          = target  # -> [] Validate()
        self.E               = E
        self.Tp              = Tp
        self.knn             = knn # 0 -> E + 1 Validate()
        self.tau             = tau
        self.exclusionRadius = exclusionRadius
        self.libSizes        = libSizes
        self.sample          = sample
        self.seed            = seed
        self.validLib        = validLib
        self.embedded        = embedded
        self.includeData     = includeData
        self.parallel        = parallel
        self.mpMethod        = mpMethod
        self.sharedMB        = sharedMB
        self.verbose         = verbose
        self.libMeans        = None # DataFrame of CrossMap results

    # ================================================================
    # Stage 1: Validate
    # ================================================================

    def Validate(self):
        """Parse and validate inputs."""

        if self.Data is None :
            raise ValueError(f'Validate() {self.name}: dataFrame required.')
        else :
            if not isinstance( self.Data, DataFrame ) :
                raise ValueError(f'Validate() {self.name}: dataFrame ' +\
                                   'is not a Pandas DataFrame.')

        if not len( self.columns ) :
            raise ValueError( f'Validate() {self.name}: columns required.' )
        if not IsIterable( self.columns ) :
            self.columns = self.columns.split()

        for column in self.columns :
            if not column in self.Data.columns :
                raise ValueError( f'Validate() {self.name}: column ' +\
                                    f'{column} not found in dataFrame.' )

        if not len( self.target ) :
            raise ValueError( f'Validate() {self.name}: target required.' )
        if not IsIterable( self.target ) :
            self.target = self.target.split()

        for target in self.target :
            if not target in self.Data.columns :
                raise ValueError( f'Validate() {self.name}: target ' +\
                                    f'{target} not found in dataFrame.' )

        if not self.embedded :
            if self.tau == 0 :
                raise ValueError(f'Validate() {self.name}:' +\
                                   ' tau must be non-zero.')
            if self.E < 1 :
                raise ValueError(f'Validate() {self.name}:' +\
                                   f' E = {self.E} is invalid.')

        # Set knn default based on E and lib size
        if self.embedded : # embedded = true: Set E to number of columns
            self.E = len( self.columns )
        if self.knn < 1 : # knn not specified : knn set to E+1
            self.knn = self.E + 1

        # CCM specific
        if len( self.validLib ):
            if len( self.validLib ) != self.Data.shape[0] :
                raise ValueError(f'Validate() {self.name}:' +\
                                 f' validLib must be same length as data')
            if self.validLib.dtype != bool:
                raise ValueError(f'Validate() {self.name}:' +\
                                 f' validLib must be boolean array')

        if not len( self.libSizes ) :
            raise ValueError(f'{self.name} Validate(): LibSizes required.')

        if not IsIterable( self.libSizes ) :
            self.libSizes = [ int(L) for L in self.libSizes.split() ]

        if self.sample == 0:
            raise ValueError(f'{self.name} Validate(): ' +\
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
                    raise ValueError( msg )

                if start > stop :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} stop {stop} are invalid.'
                    raise ValueError( msg )

                if start < self.E :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} less than E {self.E}'
                    raise ValueError( msg )
                elif start < 3 :
                    msg = f'{self.name} Validate(): ' +\
                          f'libSizes start {start} less than 3.'
                    raise ValueError( msg )

                # Fill in libSizes sequence
                self.libSizes = [i for i in range(start, stop+1, increment)]

        if self.libSizes[-1] > self.Data.shape[0] :
            msg = f'{self.name} Validate(): ' +\
                  f'Maximum libSize {self.libSizes[-1]}'    +\
                  f' exceeds data size {self.Data.shape[0]}.'
            raise ValueError( msg )

        if self.libSizes[0] < self.E + 2 :
            msg = f'{self.name} Validate(): ' +\
                  f'Minimum libSize {self.libSizes[0]}'    +\
                  f' invalid for E={self.E}. Minimum {self.E + 2}.'
            raise ValueError( msg )

        # Get target vectors
        # Univariate prediction targets: first entry of each list
        self.tgt_vec = np.asarray(self.Data[self.target[0]], dtype=np.float64)
        self.col_pred_vec = np.asarray(
            self.Data[self.columns[0]], dtype=np.float64
        )

        self.N = len(self.tgt_vec)

        # Per-variable source vectors for embedding
        self._col_vectors = {
            name: np.asarray(self.Data[name], dtype=np.float64)
            for name in self.columns
        }
        self._tgt_vectors = {
            name: np.asarray(self.Data[name], dtype=np.float64)
            for name in self.target
        }

        # Shifts derived from E and tau
        self.shifts = np.arange(self.E) * self.tau # Embedding shifts

        self._validated = True


    # ================================================================
    # Stage 2: Embed and filter
    # ================================================================

    def Embed(self):
        """
        Build delay embeddings and apply all validity filters.

        For each direction, the embedding is the horizontal stack of
        per-variable E-dimensional time-delay embeddings.  Column
        labels follow the pyEDM convention: ``'Var(t-0)'``,
        ``'Var(t-1)'``, etc. for tau=-1 shifts.

        When ``embedded=True``, the columns listed in ``columns``
        are taken directly as the pre-built embedding (no delay
        construction), and their DataFrame column names serve as labels.
        """
        if not self._validated:
            raise RuntimeError("Call validate() before embed().")

        N = self.N

        if self.embedded:
            # Pre-built embedding: take columns directly
            self.embedding_col = np.column_stack([
                np.asarray(self._dataFrame[c], dtype=np.float64)
                for c in self.columns
            ])
            self.valid_col = ~np.any(np.isnan(self.embedding_col), axis=1)
            self.embed_col_labels = self.columns

            # Reverse: target list as pre-built embedding
            self.embedding_tgt = np.column_stack([
                np.asarray(self._dataFrame[c], dtype=np.float64)
                for c in self.target
            ])
            self.valid_tgt = ~np.any(np.isnan(self.embedding_tgt), axis=1)
            self.embed_tgt_labels = self.target
        else:
            # Mixed-multivariate delay embedding
            self.embedding_col, self.valid_col, self.embed_col_labels = (
                _build_multivariate_embedding(
                    self._col_vectors, self.columns, self.shifts
                )
            )
            self.embedding_tgt, self.valid_tgt, self.embed_tgt_labels = (
                _build_multivariate_embedding(
                    self._tgt_vectors, self.target, self.shifts
                )
            )

        # validLib mask
        if len( self.validLib ):
            valid_lib_mask = np.asarray(self.validLib, dtype=bool)
            self.valid_col = self.valid_col & valid_lib_mask
            self.valid_tgt = self.valid_tgt & valid_lib_mask

        # Tp and NaN filters on the prediction target scalars
        self.valid_col = (self.valid_col
                          & _tp_valid_mask(N, self.tgt_vec, self.Tp)
                          & ~np.isnan(self.col_pred_vec))
        self.valid_tgt = (self.valid_tgt
                          & _tp_valid_mask(N, self.col_pred_vec, self.Tp)
                          & ~np.isnan(self.tgt_vec))

        self.idx_col = np.where(self.valid_col)[0]
        self.idx_tgt = np.where(self.valid_tgt)[0]

        # Resolve knn per direction
        D_col = self.embedding_col.shape[1]
        D_tgt = self.embedding_tgt.shape[1]

        if self.knn > 0:
            self.knn_fwd = self.knn
            self.knn_rev = self.knn
        else:
            self.knn_fwd = D_col + 1
            self.knn_rev = D_tgt + 1

        min_valid_col = self.knn_fwd + 1
        min_valid_tgt = self.knn_rev + 1

        if len(self.idx_col) < min_valid_col:
            raise ValueError(
                f"Insufficient valid points for forward direction: "
                f"{len(self.idx_col)} valid, need at least {min_valid_col}."
            )
        if len(self.idx_tgt) < min_valid_tgt:
            raise ValueError(
                f"Insufficient valid points for reverse direction: "
                f"{len(self.idx_tgt)} valid, need at least {min_valid_tgt}."
            )
        
        # Note: target vectors offset by Tp : Legacy Simplex() offsets knn
        self.pred_vals_col = self.tgt_vec[self.idx_col + self.Tp]
        self.pred_vals_tgt = self.col_pred_vec[self.idx_tgt + self.Tp]

        if np.any(np.isnan(self.pred_vals_col)):
            raise RuntimeError("NaN in forward pred_vals after filtering.")
        if np.any(np.isnan(self.pred_vals_tgt)):
            raise RuntimeError("NaN in reverse pred_vals after filtering.")

        # Contiguous valid embedding subsets for fast subsample indexing
        self.embed_valid_col = np.ascontiguousarray(
            self.embedding_col[self.idx_col]
        )
        self.embed_valid_tgt = np.ascontiguousarray(
            self.embedding_tgt[self.idx_tgt]
        )

        self._embedded = True

    # ================================================================
    # Project
    # ================================================================

    _UNSET = object() # If called outside API can override args

    def Project(self,
                libSizes=_UNSET,
                sample=_UNSET,
                seed=_UNSET,
                includeData=_UNSET,
                parallel=_UNSET,
                mpMethod=_UNSET,
                sharedMB=_UNSET,
                verbose=_UNSET,):
        """
        Run Convergent Cross Mapping over the specified library sizes.

        At each library size L and each sample iteration:
         - L points are randomly selected as the library.
         - ALL M valid points are the prediction set.
         - A KDTree is built from the L library embedding vectors.
         - All M points query k nearest neighbors from the L-point tree.
         - Simplex predictions and Pearson r are computed over all M.

        Returns
        -------
        self.libMeans : pandas.DataFrame
        """
        U = CCM._UNSET
        libSizes    = self.libSizes    if libSizes    is U else libSizes
        sample      = self.sample      if sample      is U else sample
        seed        = self.seed        if seed        is U else seed
        includeData = self.includeData if includeData is U else includeData
        parallel    = self.parallel    if parallel    is U else parallel
        mpMethod    = self.mpMethod    if mpMethod    is U else mpMethod
        sharedMB    = self.sharedMB    if sharedMB    is U else sharedMB
        verbose     = self.verbose     if  verbose    is U else verbose

        if not self._validated:
            self.Validate()
        if not self._embedded:
            self.Embed()

        col_label = self.columns
        target    = self.target
        knn_fwd   = self.knn_fwd
        knn_rev   = self.knn_rev
        excl      = self.exclusionRadius
        root_seq  = np.random.SeedSequence(seed)

        n_lib      = len(libSizes)
        n_tasks    = 2 * n_lib
        child_seqs = root_seq.spawn(n_tasks)

        tasks = []
        for i, L in enumerate(libSizes):
            tasks.append(('fwd', int(L), sample, knn_fwd, excl,
                          child_seqs[2 * i].entropy, includeData))
            tasks.append(('rev', int(L), sample, knn_rev, excl,
                          child_seqs[2 * i + 1].entropy, includeData))

        # ---- Dispatch ----
        n_workers = _resolve_workers(parallel)

        if n_workers > 1:
            chunksize = max(1, n_tasks // (4 * n_workers))
            ctx = _get_mp_context(mpMethod)

            all_arrays = [
                self.embed_valid_col, self.pred_vals_col, self.idx_col,
                self.embed_valid_tgt, self.pred_vals_tgt, self.idx_tgt,
            ]

            # Decide whether to use Pool initializer pickle or
            # shared memory to share all_arrays to workers.
            total_bytes = sum(np.ascontiguousarray(a).nbytes
                              for a in all_arrays)
            use_shm = total_bytes > sharedMB * 1_000_000

            if verbose:
                msg = f"Project: total_bytes {total_bytes:,d} use_shm {use_shm}"
                print( msg, flush=True)

            if use_shm:
                # Use shared memory
                shm_handles = []
                shm_specs = []
                for arr in all_arrays:
                    shm, spec = _create_shared_array(
                        np.ascontiguousarray(arr)
                    )
                    shm_handles.append(shm)
                    shm_specs.append(spec)

                try:
                    with ctx.Pool(
                        processes=n_workers,
                        initializer=_pool_initializer_shm,
                        initargs=(shm_specs,),
                    ) as pool:
                        results = pool.starmap(
                            _pool_task, tasks, chunksize=chunksize
                        )
                finally:
                    for shm in shm_handles:
                        shm.close()
                        shm.unlink()
            else:
                # Use Pool initializer
                contiguous = [np.ascontiguousarray(a) for a in all_arrays]
                with ctx.Pool(
                    processes=n_workers,
                    initializer=_pool_initializer_pickle,
                    initargs=tuple(contiguous),
                ) as pool:
                    results = pool.starmap(
                        _pool_task, tasks, chunksize=chunksize
                    )
        else:
            # Sequential, not parallel
            _worker_data['fwd'] = (self.embed_valid_col, self.pred_vals_col,
                                   self.idx_col)
            _worker_data['rev'] = (self.embed_valid_tgt, self.pred_vals_tgt,
                                   self.idx_tgt)
            results = [_pool_task(*t) for t in tasks]

        # ---- Unpack ----
        if includeData: # CCM mean, var of sample rho
            rho_col_tgt = [results[2 * i][0]     for i in range(n_lib)]
            rho_tgt_col = [results[2 * i + 1][0] for i in range(n_lib)]
            var_col_tgt = [results[2 * i][1]     for i in range(n_lib)]
            var_tgt_col = [results[2 * i + 1][1] for i in range(n_lib)]

            self.libMeans = DataFrame({
                'LibSize': libSizes,
                f'{self.columns[0]}:{self.target[0]}': rho_col_tgt,
                f'{self.target[0]}:{self.columns[0]}': rho_tgt_col,
                f'{self.columns[0]}:{self.target[0]}_var': var_col_tgt,
                f'{self.target[0]}:{self.columns[0]}_var': var_tgt_col,
            })
        else: # CCM mean of sample rho
            rho_col_tgt = [results[2 * i]     for i in range(n_lib)]
            rho_tgt_col = [results[2 * i + 1] for i in range(n_lib)]

            self.libMeans = DataFrame({
                'LibSize': libSizes,
                f'{self.columns[0]}:{self.target[0]}': rho_col_tgt,
                f'{self.target[0]}:{self.columns[0]}': rho_tgt_col,
            })


# ====================================================================
# Core CCM loop (module-level for pickling by multiprocessing)
# ====================================================================

def _ccm_for_libsize(embed, pred_vals, time_indices,
                     L, S, k, exclusionRadius, rng):
    """
    Run S subsample CCM trials for library size L.

    At each iteration:
     - L points are randomly selected as the LIBRARY.
     - ALL M points are the PREDICTION SET.
     - A KDTree is built from the L library embedding vectors.
     - All M points query k nearest neighbors from the L-point tree.
     - Self-match and temporal exclusion are filtered from results.
     - Simplex weights, predictions, and Pearson r over all M.

    Parameters
    ----------
    embed : ndarray (M, E) — contiguous valid embedding
    pred_vals : ndarray (M,) — Tp-shifted target values
    time_indices : ndarray (M,) — original data row indices
    L : int — library size
    S : int — number of subsamples
    k : int — number of nearest neighbors (E+1)
    exclusionRadius : int — temporal exclusion radius
    rng : numpy Generator
    """
    M = embed.shape[0]
    L = min(L, M)
    k = min(k, L - 1)
    if k < 1:
        return np.full(S, np.nan)

    # Query headroom: +1 for self-match, +2*radius for exclusion
    k_query = k + 1
    if exclusionRadius > 0:
        k_query += 2 * exclusionRadius
    k_query = min(k_query, L)

    rhos = np.empty(S, dtype=np.float64)
    r_M = np.arange(M)[:, np.newaxis]  # (M, 1) row indexer

    for s in range(S):
        lib_idx = rng.choice(M, size=L, replace=False)
        lib_idx.sort()

        # ---- Build KDTree from L library points, query all M ----
        lib_embed = embed[lib_idx] # (L, E)
        tree = KDTree(lib_embed)
        nn_dist_raw, nn_local_raw = tree.query(embed, k=k_query)

        # Ensure 2D for k_query == 1
        if nn_dist_raw.ndim == 1:
            nn_dist_raw  = nn_dist_raw[:, np.newaxis]
            nn_local_raw = nn_local_raw[:, np.newaxis]

        # nn_local_raw indexes into lib_embed; map to global M-indices
        nn_global_raw = lib_idx[nn_local_raw] # (M, k_query)

        # ---- Exclude self-match and temporal neighbors ----
        is_self = (nn_global_raw == r_M)

        if exclusionRadius > 0:
            query_times = time_indices[:, np.newaxis]  # (M, 1)
            nn_times = time_indices[nn_global_raw]     # (M, k_query)
            is_excl = np.abs(query_times - nn_times) <= exclusionRadius
            mask = is_self | is_excl
        else:
            mask = is_self

        # ---- Select first k valid neighbors per row ----
        valid = ~mask
        cs = np.cumsum(valid, axis=1)
        first_k = valid & (cs <= k)

        total_found = cs[:, -1]
        insufficient = total_found < k

        if np.all(insufficient):
            rhos[s] = np.nan
            continue

        if np.any(insufficient):
            first_k[insufficient, :] = False

        # Gather column positions of the k selected neighbors
        _, col_indices = np.where(first_k)

        nn_cols = np.zeros((M, k), dtype=np.intp)
        sufficient = ~insufficient
        suf_count = sufficient.sum()
        if suf_count > 0:
            nn_cols[sufficient] = col_indices.reshape(suf_count, k)

        nn_dist  = nn_dist_raw[r_M, nn_cols]      # (M, k)
        nn_global = nn_global_raw[r_M, nn_cols]   # (M, k)

        # ---- Simplex weights: w_j = exp(-d_j / d_min), normalised ----
        d_min    = nn_dist[:, 0:1]
        d_min_nz = np.where(d_min > 0.0, d_min, 1.0)
        weights  = np.exp(-nn_dist / d_min_nz)

        zero_mask = (d_min == 0.0)
        if np.any(zero_mask):
            weights = np.where(
                zero_mask,
                np.where(nn_dist == 0.0, 1.0, 0.0),
                weights
            )
        w_sum = weights.sum(axis=1, keepdims=True)
        w_sum = np.where(w_sum > 0.0, w_sum, 1.0)
        weights /= w_sum

        # ---- Cross-map predictions for all M points ----
        # Neighbor target values: pred_vals at the library neighbors
        predictions = np.sum(weights * pred_vals[nn_global], axis=1)
        actuals     = pred_vals

        if np.any(insufficient):
            predictions[insufficient] = np.nan

        # ---- NaN-safe Pearson r ----
        rhos[s] = _nan_safe_pearson(predictions, actuals)

    return rhos


# ====================================================================
# Worker process globals
# ====================================================================
_worker_data = {}


def _pool_initializer_pickle(fwd_embed, fwd_pred, fwd_tidx,
                             rev_embed, rev_pred, rev_tidx):
    """Store worker data passed via initargs (pickle path)."""
    _worker_data['fwd'] = (fwd_embed, fwd_pred, fwd_tidx)
    _worker_data['rev'] = (rev_embed, rev_pred, rev_tidx)


def _pool_initializer_shm(shm_specs):
    """
    Attach to shared memory segments in worker process.

    shm_specs: list of 6 entries:
      [fwd_embed, fwd_pred, fwd_time_idx,
       rev_embed, rev_pred, rev_time_idx]
    """
    arrays = []
    shm_handles = []
    for name, shape, dtype_str in shm_specs:
        shm = shared_memory.SharedMemory(name=name, create=False)
        shm_handles.append(shm)
        arrays.append(
            np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        )
    _worker_data['fwd'] = (arrays[0], arrays[1], arrays[2])
    _worker_data['rev'] = (arrays[3], arrays[4], arrays[5])
    _worker_data['_shm'] = shm_handles


def _pool_task(direction, L, S, k, exclusionRadius, seed_entropy,
               includeData):
    """Execute a single CCM task (one direction, one library size).

    Call: _ccm_for_libsize() as the worker to run S subsample CCM
          trials for library size L.

    Parameters
    ----------
    direction       : str — 'fwd' or 'rev'
    L               : int — library size
    S               : int — number of subsamples
    k               : int — number of nearest neighbors
    exclusionRadius : int — temporal exclusion radius
    seed_entropy    : SeedSequence(seed)
    includeData     : True: return tuple of ( mean(rhos), stdev(rhos) )
    """
    embed, pred_vals, time_indices = _worker_data[direction]
    rng  = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    rhos = _ccm_for_libsize(embed, pred_vals, time_indices,
                            L, S, k, exclusionRadius, rng)

    _mean = np.nanmean(rhos)
    if includeData:
        return (_mean, np.nanvar(rhos, mean=_mean))
    else:
        return _mean


# ====================================================================
# Shared memory helpers
# ====================================================================

def _create_shared_array(arr):
    """Copy a numpy array into a new shared memory segment."""
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[:] = arr
    return shm, (shm.name, arr.shape, arr.dtype.str)


# ====================================================================
# Helpers
# ====================================================================

def _nan_safe_pearson(predictions, actuals):
    """Pearson r excluding NaN pairs. NaN if < 3 valid pairs."""
    valid = ~(np.isnan(predictions) | np.isnan(actuals))
    n = valid.sum()
    if n < 3:
        return np.nan
    p = predictions[valid]
    a = actuals[valid]
    pm = p - p.mean()
    am = a - a.mean()
    denom = np.sqrt(np.dot(pm, pm) * np.dot(am, am))
    if denom == 0.0:
        return 0.0
    return np.dot(pm, am) / denom


def _resolve_workers(parallel):
    if parallel is True:
        return max(1, cpu_count())
    if parallel is False:
        return 1
    n_workers = min(cpu_count(), int(parallel))
    return max(1, n_workers)


def _build_embedding(vec, shifts):
    """
    Delay-coordinate embedding for a single variable.

    NaN propagates: any row referencing a NaN source value or an
    out-of-bounds shift is marked invalid. Invalid rows retain NaN
    so they are never mistaken for real data.

    Returns
    -------
    emb    : ndarray (N, E)  — invalid rows contain NaN
    valid  : ndarray (N,)    — boolean mask of usable rows
    labels : list of str     — column labels in pyEDM convention
    """
    N = len(vec)
    E = len(shifts)
    emb = np.full((N, E), np.nan, dtype=np.float64)
    for dim, s in enumerate(shifts):
        if s <= 0:
            emb[-s:, dim] = vec[:N + s]
        else:
            emb[:N - s, dim] = vec[s:]
    valid = ~np.any(np.isnan(emb), axis=1)
    return emb, valid


def _make_shift_labels(var_name, shifts):
    """
    Generate pyEDM-convention column labels for a delay embedding.

    For tau=-1, E=3  →  shifts=[0, -1, -2]
        labels = ['V(t-0)', 'V(t-1)', 'V(t-2)']

    For tau=+1, E=3  →  shifts=[0, 1, 2]
        labels = ['V(t+0)', 'V(t+1)', 'V(t+2)']

    The displayed offset is -shift for negative tau (backward in time)
    and +shift for positive tau (forward in time).
    """
    labels = []
    for s in shifts:
        if s <= 0:
            labels.append(f'{var_name}(t-{-s})')
        else:
            labels.append(f'{var_name}(t+{s})')
    return labels


def _build_multivariate_embedding(vectors, var_names, shifts):
    """
    Build a mixed-multivariate delay embedding by stacking per-variable
    E-dimensional time-delay embeddings horizontally.

    Parameters
    ----------
    vectors : dict mapping var_name → ndarray (N,)
    var_names : list of str — ordered variable names
    shifts : ndarray — embedding shifts (from E and tau)

    Returns
    -------
    emb    : ndarray (N, E * n_vars)
    valid  : ndarray (N,) — AND of all per-variable validity masks
    labels : list of str  — concatenated per-variable column labels
    """
    embeddings = []
    all_labels = []
    valid = None

    for name in var_names:
        emb_v, valid_v = _build_embedding(vectors[name], shifts)
        embeddings.append(emb_v)
        all_labels.extend(_make_shift_labels(name, shifts))
        if valid is None:
            valid = valid_v.copy()
        else:
            valid &= valid_v

    emb = np.hstack(embeddings)
    return emb, valid, all_labels


def _tp_valid_mask(N, vec, Tp):
    """True where t + Tp is in bounds and vec[t + Tp] is not NaN."""
    shifted = np.arange(N) + Tp
    in_bounds = (shifted >= 0) & (shifted < N)
    clipped = np.clip(shifted, 0, N - 1)
    return in_bounds & ~np.isnan(vec[clipped])


def _get_mp_context(mpMethod=None):
    """Return a multiprocessing context that is safe with NumPy / BLAS.

    If mpContext is not None: get mpContext
    Get default 
    Tries *forkserver* first (faster on Linux), falls back to *spawn*.
    Raises RuntimeError if neither is available.  **fork is never used.**
    """
    allowed = ("forkserver", "spawn")

    if mpMethod is not None:
        if mpMethod not in allowed:
            raise ValueError(
                "pyEDM CCM requires 'forkserver' or 'spawn' mpMethod"
            )
        return get_context(mpMethod)

    # No mpMethod try system default
    current = get_start_method(allow_none=True)
    if current in allowed:
        return get_context(current)

    # Last chance: try forkserver then spawn
    for method in ("forkserver", "spawn"):
        try:
            return get_context(method)
        except ValueError:
            continue

    raise ValueError(
        "No safe multiprocessing start method available. "
        "pyEDM CCM requires 'forkserver' or 'spawn'."
    )
