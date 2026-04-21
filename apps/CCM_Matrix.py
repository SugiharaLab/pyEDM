#! /usr/bin/env python3

"""
CCM_Matrix: compute M×M×L convergent cross mapping tensor.

For M data columns, computes CCM correlation of each column against
every other column at each library size, returning an (M, M, L)
float16 tensor with a convergence depth dimension.

Optimization: for a fixed source column i the KDTree construction,
neighbor query, self/exclusion filtering, and simplex weight computation
depend only on column i's embedding — not on the target. The expensive
per-subsample work is done once and the resulting weights are reused
across all M-1 target columns via vectorized batched prediction.

Post-processing:
 - Linear convergence slope: vectorized numpy regression across the
   full M×M matrix simultaneously, producing an (M, M) float32 array.
 - Exponential convergence (optional): scipy curve_fit of
   rho(L) = y0 + b*(1 - exp(-a*x)) per cell, storing the rate
   parameter `a` as an (M, M) float32 array.

Reference: George Sugihara et al., Detecting Causality in Complex Ecosystems.
           Science338, 496-500(2012). DOI:10.1126/science.1227079
"""
# Python distribution modules
from datetime        import datetime
from time            import perf_counter
from argparse        import ArgumentParser
from multiprocessing import cpu_count, shared_memory
from multiprocessing import get_context, get_start_method
from pickle          import dump
import sys
import warnings

# Community modules
from pyEDM          import CCM
from scipy.spatial  import KDTree
from pandas         import DataFrame, read_csv, read_feather
from scipy.optimize import curve_fit
from matplotlib     import pyplot as plt
import numpy as np

# ====================================================================
# CCM_Matrix class
# ====================================================================

class CCM_Matrix:
    """
    Compute the full M×M×L convergent cross mapping tensor.

    Parameters
    ----------
    dataFrame : pandas.DataFrame
        Input data. First column is time unless noTime=True.
    E : int or array-like of int
        Embedding dimension. Scalar or per-column vector of length M.
    Tp : int
        Prediction horizon. Default 0.
    tau : int
        Embedding delay. Default -1.
    exclusionRadius : int
        Temporal exclusion radius. Default 0.
    libSizes : list of int
        Explicit library sizes. If non-empty, used directly and
        pLibSizes is ignored. Default [].
    pLibSizes : list of float
        Percentiles of N to generate library sizes. Used only when
        libSizes is empty. Default [10, 20, 80, 90].
    sample : int
        Subsamples per library size. Default 100.
    seed : int or None
        RNG seed.
    noTime : bool
        If True, all columns are data. If False, first column is
        time (stripped). Default False.
    parallel : bool or int
        Worker count. Default True.
    mpMethod : str or None
        Multiprocessing start method: 'forkserver' or 'spawn' only.
        Default None.
    sharedMB : float
        Data size threshold for shared memory vs pickle. Default 5.
    targetBatchSize : int or None
        Max target columns per batch within each worker. Default None.
    expConverge : bool
        If True, fit exponential convergence curve. Default False.
    progressLog : None, True, or str
        None: no logging. True: log to stderr. str: log to file path.
        Default None.
    progressInterval : int
        Percentage increment for progress log lines. Default 5.

    Attributes (after Run)
    --------------------------
    tensor : ndarray (M, M, |L|), float16
    slope : ndarray (M, M), float32
    exp_a : ndarray (M, M), float32 or None
    column_names : list of str
    lib_sizes_arr : ndarray of int
    lib_sizes_norm : ndarray of float

    The slope of CCM rho(libSizes) is computed based on a [0,1]
    normalization of libSizes.

    if expConverge = True a nonlinear convergence function is fit
    to rho(libSizes) : y0 + b * ( 1 - exp(-a * x) ) with fit coefficient
    a returned in the (M, M) matrix self.exp_a
    """

    _validated = False

    def __init__(self,
                 dataFrame,
                 E,
                 libSizes         = [],
                 pLibSizes        = [10,20,80,90],
                 Tp               = 0,
                 tau              = -1,
                 exclusionRadius  = 0,
                 sample           = 30,
                 seed             = None,
                 noTime           = False,
                 parallel         = True,
                 mpMethod         = None,
                 sharedMB   = 0.01,
                 targetBatchSize  = None,
                 expConverge      = False,
                 progressLog      = None,
                 progressInterval = 5):

        self.E                = E
        self.Tp               = Tp
        self.tau              = tau
        self.exclusionRadius  = exclusionRadius
        self.libSizes         = libSizes
        self.pLibSizes        = pLibSizes
        self.sample           = sample
        self.seed             = seed
        self.noTime           = noTime
        self.parallel         = parallel
        self.mpMethod         = mpMethod
        self.sharedMB         = sharedMB
        self.targetBatchSize  = targetBatchSize
        self.expConverge      = expConverge
        self.progressLog      = progressLog
        self.progressInterval = progressInterval
        self._dataFrame       = dataFrame

        self.tensor         = None
        self.slope          = None
        self.exp_a          = None
        self.column_names   = None
        self.lib_sizes_arr  = None
        self.lib_sizes_norm = None

    # ================================================================
    # Validate
    # ================================================================

    def Validate(self):
        """Parse DataFrame, extract data, resolve E and library sizes."""
        df = self._dataFrame

        if self.noTime:
            self.column_names = list(df.columns)
            self.data_matrix = np.ascontiguousarray(
                df.values.astype(np.float64)
            )
        else:
            self.column_names = list(df.columns[1:])
            self.data_matrix = np.ascontiguousarray(
                df.iloc[:, 1:].values.astype(np.float64)
            )

        self.N, self.M = self.data_matrix.shape

        E_input = self.E
        if np.ndim(E_input) == 0:
            # scalar E : create M dimensional vector
            self.E_vec = np.full(self.M, int(E_input), dtype=int)
        else:
            self.E_vec = np.asarray(E_input, dtype=int)
            if len(self.E_vec) != self.M:
                raise ValueError(
                    f"E has length {len(self.E_vec)} but there are "
                    f"{self.M} data columns."
                )

        self.k_vec = self.E_vec + 1

        if len(self.libSizes) > 0:
            self.lib_sizes_arr = np.asarray(
                self.libSizes, dtype=int
            )
        else:
            pcts = np.asarray(self.pLibSizes, dtype=float)
            self.lib_sizes_arr = np.unique(np.clip(
                np.round(pcts / 100.0 * self.N).astype(int),
                2, self.N
            ))
            self.libSizes = self.lib_sizes_arr

        self.lib_sizes_norm = (self.lib_sizes_arr.astype(np.float64)
                               / self.N)

        if self.M < 2:
            raise ValueError(
                f"Need at least 2 data columns, got {self.M}."
            )
        if np.any(self.E_vec < 1):
            raise ValueError("All E values must be >= 1.")

        self._validated = True

    # ================================================================
    # Run
    # ================================================================

    def Run(self):
        """
        Compute the M×M×L CCM tensor, linear convergence slope,
        and (optionally) exponential convergence rate.

        Returns
        -------
        tensor : ndarray (M, M, |L|), float16
        """
        if not self._validated:
            self.Validate()

        M = self.M
        N = self.N
        n_lib     = len(self.lib_sizes_arr)
        logging   = self.progressLog is not None
        dest      = self.progressLog # None, True or filename
        interval  = self.progressInterval
        n_workers = _resolve_workers(self.parallel)

        if logging:
            datetime_start = datetime.now()
            _log_progress(dest, "CCM_Matrix.Run() starting.")

        root_seq = np.random.SeedSequence(self.seed)
        child_seqs = root_seq.spawn(M)

        tasks = []
        for i in range(M):
            tasks.append((
                i, M, N, int(self.E_vec[i]), self.tau,
                self.lib_sizes_arr.tolist(),
                self.sample, self.exclusionRadius, self.Tp,
                self.targetBatchSize, child_seqs[i].entropy
            ))

        # ---- Dispatch ----
        if n_workers > 1:
            ctx = _resolve_mp_context(self.mpMethod)
            total_bytes = self.data_matrix.nbytes
            use_shm = total_bytes > self.sharedMB * 1_000_000

            if logging:
                results = self._dispatch_parallel_logged(
                    ctx, tasks, use_shm, M, n_workers, dest, interval
                )
            else:
                results = self._dispatch_parallel_silent(
                    ctx, tasks, use_shm, n_workers
                )
        else:
            _mw_data['data'] = self.data_matrix
            if logging:
                results = self._dispatch_sequential_logged(
                    tasks, M, dest, interval
                )
            else:
                results = [_mw_task(*t) for t in tasks]

        # ---- Assemble tensor ----
        tensor_f32 = np.full((M, M, n_lib), np.nan, dtype=np.float32)
        for src_idx, row in results:
            tensor_f32[src_idx, :, :] = row

        self.tensor = tensor_f32.astype(np.float16)

        if logging:
            msg = (f"100% ({M}/{M}) complete")
            _log_progress(dest, msg)

        # ---- Post-processing: convergence estimates ----
        t_post = perf_counter()

        if self.tensor.shape[2] > 1:
            self.slope = _compute_slope(tensor_f32, self.lib_sizes_norm)

        t_slope = perf_counter() - t_post
        datetime_end = datetime.now()

        if self.expConverge:
            t_exp0 = perf_counter()
            self.exp_a = _compute_exp_converge(
                tensor_f32, self.lib_sizes_norm
            )
            t_exp = perf_counter() - t_exp0
            datetime_end = datetime.now()
        else:
            self.exp_a = None
            t_exp = 0.0

        if logging:
            msg = (f"slope complete: {_fmt_duration(t_slope)}")
            _log_progress(dest, msg)

            if self.expConverge:
                msg = f"expConverge: {_fmt_duration(t_exp)}"
                _log_progress(dest, msg)

            msg = f"Elapsed: {datetime_end - datetime_start}"
            _log_progress(dest, msg)

        return dict( tensor   = self.tensor,
                     columns  = self.column_names,
                     slope    = self.slope,
                     exp_a    = self.exp_a,
                     libSizes = self.libSizes )

    # ---- Parallel dispatch with logging ----

    def _dispatch_parallel_logged(self, ctx, tasks, use_shm,
                                  M, n_workers, dest, interval):
        chunksize = max(1, M // (4 * n_workers))
        results = []

        if use_shm:
            shm, spec = _create_shared_array(self.data_matrix)
            init_fn = _mw_init_shm
            init_args = ([spec],)
        else:
            shm = None
            init_fn = _mw_init_pickle
            init_args = (self.data_matrix,)

        try:
            with ctx.Pool(
                processes=n_workers,
                initializer=init_fn,
                initargs=init_args,
            ) as pool:
                completed = 0
                next_threshold = interval
                t_start = perf_counter()

                for result in pool.imap_unordered(
                    _mw_task_unpack, tasks, chunksize=chunksize
                ):
                    results.append(result)
                    completed += 1
                    pct = completed * 100.0 / M

                    if pct >= next_threshold:
                        elapsed = perf_counter() - t_start
                        rate = completed / elapsed
                        remaining = (M - completed) / rate
                        _log_progress(
                            dest,
                            f"{int(pct)}% ({completed}/{M}) | "
                            f"elapsed {_fmt_duration(elapsed)} | "
                            f"~{_fmt_duration(remaining)} remaining | "
                            f"{rate:.1f} tasks/s"
                        )
                        next_threshold += interval
        finally:
            if shm is not None:
                shm.close()
                shm.unlink()

        return results

    # ---- Parallel dispatch without logging ----

    def _dispatch_parallel_silent(self, ctx, tasks, use_shm, n_workers):
        chunksize = max(1, len(tasks) // (4 * n_workers))

        if use_shm:
            shm, spec = _create_shared_array(self.data_matrix)
            try:
                with ctx.Pool(
                    processes=n_workers,
                    initializer=_mw_init_shm,
                    initargs=([spec],),
                ) as pool:
                    results = pool.starmap(
                        _mw_task, tasks, chunksize=chunksize
                    )
            finally:
                shm.close()
                shm.unlink()
        else:
            with ctx.Pool(
                processes=n_workers,
                initializer=_mw_init_pickle,
                initargs=(self.data_matrix,),
            ) as pool:
                results = pool.starmap(
                    _mw_task, tasks, chunksize=chunksize
                )

        return results

    # ---- Sequential dispatch with logging ----

    def _dispatch_sequential_logged(self, tasks, M, dest, interval):
        results = []
        completed = 0
        next_threshold = interval
        t_start = perf_counter()

        for t in tasks:
            results.append(_mw_task(*t))
            completed += 1
            pct = completed * 100.0 / M

            if pct >= next_threshold:
                elapsed = perf_counter() - t_start
                rate = completed / elapsed
                remaining = (M - completed) / rate
                _log_progress(
                    dest,
                    f"{int(pct)}% ({completed}/{M}) | "
                    f"elapsed {_fmt_duration(elapsed)} | "
                    f"~{_fmt_duration(remaining)} remaining | "
                    f"{rate:.1f} tasks/s"
                )
                next_threshold += interval

        return results

    def __repr__(self):
        state = 'ready' if self._validated else 'init'
        if self.tensor is not None:
            state = f'computed({self.tensor.shape})'

        if self._validated:
            e_min, e_max = self.E_vec.min(), self.E_vec.max()
            e_str = str(e_min) if e_min == e_max else f'{e_min}..{e_max}'
        else:
            e_str = str(self.E)

        parts = [
            f"CCM_Matrix(M={self.M if self._validated else '?'}",
            f"E={e_str}, Tp={self.Tp}, tau={self.tau}",
            f"state='{state}'",
        ]
        return ', '.join(parts) + ')'


# ====================================================================
# Batched Pearson correlation across columns
# ====================================================================

def _batched_pearson_cols(preds, actuals, skip_col, col_offset):
    """Pearson r per column between preds and actuals."""
    B = preds.shape[1]
    has_nan = np.isnan(preds).any() or np.isnan(actuals).any()

    if not has_nan:
        pm = preds - preds.mean(axis=0, keepdims=True)
        am = actuals - actuals.mean(axis=0, keepdims=True)
        num = np.sum(pm * am, axis=0)
        den = np.sqrt(np.sum(pm * pm, axis=0) * np.sum(am * am, axis=0))
        rhos = np.where(den > 0, num / den, 0.0).astype(np.float32)
    else:
        rhos = np.full(B, np.nan, dtype=np.float32)
        for b in range(B):
            j = col_offset + b
            if j == skip_col:
                continue
            rhos[b] = _nan_safe_pearson(preds[:, b], actuals[:, b])

    diag_b = skip_col - col_offset
    if 0 <= diag_b < B:
        rhos[diag_b] = np.nan

    return rhos


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


# ====================================================================
# Helpers
# ====================================================================

def _resolve_mp_context(mpMethod):
    """Return a multiprocessing context restricted to forkserver/spawn."""
    allowed = ('forkserver', 'spawn')
    if mpMethod is not None:
        if mpMethod not in allowed:
            raise ValueError(
                f"mpMethod must be one of {allowed}, got '{mpMethod}'."
            )
        return get_context(mpMethod)
    default = get_start_method(allow_none=True)
    if default in allowed:
        return get_context(default)
    return get_context('forkserver')


def _resolve_workers(parallel):
    """Resolve the parallel parameter to a worker count."""
    if parallel is True:
        return max(1, cpu_count())
    if parallel is False:
        return 1
    n_workers = min(cpu_count(), int(parallel))
    return max(1, n_workers)


def _create_shared_array(arr):
    """Copy a numpy array into a new shared memory segment."""
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    buf = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    buf[:] = arr
    return shm, (shm.name, arr.shape, arr.dtype.str)


def _build_embedding(vec, shifts):
    """Delay-coordinate embedding. Invalid rows retain NaN."""
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


# ====================================================================
# Worker process globals
# ====================================================================
_mw_data = {}


def _mw_init_pickle(data_matrix):
    """Store data matrix passed via initargs (pickle path)."""
    _mw_data['data'] = data_matrix


def _mw_init_shm(shm_specs):
    """Attach to shared memory data matrix in worker process."""
    arrays = []
    handles = []
    for name, shape, dtype_str in shm_specs:
        shm = shared_memory.SharedMemory(name=name, create=False)
        handles.append(shm)
        arrays.append(
            np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        )
    _mw_data['data'] = arrays[0]
    _mw_data['_shm'] = handles


def _mw_task(src_idx, M, N, E_src, tau, lib_sizes, sample,
             exclusionRadius, Tp, target_batch_size, seed_entropy):
    """
    Process one source column → one row of the M×M×L tensor.
    """
    data = _mw_data['data']
    n_lib = len(lib_sizes)
    row = np.full((M, n_lib), np.nan, dtype=np.float32)

    k = E_src + 1
    shifts = np.arange(E_src) * tau

    src_vec = data[:, src_idx]
    embed_src, valid_src = _build_embedding(src_vec, shifts)
    valid_src &= ~np.isnan(src_vec)

    idx_src = np.where(valid_src)[0]
    M_valid = len(idx_src)

    if M_valid < E_src + 2:
        return src_idx, row

    embed_valid = np.ascontiguousarray(embed_src[idx_src])

    shifted_idx = idx_src + Tp
    in_bounds = (shifted_idx >= 0) & (shifted_idx < N)
    shifted_safe = np.clip(shifted_idx, 0, N - 1)

    target_all = data[shifted_safe, :].copy()
    target_all[~in_bounds, :] = np.nan

    rng = np.random.default_rng(np.random.SeedSequence(seed_entropy))
    r_Mv = np.arange(M_valid)[:, np.newaxis]

    if target_batch_size is None or target_batch_size <= 0:
        target_batch_size = M

    k_base = min(k, M_valid - 1)
    if k_base < 1:
        return src_idx, row

    for li, L in enumerate(lib_sizes):
        L = min(int(L), M_valid)
        k_use = min(k, L - 1)
        if k_use < 1:
            continue

        k_query = min(k_use + 1 + (2 * exclusionRadius
                                    if exclusionRadius > 0 else 0), L)

        sample_rhos = np.full((sample, M), np.nan, dtype=np.float32)

        for s in range(sample):
            lib_idx = rng.choice(M_valid, size=L, replace=False)
            lib_idx.sort()

            tree = KDTree(embed_valid[lib_idx])
            nn_dist_raw, nn_local_raw = tree.query(embed_valid, k=k_query)

            if nn_dist_raw.ndim == 1:
                nn_dist_raw  = nn_dist_raw[:, np.newaxis]
                nn_local_raw = nn_local_raw[:, np.newaxis]

            nn_global_raw = lib_idx[nn_local_raw]

            is_self = (nn_global_raw == r_Mv)

            if exclusionRadius > 0:
                src_rows = idx_src[:, np.newaxis]
                nn_rows  = idx_src[nn_global_raw]
                mask = is_self | (np.abs(src_rows - nn_rows)
                                  <= exclusionRadius)
            else:
                mask = is_self

            valid_nn = ~mask
            cs = np.cumsum(valid_nn, axis=1)
            first_k = valid_nn & (cs <= k_use)

            total_found = cs[:, -1]
            insufficient = total_found < k_use

            if np.all(insufficient):
                continue

            if np.any(insufficient):
                first_k[insufficient, :] = False

            _, col_indices = np.where(first_k)

            nn_cols = np.zeros((M_valid, k_use), dtype=np.intp)
            sufficient = ~insufficient
            suf_count = sufficient.sum()
            if suf_count > 0:
                nn_cols[sufficient] = col_indices.reshape(suf_count, k_use)

            nn_dist   = nn_dist_raw[r_Mv, nn_cols]
            nn_global = nn_global_raw[r_Mv, nn_cols]

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

            if np.any(insufficient):
                weights[insufficient, :] = 0.0

            weights_3d = weights[:, :, np.newaxis]

            for t_start in range(0, M, target_batch_size):
                t_end = min(t_start + target_batch_size, M)
                tgt_batch = target_all[:, t_start:t_end]

                nn_tgt = tgt_batch[nn_global]
                preds = np.sum(weights_3d * nn_tgt, axis=1)

                if np.any(insufficient):
                    preds[insufficient, :] = np.nan

                batch_rhos = _batched_pearson_cols(
                    preds, tgt_batch, src_idx, t_start
                )
                sample_rhos[s, t_start:t_end] = batch_rhos

        any_valid = ~np.all(np.isnan(sample_rhos), axis=0)
        if np.any(any_valid):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                row[any_valid, li] = np.nanmean(
                    sample_rhos[:, any_valid], axis=0
                )

    row[src_idx, :] = np.nan
    return src_idx, row


def _mw_task_unpack(args):
    """Unpack single tuple argument for imap_unordered."""
    return _mw_task(*args)


# ====================================================================
# Convergence fitting functions
# ====================================================================

def _compute_slope(tensor, lib_sizes_norm):
    """
    Vectorized linear regression of CCM rho vs normalised library size
    across the full M×M matrix simultaneously.
    """
    M = tensor.shape[0]
    n_L = tensor.shape[2]

    Y = tensor.astype(np.float32).reshape(M * M, n_L)
    X = lib_sizes_norm.astype(np.float32)

    all_nan = np.all(np.isnan(Y), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        row_mean = np.nanmean(Y, axis=1, keepdims=True)
    row_mean = np.where(np.isnan(row_mean), 0.0, row_mean)
    Y = np.where(np.isnan(Y), row_mean, Y)

    x_bar = X.mean()
    x_dev = X - x_bar
    y_bar = Y.mean(axis=1, keepdims=True)
    y_dev = Y - y_bar

    ss_xy = y_dev @ x_dev
    ss_xx = np.dot(x_dev, x_dev)

    slope = np.where(ss_xx > 0, ss_xy / ss_xx, 0.0)
    slope[all_nan] = np.nan

    return slope.reshape(M, M).astype(np.float32)


def _CCM_rho_L_fit(x, a, b, y0):
    """CCM rho(L) curve for L normalised to [0,1]."""
    return (y0 + b * (1.0 - np.exp(-a * x))).flatten()


def _compute_exp_converge(tensor, lib_sizes_norm):
    """
    Fit rho(L) = y0 + b*(1 - exp(-a*x)) per cell.
    Returns only the rate parameter `a`.
    """
    from scipy.optimize import curve_fit

    M = tensor.shape[0]
    a_matrix = np.full((M, M), np.nan, dtype=np.float32)
    xdata = lib_sizes_norm.astype(np.float64)

    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            ydata = tensor[i, j, :].astype(np.float64)
            if np.all(np.isnan(ydata)) or np.sum(np.isfinite(ydata)) < 3:
                continue
            try:
                popt, _ = curve_fit(
                    _CCM_rho_L_fit,
                    xdata  = xdata,
                    ydata  = ydata,
                    p0     = [2.0, 1.0, 0.1],
                    bounds = ([0, 0, 0], [100, 1, 1]),
                    method = 'dogbox',
                )
                a_matrix[i, j] = popt[0]
            except Exception:
                pass

    return a_matrix


# ====================================================================
# Progress logging
# ====================================================================

def _log_progress(dest, message):
    """
    Write a timestamped log line. Self-contained: opens and closes
    the file on each call so nothing is lost on hard termination.

    Parameters
    ----------
    dest : True (stderr), str (file path), or None (no-op)
    message : str
    """
    if dest is None:
        return

    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"

    if dest is True:
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            pass
    elif isinstance(dest, str):
        try:
            with open(dest, 'a') as f:
                f.write(line)
        except Exception:
            pass


def _fmt_duration(seconds):
    """Format seconds into 'Xh Ym Zs' or 'Ym Zs' or 'Zs'."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"

    
# ====================================================================
# PlotMatrix
# ====================================================================

def PlotMatrix( xm, columns, figsize = (5,5), dpi = 150, title = None,
                plot = True, plotFile = None, cmap = None, norm = None,
                aspect = None, vmin = None, vmax = None, colorBarShrink = 1. ):
    '''Generic function to plot numpy matrix'''

    fig = plt.figure( figsize = figsize, dpi = dpi )
    ax  = fig.add_subplot()

    #fig.suptitle( title )
    ax.set( title = f'{title}' )
    ax.xaxis.set_ticks( [x for x in range( len(columns) )] )
    ax.yaxis.set_ticks( [x for x in range( len(columns) )] )
    ax.set_xticklabels(columns, rotation = 90)
    ax.set_yticklabels(columns)

    cax = ax.matshow( xm, cmap = cmap, norm = norm,
                      aspect = aspect, vmin = vmin, vmax = vmax )
    fig.colorbar( cax, shrink = colorBarShrink )

    plt.tight_layout()

    if plotFile :
        fname = f'{plotFile}'
        plt.savefig( fname, dpi = 'figure', format = 'png' )

    if plot :
        plt.show()


# ====================================================================
# CCM_Matrix_CmdLine
# ====================================================================

def CCM_Matrix_CmdLine():
    '''Wrapper for CCM_Matrix with command line parsing'''

    args = ParseCmdLine()

    # Read data
    # If -i input file: load it, else look for inputData in pyEDM sampleData
    if args.inputFile:
        if '.csv' in args.inputFile[-4:] :
            dataFrame = read_csv( args.inputFile )
        elif '.feather' in args.inputFile[-8:] :
            dataFrame = read_feather( args.inputFile )
        else :
            msg = f'Input file {args.inputFile} must be csv or feather'
            raise( RuntimeError( msg ) )
    elif args.inputData:
        from pyEDM import sampleData
        dataFrame = sampleData[ args.inputData ]
    else:
        raise RuntimeError( "Invalid inputFile or inputData" )

    if len(args.E) == 1:
        args.E = args.E[0]

    # Instantiate CCM_Matrix()
    ccm = CCM_Matrix( dataFrame       = dataFrame,
                      libSizes        = args.libSizes,
                      pLibSizes       = args.pLibSizes,
                      E               = args.E,
                      Tp              = args.Tp,
                      tau             = args.tau,
                      exclusionRadius = args.exclusionRadius,
                      sample          = args.sample,
                      seed            = args.seed,
                      noTime          = args.noTime,
                      parallel        = args.parallel,
                      mpMethod        = args.mpMethod,
                      sharedMB        = args.sharedMB,
                      targetBatchSize = args.targetBatchSize,
                      expConverge     = args.expConverge,
                      progressLog     = args.progressLog,
                      progressInterval= args.progressInterval )

    D = ccm.Run()

    if args.outputFile is not None:
        if args.returnObject:
            with open(args.outputFile,'wb') as f:
                dump( ccm, f )
        else:
            np.savez_compressed( args.outputFile,
                                 tensor   = ccm.tensor,
                                 columns  = ccm.column_names,
                                 slope    = ccm.slope,
                                 exp_a    = ccm.exp_a,
                                 libSizes = ccm.libSizes,)

    if args.Plot:
        # Plot last libSize
        PlotMatrix( tensor[:,:,tensor.shape[2]-1],
                    columns = ccm.column_names,
                    title   = args.plotTitle,
                    figsize = args.figureSize,
                    dpi     = args.dpi )


# ====================================================================
# ParseCmdLine
# ====================================================================

def ParseCmdLine():

    parser = ArgumentParser( description = 'CCM Matrix' )

    parser.add_argument('-i', '--inputFile',
                        dest    = 'inputFile', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Input data file .csv or .feather')

    parser.add_argument('-d', '--inputData',
                        dest    = 'inputData', type = str,
                        action  = 'store',
                        default = 'Lorenz5D',
                        help    = 'pyEDM sampleData name')

    parser.add_argument('-of', '--outputFile',
                        dest    = 'outputFile', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'CCM/slope matrix output file name')

    parser.add_argument('-r', '--returnObject',
                        dest    = 'returnObject',
                        action  = 'store_true',
                        default = False,
                        help    = 'Return CCM_Matric class instance')

    parser.add_argument('-E', '--E', nargs = '*',
                        dest    = 'E', type = int,
                        action  = 'store',
                        default = [],
                        help    = 'E')

    parser.add_argument('-x', '--exclusionRadius',
                        dest    = 'exclusionRadius', type = int,
                        action  = 'store',
                        default = 0,
                        help    = 'Exclusion Radius')

    parser.add_argument('-T', '--Tp',
                        dest    = 'Tp', type = int,
                        action  = 'store',
                        default = 0,
                        help    = 'Tp')

    parser.add_argument('-tau', '--tau', nargs = '*',
                        dest    = 'tau', type = int,
                        action  = 'store',
                        default = [-1],
                        help    = 'tau')

    parser.add_argument('-l', '--libSizes', nargs = '*',
                        dest    = 'libSizes', type = int,
                        action  = 'store',
                        default = [],
                        help    = 'library sizes')

    parser.add_argument('-p', '--pLibSizes', nargs = '*',
                        dest    = 'pLibSizes', type = int,
                        action  = 'store',
                        default = [10,20,80,90],
                        help    = 'percentile library sizes')

    parser.add_argument('-s', '--sample',
                        dest    = 'sample', type = int,
                        action  = 'store',
                        default = 30,
                        help    = 'CCM sample')

    parser.add_argument('-seed', '--seed',
                        dest    = 'seed', type = int,
                        action  = 'store',
                        default = None,
                        help    = 'RNG seed')

    parser.add_argument('-nT', '--noTime',
                        dest    = 'noTime',
                        action  = 'store_true',
                        default = False,
                        help    = 'set noTime True')

    parser.add_argument('-parallel', '--parallel',
                        dest   = 'parallel',
                        action = 'store', default = True,
                        help = 'Number of parallel processes')

    parser.add_argument('-mp', '--mpMethod',
                        dest    = 'mpMethod', type = str,
                        action  = 'store',
                        default = None,
                        help    = 'Multiprocessing start method')

    parser.add_argument('-m', '--sharedMB',
                        dest    = 'sharedMB', type = float,
                        action  = 'store',
                        default = 0.01,
                        help    = 'Shared memory data size threshold')
    
    parser.add_argument('-b', '--targetBatchSize',
                        dest    = 'targetBatchSize', type = int,
                        action  = 'store',
                        default = None,
                        help    = 'Batch size for parallel dispatch')
    
    parser.add_argument('-ec', '--expConverge',
                        dest    = 'expConverge',
                        action  = 'store_true',
                        default = False,
                        help    = 'Compute exp convergence')

    parser.add_argument('-log', '--log',
                        dest    = 'progressLog',
                        action  = 'store_true',
                        default = False,
                        help    = 'activate logging to console')

    parser.add_argument('-logFile', '--logFile', type = str,
                        dest    = 'progressLogFile',
                        action  = 'store',
                        default = None,
                        help    = 'file for progress Log')

    parser.add_argument('-z', '--progressInterval',
                        dest    = 'progressInterval', type = int,
                        action  = 'store',
                        default = 10,
                        help    = 'progress interval percentile')

    parser.add_argument('-v', '--verbose',
                        dest    = 'verbose',
                        action  = 'store_true',
                        default = False,
                        help    = 'verbose')

    parser.add_argument('-P', '--Plot',
                        dest    = 'Plot',
                        action  = 'store_true',
                        default = False,
                        help    = 'Plot CCM matrix')

    parser.add_argument('-title', '--title',
                        dest    = 'plotTitle', type = str,
                        action  = 'store',
                        default = "",
                        help    = 'CCM matrix plot title.')

    parser.add_argument('-fs', '--figureSize', nargs = 2,
                        dest    = 'figureSize', type = float,
                        action  = 'store',
                        default = [5,5],
                        help    = 'CCM matrix figure size.')

    parser.add_argument('-dpi', '--dpi',
                        dest    = 'dpi', type = int,
                        action  = 'store',
                        default = 150,
                        help    = 'CCM matrix figure dpi.')

    args = parser.parse_args()

    # progressLogFile overrides progressLog from bool to filename
    if args.progressLogFile is not None:
        args.progressLog = args.progressLogFile

    return args

#----------------------------------------------------------------------------
# Provide for cmd line invocation and clean module loading
if __name__ == "__main__":
    CCM_Matrix_CmdLine()
