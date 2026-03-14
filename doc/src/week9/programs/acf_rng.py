"""
Autocorrelation analysis of NumPy's default RNG (PCG64).
Includes:
  - Normalized autocorrelation function (ACF) C(t)
  - Integrated autocorrelation time tau_int
  - Naive standard deviation (ignores correlations)
  - Corrected standard deviation (uses covariance / tau_int)
  - Bootstrap standard deviation
  - All diagnostic plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate RNG sample
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample(N: int, seed: int = 42) -> np.ndarray:
    """Draw N uniform [0,1) samples from NumPy's default PCG64 RNG."""
    rng = np.random.default_rng(seed)
    return rng.random(N)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Autocorrelation function
# ─────────────────────────────────────────────────────────────────────────────

def autocorrelation(x: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    Compute the normalised autocorrelation function

        C(t) = <(x_i - <x>)(x_{i+t} - <x>)> / Var(x)

    using FFT for O(N log N) speed.  Returns C(0)..C(max_lag).
    """
    N = len(x)
    if max_lag is None:
        max_lag = N // 2
    max_lag = min(max_lag, N - 1)

    xc  = x - x.mean()
    # Zero-pad to next power of 2 for efficient FFT
    fft_size = 1
    while fft_size < 2 * N:
        fft_size <<= 1

    F   = np.fft.rfft(xc, n=fft_size)
    acf = np.fft.irfft(F * np.conj(F))[:N]
    # Normalise by lag-dependent number of pairs and by variance
    counts = N - np.arange(N)
    acf    = acf / counts          # normalise by number of pairs at each lag
    acf    = acf / acf[0]          # normalise so C(0) = 1  (= variance at lag 0)
    return acf[:max_lag + 1]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Integrated autocorrelation time
# ─────────────────────────────────────────────────────────────────────────────

def integrated_autocorrelation_time(acf: np.ndarray,
                                     c_factor: float = 6.0) -> tuple:
    """
    Estimate the integrated autocorrelation time

        tau_int = 1/2 + sum_{t=1}^{W} C(t)

    using the automatic windowing procedure of Madras & Sokal (1988):
    stop the sum at the first window W where W < c_factor * tau_int.

    Returns (tau_int, window W).
    """
    tau = 0.5
    for t in range(1, len(acf)):
        tau += acf[t]
        if t >= c_factor * tau:
            return tau, t
    return tau, len(acf) - 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. Standard deviations
# ─────────────────────────────────────────────────────────────────────────────

def std_naive(x: np.ndarray) -> float:
    """
    Naive standard deviation of the mean assuming i.i.d. samples:
        sigma_naive = std(x) / sqrt(N)
    """
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def std_with_covariance(x: np.ndarray, tau_int: float) -> float:
    """
    Corrected standard deviation of the mean accounting for correlations:
        sigma_corr = sqrt(2 * tau_int / N) * std(x)

    This follows from the exact formula
        Var(<x>) = (1/N^2) * sum_{i,j} C_{ij}
                 ≈ (2 * tau_int / N) * Var(x)
    for a stationary time series.
    """
    N = len(x)
    return float(np.std(x, ddof=1) * np.sqrt(2 * tau_int / N))


def std_bootstrap(x: np.ndarray,
                  n_boot: int = 2000,
                  block_size: int = 1,
                  seed: int = 0) -> float:
    """
    Bootstrap estimate of the standard deviation of the mean.

    For correlated data use block bootstrapping (block_size > 1).
    The block size should be of order tau_int or larger.

    Returns the standard deviation of the bootstrap distribution of means.
    """
    rng  = np.random.default_rng(seed)
    N    = len(x)
    means = np.empty(n_boot)

    if block_size <= 1:
        # Standard (i.i.d.) bootstrap
        for b in range(n_boot):
            sample  = rng.choice(x, size=N, replace=True)
            means[b] = sample.mean()
    else:
        # Circular block bootstrap
        n_blocks = int(np.ceil(N / block_size))
        for b in range(n_boot):
            starts = rng.integers(0, N, size=n_blocks)
            indices = np.concatenate([
                np.arange(s, s + block_size) % N for s in starts
            ])[:N]
            means[b] = x[indices].mean()

    return float(means.std(ddof=1))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Blocking method
# ─────────────────────────────────────────────────────────────────────────────

def blocking_analysis(x: np.ndarray, n_blocks_range=None) -> dict:
    """
    Blocking (or 'binning') analysis of a time series.

    The series of N values is split into n_b non-overlapping blocks of
    equal size b = floor(N / n_b).  The standard deviation of the mean is
    estimated from the n_b block averages:

        sigma_block(n_b) = std( {y_k} ) / sqrt(n_b)

    where y_k = (1/b) * sum_{i in block k} x_i.

    For block sizes b >> tau_int the block means are approximately i.i.d.,
    so sigma_block(n_b) reaches a plateau equal to the true sigma_mean.
    For b << tau_int correlations are still present and sigma_block is
    underestimated (biased low).

    The plateau onset gives a second, independent estimate of tau_int:
        tau_int ≈ b_plateau / 2

    Parameters
    ----------
    x              : time series
    n_blocks_range : iterable of n_b values to sweep; defaults to
                     all divisors of N from 4 up to N//4

    Returns
    -------
    dict with keys
        n_blocks      : array of n_b values tried
        block_sizes   : corresponding block sizes b = N // n_b
        sigma_block   : sigma_mean estimate at each n_b
        sigma_err     : statistical uncertainty on sigma_block
        plateau_idx   : index where the plateau is first detected
        plateau_value : sigma_mean at the plateau
        tau_from_block: tau_int estimated from plateau onset
    """
    N = len(x)

    if n_blocks_range is None:
        # Use all n_b such that b = N//n_b gives at least 4 blocks
        # and at most N//4 blocks (so b >= 4)
        bs = np.unique(np.round(np.geomspace(4, N // 4, 80)).astype(int))
        # Convert block sizes -> number of blocks
        n_blocks_range = np.unique((N // bs).astype(int))
        n_blocks_range = n_blocks_range[n_blocks_range >= 4]

    n_blocks_arr = np.asarray(sorted(set(int(v) for v in n_blocks_range
                                          if 4 <= v <= N // 4)), dtype=int)

    sigma_block = np.zeros(len(n_blocks_arr))
    sigma_err   = np.zeros(len(n_blocks_arr))

    for idx, n_b in enumerate(n_blocks_arr):
        b      = N // n_b                          # block size (discard remainder)
        blocks = x[:n_b * b].reshape(n_b, b)      # shape (n_b, b)
        y      = blocks.mean(axis=1)               # block means
        s      = y.std(ddof=1) / np.sqrt(n_b)     # sigma_mean from blocks
        sigma_block[idx] = s
        # Uncertainty on s itself: Var(s) ≈ s² / (2*(n_b-1))
        sigma_err[idx]   = s / np.sqrt(2 * (n_b - 1))

    block_sizes = N // n_blocks_arr

    # ── Detect plateau ─────────────────────────────────────────────────────
    # The plateau is where sigma_block stops rising as b increases.
    # Strategy: find the block size at which the relative change in sigma_block
    # (smoothed over a window) first falls below a threshold and stays there.
    # We work with sigma_block as a function of increasing b (decreasing n_b),
    # so we reverse the arrays.
    sb_by_b   = sigma_block[::-1]   # indexed by increasing b
    bs_sorted = block_sizes[::-1]

    window    = max(3, len(sb_by_b) // 10)
    smoothed  = np.convolve(sb_by_b, np.ones(window)/window, mode='valid')
    rel_change = np.abs(np.diff(smoothed) / (smoothed[:-1] + 1e-30))
    threshold  = 0.05   # < 5% change → plateau

    plateau_idx_rev = len(sb_by_b) - 1   # default: last point (largest b)
    for i in range(len(rel_change) - window):
        if np.all(rel_change[i:i+window] < threshold):
            plateau_idx_rev = i + window // 2
            break

    # Map back to original (decreasing b) indexing
    plateau_idx   = len(sigma_block) - 1 - plateau_idx_rev
    plateau_idx   = int(np.clip(plateau_idx, 0, len(sigma_block) - 1))
    plateau_value = float(np.median(
        sigma_block[max(0, plateau_idx - 3): plateau_idx + 4]))
    b_plateau     = int(block_sizes[plateau_idx])
    tau_from_block = b_plateau / 2.0

    return dict(
        n_blocks      = n_blocks_arr,
        block_sizes   = block_sizes,
        sigma_block   = sigma_block,
        sigma_err     = sigma_err,
        plateau_idx   = plateau_idx,
        plateau_value = plateau_value,
        tau_from_block= tau_from_block,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────────────────────────────────────

BG    = "#0d0f1a"
PANEL = "#131629"
ACCENT= "#4fc3f7"
WARM  = "#ff7043"
GREEN = "#69f0ae"
PURPLE= "#ce93d8"
GRID  = "#1e2340"
TEXT  = "#e8eaf6"


def apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  BG,
        'axes.facecolor':    PANEL,
        'axes.edgecolor':    GRID,
        'axes.labelcolor':   TEXT,
        'xtick.color':       TEXT,
        'ytick.color':       TEXT,
        'text.color':        TEXT,
        'grid.color':        GRID,
        'grid.linewidth':    0.6,
        'legend.facecolor':  PANEL,
        'legend.edgecolor':  GRID,
    })


def reset_style():
    plt.rcParams.update(plt.rcParamsDefault)


def plot_blocking(blk_result: dict, sigma_naive: float, sigma_corr: float,
                   sigma_boot: float, N: int, tau_int: float,
                   savename='blocking.png'):
    """
    Two-panel blocking-analysis figure.

    Left  — sigma_block vs block size b (x-axis = b, log scale).
             Shows the rise from underestimated (b << tau_int) to the
             plateau (b >> tau_int) with error bars and reference lines.

    Right — sigma_block vs number of blocks n_b (x-axis = n_b, log scale).
             Useful for seeing how many independent blocks are needed.
    """
    apply_dark_style()

    bs   = blk_result['block_sizes']
    nb   = blk_result['n_blocks']
    sig  = blk_result['sigma_block']
    err  = blk_result['sigma_err']
    pidx = blk_result['plateau_idx']
    plat = blk_result['plateau_value']
    b_pl = bs[pidx]
    tau_b= blk_result['tau_from_block']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)

    # ── Left: sigma vs block size ──────────────────────────────────────
    ax1.errorbar(bs, sig, yerr=err, fmt='o-', color=ACCENT,
                 ms=4, lw=1.6, elinewidth=1.0, capsize=3, alpha=0.85,
                 label='$\\sigma_{\\rm block}(b)$')
    ax1.axhline(plat, color=GREEN, lw=2.0, ls='--',
                label=f'Plateau  {plat:.6f}')
    ax1.axhline(sigma_naive, color=WARM, lw=1.5, ls=':',
                label=f'Naive    {sigma_naive:.6f}')
    ax1.axhline(sigma_corr,  color=PURPLE, lw=1.5, ls='-.',
                label=f'Cov-corr {sigma_corr:.6f}')
    ax1.axhline(sigma_boot,  color='#f48fb1', lw=1.5, ls=(0,(3,1,1,1)),
                label=f'Bootstrap {sigma_boot:.6f}')
    ax1.axvline(b_pl, color=GREEN, lw=1.2, ls='--', alpha=0.5,
                label=f'Plateau onset  $b={b_pl}$')
    ax1.axvspan(1, 2*tau_int, color=WARM, alpha=0.07,
                label=f'$b < 2\\tau_{{\\rm int}}={2*tau_int:.1f}$  (biased)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Block size  $b$', fontsize=13)
    ax1.set_ylabel('$\\sigma_{\\bar{x}}^{\\rm block}(b)$', fontsize=13)
    ax1.set_title('Blocking Analysis: $\\sigma$ vs Block Size',
                  fontweight='bold', color=TEXT)
    ax1.legend(fontsize=8.5, loc='lower right')
    ax1.grid(True, alpha=0.35, which='both')

    # ── Right: sigma vs number of blocks ──────────────────────────────
    ax2.errorbar(nb, sig, yerr=err, fmt='s-', color=ACCENT,
                 ms=4, lw=1.6, elinewidth=1.0, capsize=3, alpha=0.85,
                 label='$\\sigma_{\\rm block}(n_b)$')
    ax2.axhline(plat, color=GREEN, lw=2.0, ls='--',
                label=f'Plateau  {plat:.6f}')
    ax2.axhline(sigma_naive, color=WARM, lw=1.5, ls=':',
                label=f'Naive  {sigma_naive:.6f}')
    nb_pl = nb[pidx]
    ax2.axvline(nb_pl, color=GREEN, lw=1.2, ls='--', alpha=0.5,
                label=f'Plateau at $n_b={nb_pl}$')
    ax2.set_xscale('log')
    ax2.invert_xaxis()   # large n_b (small b) on left → mirrors left panel
    ax2.set_xlabel('Number of blocks  $n_b$  (log, reversed)', fontsize=13)
    ax2.set_ylabel('$\\sigma_{\\bar{x}}^{\\rm block}(n_b)$', fontsize=13)
    ax2.set_title('Blocking Analysis: $\\sigma$ vs $n_b$',
                  fontweight='bold', color=TEXT)
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(True, alpha=0.35, which='both')

    plt.suptitle(
        f'Blocking Analysis  ($N={N}$,  '
        f'$\\hat{{\\tau}}_{{\\rm int}}={tau_int:.3f}$,  '
        f'$\\tau_{{\\rm block}}\\approx{tau_b:.1f}$)',
        fontsize=14, fontweight='bold', color=TEXT, y=1.01)
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    reset_style()
    print(f'Saved {savename}')



def plot_acf(acf, tau_int, window, N, savename='acf.png'):
    """Plot the ACF with confidence band and tau_int window marker."""
    apply_dark_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)

    lags = np.arange(len(acf))

    # ── Left: full ACF ────────────────────────────────────────────────────
    ax = axes[0]
    ax.axhline(0, color=GRID, lw=1)
    conf = 1.96 / np.sqrt(N)
    ax.fill_between(lags, -conf, conf, color=ACCENT, alpha=0.12,
                    label=f'95% CI  (±{conf:.4f})')
    ax.bar(lags, acf, color=ACCENT, alpha=0.7, width=0.8)
    ax.axvline(window, color=WARM, lw=1.8, ls='--',
               label=f'Sokal window W={window}')
    ax.set_xlabel('Lag $t$', fontsize=13)
    ax.set_ylabel('$C(t)$', fontsize=13)
    ax.set_title('Normalised Autocorrelation Function  $C(t)$',
                 fontweight='bold', color=TEXT)
    ax.set_xlim(-0.5, min(len(acf)-1, 100) + 0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)

    # ── Right: log|ACF| to reveal exponential decay ───────────────────────
    ax2 = axes[1]
    pos = acf > 0
    ax2.semilogy(lags[pos], acf[pos], color=ACCENT, lw=1.5,
                 label='$C(t) > 0$')
    ax2.semilogy(lags[~pos], np.abs(acf[~pos]), 'o', color=WARM, ms=3,
                 alpha=0.6, label='$|C(t)|,\\ C(t)<0$')
    ax2.axhline(conf, color=GREEN, lw=1.4, ls='--',
                label=f'95% noise floor')
    ax2.set_xlabel('Lag $t$', fontsize=13)
    ax2.set_ylabel('$|C(t)|$  (log scale)', fontsize=13)
    ax2.set_title('Log-scale ACF  — Exponential Decay Check',
                  fontweight='bold', color=TEXT)
    ax2.set_xlim(-0.5, min(len(acf)-1, 100) + 0.5)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.35, which='both')

    plt.suptitle(
        f'Autocorrelation of NumPy PCG64 RNG  '
        f'($N={N}$,  $\\hat{{\\tau}}_{{\\rm int}}={tau_int:.3f}$)',
        fontsize=15, fontweight='bold', color=TEXT, y=1.01)
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    reset_style()
    print(f'Saved {savename}')


def plot_sample_and_running_mean(x, savename='sample.png'):
    """Plot raw samples and cumulative mean."""
    apply_dark_style()
    N = len(x)
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), facecolor=BG)

    ax = axes[0]
    ax.plot(x[:500], color=ACCENT, lw=0.8, alpha=0.85)
    ax.axhline(x.mean(), color=WARM, lw=1.5, ls='--',
               label=f'mean = {x.mean():.5f}')
    ax.set_xlabel('Sample index $i$', fontsize=12)
    ax.set_ylabel('$x_i$', fontsize=12)
    ax.set_title('First 500 RNG samples', fontweight='bold', color=TEXT)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)

    ax2 = axes[1]
    running = np.cumsum(x) / np.arange(1, N+1)
    ax2.plot(running, color=GREEN, lw=1.0, alpha=0.85)
    ax2.axhline(0.5, color=WARM, lw=1.5, ls='--', label='True mean = 0.5')
    ax2.set_xlabel('$N$', fontsize=12)
    ax2.set_ylabel('Running mean', fontsize=12)
    ax2.set_title('Convergence of the Sample Mean', fontweight='bold', color=TEXT)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    reset_style()
    print(f'Saved {savename}')


def plot_std_comparison(N_values, seed=42, savename='std_comparison.png'):
    """Compare three std estimators across sample sizes."""
    apply_dark_style()
    naive_list, corr_list, boot_list = [], [], []

    for N in N_values:
        rng  = np.random.default_rng(seed)
        x    = rng.random(N)
        acf  = autocorrelation(x, max_lag=min(N//2, 500))
        tau, _  = integrated_autocorrelation_time(acf)
        block   = max(1, int(2 * tau))
        naive_list.append(std_naive(x))
        corr_list.append(std_with_covariance(x, tau))
        boot_list.append(std_bootstrap(x, n_boot=1000,
                                        block_size=block, seed=seed))

    # Reference: exact sigma_mean for U[0,1] i.i.d.
    exact = np.array([1 / (np.sqrt(12) * np.sqrt(N)) for N in N_values])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)

    for data, label, col in [
        (naive_list, 'Naive  $\\sigma_N$',   ACCENT),
        (corr_list,  'Covariance-corrected', WARM),
        (boot_list,  'Block bootstrap',      GREEN),
        (exact,      'Exact  $1/\\sqrt{12N}$', PURPLE),
    ]:
        ax1.plot(N_values, data, 'o-', color=col, lw=2, ms=5, label=label)
    ax1.set_xlabel('Sample size $N$', fontsize=13)
    ax1.set_ylabel('$\\sigma_{\\bar{x}}$', fontsize=13)
    ax1.set_title('Standard Deviation of the Mean vs $N$',
                  fontweight='bold', color=TEXT)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.35)

    for data, label, col in [
        (naive_list, 'Naive',   ACCENT),
        (corr_list,  'Corrected', WARM),
        (boot_list,  'Bootstrap', GREEN),
    ]:
        ax2.loglog(N_values, data, 'o-', color=col, lw=2, ms=5, label=label)
    ax2.loglog(N_values, exact, '--', color=PURPLE, lw=2, label='Exact')
    ax2.set_xlabel('$N$  (log scale)', fontsize=13)
    ax2.set_ylabel('$\\sigma_{\\bar{x}}$  (log scale)', fontsize=13)
    ax2.set_title('Log–Log: $1/\\sqrt{N}$ Scaling',
                  fontweight='bold', color=TEXT)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.35, which='both')

    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    reset_style()
    print(f'Saved {savename}')


def plot_bootstrap_distribution(x, n_boot=3000, block_size=None,
                                 savename='bootstrap_dist.png'):
    """Plot the bootstrap distribution of the mean."""
    apply_dark_style()
    tau_val, _ = integrated_autocorrelation_time(
        autocorrelation(x, max_lag=min(len(x)//2, 500)))
    if block_size is None:
        block_size = max(1, int(2 * tau_val))

    rng   = np.random.default_rng(0)
    N     = len(x)
    means = np.empty(n_boot)
    for b in range(n_boot):
        n_blocks = int(np.ceil(N / block_size))
        starts   = rng.integers(0, N, size=n_blocks)
        indices  = np.concatenate([
            np.arange(s, s + block_size) % N for s in starts
        ])[:N]
        means[b] = x[indices].mean()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.hist(means, bins=60, color=ACCENT, edgecolor=BG, alpha=0.85,
            density=True, label='Bootstrap means')

    mu_b  = means.mean()
    sig_b = means.std(ddof=1)
    t = np.linspace(means.min(), means.max(), 400)
    gauss = np.exp(-0.5 * ((t - mu_b) / sig_b) ** 2) / (sig_b * np.sqrt(2*np.pi))
    ax.plot(t, gauss, color=WARM, lw=2.5, label='Gaussian fit')
    ax.axvline(mu_b, color=GREEN, lw=2, ls='--',
               label=f'$\\hat{{\\mu}}={mu_b:.5f}$')
    ax.axvline(mu_b - sig_b, color=PURPLE, lw=1.5, ls=':')
    ax.axvline(mu_b + sig_b, color=PURPLE, lw=1.5, ls=':',
               label=f'$\\hat{{\\sigma}}={sig_b:.6f}$')
    ax.set_xlabel('Bootstrap mean', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'Block Bootstrap Distribution of the Mean\n'
                 f'($N={N}$, block size={block_size}, $n_{{\\rm boot}}={n_boot}$)',
                 fontweight='bold', color=TEXT)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(savename, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    reset_style()
    print(f'Saved {savename}')


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    N    = 50_000
    SEED = 42

    x   = generate_sample(N, seed=SEED)
    acf = autocorrelation(x, max_lag=200)
    tau, W = integrated_autocorrelation_time(acf)

    sigma_naive  = std_naive(x)
    sigma_corr   = std_with_covariance(x, tau)
    block        = max(1, int(2 * tau))
    sigma_boot   = std_bootstrap(x, n_boot=3000, block_size=block, seed=0)

    blk = blocking_analysis(x)
    sigma_block_plateau = blk['plateau_value']
    tau_from_block      = blk['tau_from_block']

    print(f'N                       = {N}')
    print(f'Sample mean             = {x.mean():.6f}  (true = 0.5)')
    print(f'tau_int (Sokal)         = {tau:.4f}  (window W={W})')
    print(f'tau_int (blocking)      = {tau_from_block:.4f}')
    print(f'Naive sigma_mean        = {sigma_naive:.6f}')
    print(f'Covariance sigma_mean   = {sigma_corr:.6f}')
    print(f'Bootstrap sigma_mean    = {sigma_boot:.6f}')
    print(f'Blocking sigma_mean     = {sigma_block_plateau:.6f}')
    print(f'Exact sigma_mean (iid)  = {1/(np.sqrt(12)*np.sqrt(N)):.6f}')

    plot_sample_and_running_mean(x, savename='fig_sample.png')
    plot_acf(acf, tau, W, N, savename='fig_acf.png')
    plot_bootstrap_distribution(x, savename='fig_bootstrap.png')
    plot_blocking(blk, sigma_naive, sigma_corr, sigma_boot, N, tau,
                  savename='fig_blocking.png')
    plot_std_comparison([500, 1000, 5000, 10000, 50000],
                         savename='fig_std_comparison.png')
