"""Channel data utilities: loading, AWGN corruption, pilot interpolation, and dataset construction.

Channels are OFDM resource grids of 72 subcarriers x 14 OFDM symbols (the VehA
dataset released with ChannelNet). A complex grid is turned into two real-valued
"images": the real parts of all N channels are stacked on top of the imaginary
parts, giving an array of shape (2N, 72, 14, 1).
"""

import numpy as np
import torch
from scipy import interpolate
from scipy.io import loadmat

N_SUBCARRIERS = 72
N_SYMBOLS = 14


def load_perfect_channel(path, key="My_perfect_H"):
    """Load the noiseless complex channel tensor of shape (N, 72, 14) from a .mat file."""
    return loadmat(path)[key]


def add_awgn_noise(perfect_channel, snr_db):
    """Add complex AWGN to a channel so that the result has the requested SNR (in dB)."""
    snr_linear = 10 ** (snr_db / 10.0)
    signal_power = np.mean(np.abs(perfect_channel) ** 2)
    noise_power = signal_power / snr_linear

    # Real and imaginary parts are independent Gaussians, each with half the noise power
    noise_real = np.sqrt(noise_power / 2) * np.random.randn(*perfect_channel.shape)
    noise_imag = np.sqrt(noise_power / 2) * np.random.randn(*perfect_channel.shape)
    return perfect_channel + noise_real + 1j * noise_imag


def pilot_positions(num_pilots):
    """Return (subcarrier, symbol) coordinates of the pilots in the 72 x 14 grid."""
    patterns = {
        48: [14 * i for i in range(1, 72, 6)] + [4 + 14 * i for i in range(4, 72, 6)]
            + [7 + 14 * i for i in range(1, 72, 6)] + [11 + 14 * i for i in range(4, 72, 6)],
        36: [14 * i for i in range(1, 72, 6)] + [6 + 14 * i for i in range(4, 72, 6)]
            + [11 + 14 * i for i in range(1, 72, 6)],
        24: [14 * i for i in range(1, 72, 9)] + [6 + 14 * i for i in range(4, 72, 9)]
            + [11 + 14 * i for i in range(1, 72, 9)],
        16: [4 + 14 * i for i in range(1, 72, 9)] + [9 + 14 * i for i in range(4, 72, 9)],
        8: [4 + 14 * i for i in range(5, 72, 18)] + [9 + 14 * i for i in range(8, 72, 18)],
    }
    if num_pilots not in patterns:
        raise ValueError(f"Unsupported number of pilots: {num_pilots}. "
                         f"Choose one of {sorted(patterns)}.")
    idx = patterns[num_pilots]
    rows = np.array([x // N_SYMBOLS for x in idx], dtype=float)
    cols = np.array([x % N_SYMBOLS for x in idx], dtype=float)
    return rows, cols


def _interpolate_grid(values, rows, cols, method):
    """Interpolate pilot values onto the full 72 x 14 grid."""
    if method == "rbf":
        f = interpolate.Rbf(rows, cols, values, function="gaussian")
        X, Y = np.meshgrid(range(N_SUBCARRIERS), range(N_SYMBOLS))
        return f(X, Y).T
    if method == "spline":
        tck = interpolate.bisplrep(rows, cols, values)
        return interpolate.bisplev(range(N_SUBCARRIERS), range(N_SYMBOLS), tck)
    raise ValueError(f"Unknown interpolation method: {method}. Use 'rbf' or 'spline'.")


def channel_to_image(channel):
    """(N, 72, 14) complex -> (2N, 72, 14, 1) real: real parts first, then imaginary parts."""
    return np.concatenate((np.real(channel), np.imag(channel)), axis=0)[..., np.newaxis]


def interpolation(noisy, num_pilots, method="rbf"):
    """Coarse channel estimate: sample the noisy channel at the pilots and interpolate.

    This is the network input (the "low-resolution image" of ChannelNet).
    Returns an array of shape (2N, 72, 14, 1).
    """
    rows, cols = pilot_positions(num_pilots)
    r_idx, c_idx = rows.astype(int), cols.astype(int)

    interp = np.zeros(noisy.shape, dtype=complex)
    for i in range(len(noisy)):
        pilots = noisy[i, r_idx, c_idx]
        interp[i] = (_interpolate_grid(pilots.real, rows, cols, method)
                     + 1j * _interpolate_grid(pilots.imag, rows, cols, method))
    return channel_to_image(interp)


def make_snr_dataset(perfect, snr_db, num_pilots, method="rbf"):
    """Build (input, label) arrays for one SNR regime.

    Inputs are interpolated pilot estimates of the channel corrupted with AWGN at
    `snr_db`; labels are the noiseless channels. Both have shape (2N, 72, 14, 1).
    """
    noisy = add_awgn_noise(perfect, snr_db)
    return interpolation(noisy, num_pilots, method), channel_to_image(perfect)


def to_tensor(images):
    """(N, H, W, C) numpy array -> (N, C, H, W) float tensor."""
    return torch.FloatTensor(images).permute(0, 3, 1, 2)
