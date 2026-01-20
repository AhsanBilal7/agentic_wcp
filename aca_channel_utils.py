

import numpy as np


def add_awgn_noise(perfect_channel, snr_db):
    """
    Add AWGN (Additive White Gaussian Noise) to perfect channel
    
    Args:
        perfect_channel: Perfect channel matrix (complex values)
        snr_db: Signal-to-Noise Ratio in dB
    
    Returns:
        noisy_channel: Channel with AWGN noise added
    """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10.0)
    
    # Calculate signal power
    signal_power = np.mean(np.abs(perfect_channel) ** 2)
    
    # Calculate noise power based on desired SNR
    noise_power = signal_power / snr_linear
    
    # Generate complex AWGN noise
    # Real and imaginary parts are independent Gaussian
    noise_real = np.sqrt(noise_power / 2) * np.random.randn(*perfect_channel.shape)
    noise_imag = np.sqrt(noise_power / 2) * np.random.randn(*perfect_channel.shape)
    noise = noise_real + 1j * noise_imag
    
    # Add noise to signal
    noisy_channel = perfect_channel + noise
    
    return noisy_channel