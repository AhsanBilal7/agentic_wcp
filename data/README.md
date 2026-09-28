# Data

Place the noiseless VehA channel file here:

```
data/Perfect_H_40000.mat     # variable "My_perfect_H", complex, shape (40000, 72, 14)
```

It is the **"Perfect channels – VehA model (without noise)"** file released with
[ChannelNet](https://github.com/Mehran-Soltani/ChannelNet)
([direct Google Drive link](https://drive.google.com/file/d/1H5GiEWITfM00R4BS2uC3SiBLR0EZKX8m/view?usp=sharing)).

Each sample is one OFDM resource grid of 72 subcarriers × 14 OFDM symbols. You do **not**
need the noisy files from ChannelNet: the scripts add AWGN at every SNR themselves
(`channel_utils.add_awgn_noise`).

To keep the file somewhere else, pass `--data_path /path/to/Perfect_H_40000.mat`.
`.mat` files are ignored by git.
