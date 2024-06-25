from scipy.io import wavfile
import scipy.signal as sp
import numpy as np
import numpy.linalg as LA
import os
import torch
import torchaudio

from config import fs, window

def calc_stft(x, N):
    f, t, stft_data = sp.stft(x, fs=fs, window=window, nperseg=N)
    return stft_data

def load_stft(fname, N):
    fs_, audio = wavfile.read(fname)
    x = audio.astype(np.float32).T / np.iinfo(np.int16).max
    if fs != fs_:
        x = torchaudio.transforms.Resample(fs_, fs)(torch.from_numpy(x)).numpy()
    
    return calc_stft(x, N)
    
def load_stfts(fnames, N):
    specs = [load_stft(f, N) for f in fnames]
    lens = [x.shape[1] for x in specs]
    X = np.zeros((len(fnames), len(specs[0]), max(lens)), dtype=specs[0].dtype)
    for i,(spec,l) in enumerate(zip(specs, lens)):
        X[i, :, :l] = spec
    return X

def inv_stft(spec, fname, monoral_fnames=None):
    N = 2*(spec.shape[-2]-1)
    t, ds_out = sp.istft(spec, fs=fs, window=window, nperseg=N)
    if fname is not None:
        make_file_dir(fname)
        result = (ds_out*np.iinfo(np.int16).max).astype(np.int16)
        if len(result.shape) == 2:
            result = result.T
        wavfile.write(fname, fs, result)

        if monoral_fnames:
            assert len(monoral_fnames) == result.shape[1]
            for c, m_fname in enumerate(monoral_fnames):
                wavfile.write(m_fname, fs, result[:,c])

    return ds_out

def split_stereo(ifname, ofnames):
    fs, x = wavfile.read(ifname)
    assert x.shape[1] == len(ofnames)
    for c, ofname in enumerate(ofnames):
        make_file_dir(ofname)
        wavfile.write(ofname, fs, x[:,c])

def make_file_dir(fname):
    dir_name = os.path.dirname(fname)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)
