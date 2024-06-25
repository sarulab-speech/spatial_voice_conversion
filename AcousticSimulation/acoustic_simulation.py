import pyroomacoustics as pa
import numpy as np
from typing import List, Tuple
import torchaudio
import os
from scipy.io.wavfile import write as write_wavfile
from dataclasses import dataclass

DEG_TO_RAD = np.pi / 180

@dataclass
class RoomParam:
    fs: int
    mic_distance: float = 0.15
    seed: int = 765
    room_size: np.ndarray = np.array([9., 7.5, 3.5]) 
    snr: float = float("inf")
    mic_center_pos_sigma: float = 0.1
    absorption: float = 0.5
    overlap: float = 0.0

@dataclass
class SourceInfo:
    fname: str
    deg: int
    r: float = 2.0
    amp: float = 0.8

def load_sources(sources, fs):
    wavs = []
    for source in sources:
        wav, file_sample_rate = torchaudio.load(source.fname)
        if file_sample_rate != fs:
            wav = torchaudio.transforms.Resample(file_sample_rate, fs)(wav)
        assert len(wav) == 1
        wav = wav[0].numpy()
        wavs.append(wav)
    return wavs

def zero_padding(wavs):
    x = np.zeros([len(wavs), max(map(len, wavs))], dtype=wavs[0].dtype)
    for i, wav in enumerate(wavs):
        x[i, :len(wav)] = wav
    
    return x

def conversation(wavs, overlap):
    l = []
    T = 0
    d = 0
    for x in wavs:
        z = np.hstack([np.zeros(int(T+overlap*d)), x])
        l.append(z)
        T += d
        d = len(z)
    return zero_padding(l)

def simulate(room_param: RoomParam, sources: List[SourceInfo], fname: str):
    # 乱数初期化
    np.random.seed(room_param.seed)

    # マイク位置
    mic_center = room_param.room_size/2 
    mic_relative_pos = np.array([[0., x*room_param.mic_distance, 0.] for x in [0.5, -0.5]])
    mic_pos = mic_relative_pos.T + mic_center[:, None]

    # 音源位置
    r = np.array([source.r for source in sources])
    theta = DEG_TO_RAD * np.array([source.deg for source in sources])
    source_relative_pos = r * np.array([np.cos(theta), np.sin(theta), np.zeros(len(theta))])
    source_pos = source_relative_pos + mic_center[:, None]

    # 音源信号
    wavs = load_sources(sources, room_param.fs)
    if room_param.overlap == 0.0:
        source_signals = zero_padding(wavs)
    else:
        source_signals = conversation(wavs, room_param.overlap)

    # 部屋の作成
    room = pa.ShoeBox(room_param.room_size, fs=room_param.fs, max_order=17, absorption=room_param.absorption)
    room.add_microphone_array(pa.MicrophoneArray(mic_pos, fs=room.fs))
    for s, wav in enumerate(source_signals):
        wav = wav * (sources[s].amp / np.std(wav))
        room.add_source(source_pos[:, s], signal=wav)
    
    # シミュレーションを回す
    room.simulate(snr=room_param.snr)
    result = room.mic_array.signals

    # 保存する
    save_dir = os.path.dirname(fname)
    if len(save_dir) > 0:
        os.makedirs(save_dir, exist_ok=True)
    write_wavfile(
        fname, 
        room_param.fs, 
        (result * np.iinfo(np.int16).max / 20).astype(np.int16).T
    )

    return room

if __name__ == '__main__':
    sample_rate = 24000

    sfiles = [f"test/{str(i).zfill(3)}.wav" for i in [1,2]]
    degs = [60, -60]
    sources = [SourceInfo(fname, deg) for fname, deg in zip(sfiles, degs)]

    param = RoomParam(fs=sample_rate)

    room = simulate(param, sources, "output.wav")
    print(room.measure_rt60())

