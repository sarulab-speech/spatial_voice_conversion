import os
import copy
import numpy as np
import numpy.linalg as LA
from typing import List

from AcousticSimulation import RoomParam, SourceInfo
from BSS import GC_AUX_IVA_PARAMS
from VC import VC_PARAM

from AcousticSimulation import simulate
from BSS import GC_AUX_IVA
from VC import voice_conversion

from util import load_stft, load_stfts, inv_stft, split_stereo
from config import fs, stft_window_size

def spatical_vc(output_top_dir: str, srcs: List[SourceInfo], room_param: RoomParam, bss_param: GC_AUX_IVA_PARAMS, vc_param: VC_PARAM):
    # step 0: simulate observed signal
    obsereved_signal = os.path.join(output_top_dir, "0_mixed_signal", "output.wav")
    simulate(
        room_param, 
        srcs,
        obsereved_signal
    )
    Y = load_stft(obsereved_signal, stft_window_size)
    n_ch = Y.shape[0]
    src_num = n_ch

    # step 1: separation
    fname_separation = [os.path.join(output_top_dir, "1_separation", f"output_{i}.wav") for i in range(n_ch)]
    bss = GC_AUX_IVA(Y, bss_param)
    for i in range(src_num):
        inv_stft(bss.output[i], fname_separation[i])

    # step 2: voice conversion    
    fname_vc = os.path.join(output_top_dir, "2_voice_conversion", f"output.wav")
    voice_conversion(fname_separation[0], fname_vc, vc_param)

    # step3: remixing
    Ahat = LA.pinv(bss.W)   # freq, src, ch
    configs = [
        (fname_vc, "3a_inverse", None),
        (fname_vc, "3b_steering", bss_param.steering_vector)
    ]
    for src_fname, suf, d in configs:
        fname = fname_separation.copy()
        fname[0] = src_fname
        spec = load_stfts(fname, stft_window_size)

        mixing_matrix = np.array(Ahat)
        if d is not None:
            d = d / (d[:,0][:,None])
            mixing_matrix[:,0,:] = d / (d[:,0][:,None])

        Z = np.einsum("fsc,sft->scft", mixing_matrix, spec)

        out_dir = os.path.join(output_top_dir, suf)
        for s in range(src_num):
            inv_stft(
                Z[s], 
                os.path.join(out_dir, f"stereo_{s}.wav"),
                [os.path.join(out_dir, f"monaural{s}{c}.wav") for c in range(n_ch)]
            )

        inv_stft(
            sum(Z), 
            os.path.join(out_dir, "output.wav"),
            [os.path.join(out_dir, f"ch_{c}.wav") for c in range(n_ch)]
        )

def ideal_spatical_vc(output_top_dir: str, srcs: List[SourceInfo], room_param: RoomParam, bss_param: GC_AUX_IVA_PARAMS, vc_param: VC_PARAM):
    ideal_out_dir = os.path.join(output_top_dir, "3z_ideal_result")
    n_ch = len(srcs)
    del output_top_dir

    converted_source = copy.deepcopy(srcs[0])
    converted_source.fname = os.path.join(ideal_out_dir, "converted_source.wav")
    voice_conversion(
        srcs[0].fname,
        converted_source.fname,
        vc_param
    )

    srcs = srcs.copy()
    srcs[0] = converted_source

    ofname = os.path.join(ideal_out_dir, "output.wav")
    simulate(
        room_param, 
        srcs,
        ofname
    )

    split_stereo(
        ofname,
        [os.path.join(ideal_out_dir, f"ch_{c}.wav") for c in range(n_ch)] 
    )

    for s, src in enumerate(srcs):
        ofname = os.path.join(ideal_out_dir, f"stereo_{s}.wav")
        simulate(
            room_param,
            [src],
            ofname
        )
        split_stereo(
            ofname, 
            [os.path.join(ideal_out_dir, f"monaural{s}{c}.wav") for c in range(n_ch)]
        )

