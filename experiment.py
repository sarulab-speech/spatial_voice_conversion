from AcousticSimulation import RoomParam, SourceInfo
from BSS import GC_AUX_IVA_PARAMS
from VC import VC_PARAM

from spatial_voice_conversion import spatical_vc, ideal_spatical_vc

def execute(
        output_top_dir: str,
        source_info: SourceInfo,
        noise_info: SourceInfo,
        room_param: RoomParam,
        bss_param: GC_AUX_IVA_PARAMS,
        vc_param: VC_PARAM
    ):
    # spatial vc
    spatical_vc(
        output_top_dir,
        [source_info, noise_info],
        room_param,
        bss_param,
        vc_param
    )

    # ideal
    ideal_spatical_vc(
        output_top_dir,
        [source_info, noise_info],
        room_param,
        bss_param,
        vc_param
    )

def test():
    from BSS import get_steering_vector
    from config import fs, stft_window_size
    output_top_dir = "output/test"
    source_info = SourceInfo("test_data/jvs001.wav", -60)
    noise_info  = SourceInfo("test_data/jvs002.wav",  60)
    room_param = RoomParam(fs)
    steering_vector = get_steering_vector(
        source_info.deg, fs, stft_window_size, room_param.mic_distance
    )
    bss_param = GC_AUX_IVA_PARAMS(steering_vector)
    vc_param = VC_PARAM(3, 0)

    execute(
        output_top_dir,
        source_info,
        noise_info,
        room_param,
        bss_param,
        vc_param
    )

if __name__ == '__main__':
    test()