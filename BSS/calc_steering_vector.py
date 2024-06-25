import numpy as np
import numpy.linalg as LA

DEG_TO_RAD = np.pi / 180

def get_steering_vector(target_direction, fs, N, width, sound_speed=340):
    assert type(target_direction) is int, "target direction [deg]"
    freqs = np.arange(0, N/2+1, 1) * fs / N
    omegas = 2 * np.pi * freqs
    mic_x = np.array([0, -width])
    theta = target_direction * DEG_TO_RAD

    phase = 1.j * omegas[:,None] * mic_x[None,:] * np.sin(theta) / sound_speed

    a = np.exp(phase)
    a = a / LA.norm(a, axis=1)[:,None]
    return a

