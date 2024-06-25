import numpy as np
import numpy.linalg as LA
import shutil
from scipy.io import wavfile
from dataclasses import dataclass

@dataclass
class GC_AUX_IVA_PARAMS:
    steering_vector: np.array
    lamda: float = 1.0
    tempering_rate: float = 0.5
    total_step: int = 100
    eps: float = 1e-15

# call constructor as function
# BSS result will be saved in member variable
class GC_AUX_IVA:
    def __init__(self, spec, params):
        Y = spec.transpose(2, 1, 0)
        n_frames, n_freq, n_ch = Y.shape
        N = 2*(n_freq-1)
        
        YYH = np.einsum("tfn,tfm->tfnm", Y, np.conj(Y))

        # freq, ch
        d = params.steering_vector
        ddH = np.einsum("fn,fm->fnm", d, np.conj(d))

        ## separation
        # freq, out_ch, in_ch
        W = np.tile(np.eye(n_ch, dtype=complex), (n_freq, 1, 1))
        for step in range(params.total_step):
            for s in range(n_ch):
                y = np.einsum("fm,tfm->tf", W[:,s,:], Y)
                r = LA.norm(y, axis=1) / (n_freq**0.5)
                V = np.einsum("tfnm,t->fnm", YYH, 1/(r+params.eps)) / n_frames

                if s != n_ch-1:
                    lamda = params.lamda \
                        if (step / params.total_step) < params.tempering_rate \
                        else 0.
                    D = V + lamda * ddH
                else:
                    D = V
                Dinv = LA.pinv(D)
                a = LA.pinv(W)[:,:,s]
                
                u = np.einsum("fnm,fm->fn", Dinv, a)
                h = np.abs(np.einsum("fn,fnm,fm->f", np.conj(u), D, u))
                w = u / (np.sqrt(h)[:,None])
                W[:,s,:] = np.conj(w)
        W = W.transpose(0, 2, 1)

        ## projection back
        Xt= np.einsum("tfm,fmn->ftn", Y, W)
        y = Y[:,:,0]
        c = np.einsum("fnt,tf->fn", LA.pinv(Xt), y)
        W = np.einsum("fmn,fn->fmn", W, c)
        
        ## set first component as target output
        W = np.array(W[:,:,::-1])
        result = np.einsum("tfm,fmn->tfn", Y, W).transpose(2, 1, 0)
        
        # save result
        self.W = W
        self.output = result

if __name__ == '__main__':
    from util import load_stft, inv_stft
    
    gc_aux_iva = GC_AUX_IVA(90, 0.15)

    ifname = "../data/reverb/0/input.wav"
    spec = load_stft(ifname, 2**12)
    result, W = gc_aux_iva(spec)
    
    inv_stft(result, "tmp/gc_iva.wav")
    
