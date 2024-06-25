from dataclasses import dataclass
import os
import sys
import subprocess as sp

@dataclass
class VC_PARAM:
    target_speaker: int
    key: int

def make_file_dir(fname):
    dir_name = os.path.dirname(fname)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

def voice_conversion(ifname, ofname, param):
    make_file_dir(ofname)
    here = os.path.dirname(os.path.abspath(__file__))
    relative_path_root = os.path.dirname(os.path.abspath(sys.argv[0]))
    cmd = " ".join([
        "cd", os.path.join(here, "DDSP-SVC"), ";",
        "python3", "main_diff.py",
        "-i",  os.path.join(relative_path_root, ifname),
        "-o",  os.path.join(relative_path_root, ofname),
        "-id", str(param.target_speaker),
        "-k",  str(param.key),
        "-diff", "exp/diffusion-test/model_100000.pt"
    ])
    proc = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    if proc.returncode != 0:
        print(proc.stderr.decode())
        
if __name__ == '__main__':
    voice_conversion("../test_data/jvs001.wav", "output.wav", VC_PARAM(3, 0))

