#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

# path= demucsdata/tr
# if [[ ! -e $path ]]; then
#     mkdir -p $path
# fi


from collections import namedtuple
import json
from pathlib import Path
import math
import os
import sys
import torchaudio
from torch.nn import functional as F

Info = namedtuple("Info", ["length", "sample_rate", "channels"])

def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta


if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_audio_files(path)
    json.dump(meta, sys.stdout, indent=4)
