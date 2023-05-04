import math
import os
import random
import torch
import torch.utils.data
import numpy as np
import pandas as pd
import pyworld as pw
import torch.nn.functional as F
import torchaudio
import json
import augment
import scipy.signal as ss
import soundfile as sf


def get_dataset_filelist(a):
    json_dir = 'Directory where the json file is'
    # json file only noisy wav (no reverb)(clean + noise)
    train_noisy_json = os.path.join(json_dir, 'train', 'noisy.json')
    train_clean_json = os.path.join(json_dir, 'train', 'clean.json')
    # json file for what environment you want to evaluate
    test_noisy_json = os.path.join(json_dir, 'test', 'all_noisy.json')
    test_clean_json = os.path.join(json_dir, 'test', 'clean.json')
    with open(train_noisy_json, 'r') as f:
        train_noisy = json.load(f)
    with open(train_clean_json, 'r') as f:
        train_clean = json.load(f)
    with open(test_noisy_json, 'r') as f:
        test_noisy = json.load(f)
    with open(test_clean_json, 'r') as f:
        test_clean = json.load(f)

    return [train_clean, train_noisy], [test_clean, test_noisy]

def hz_to_onehot(hz, freq_bins=360, bins_per_octave=48):
        # input: [b, frame_num]
        # output: [b, frame_num, freq_bins]
        fmin = 32.7
        hz = torch.tensor(hz)
        indexs = ( torch.log((hz+0.0000001)/fmin) / np.log(2.0**(1.0/bins_per_octave)) + 0.5).long()
        assert(torch.max(indexs) < freq_bins)
        mask = (indexs >= 0).long()
        # [B, frame_num, 1]
        mask = torch.unsqueeze(mask, dim=1)
        # [B, frame_num, freq_bins]
        onehot = F.one_hot(torch.clip(indexs, 0), freq_bins)
        # Mask the freq below fmin
        onehot = onehot * mask
        return onehot

def sort(noisy, clean):
    noisy.sort()
    clean.sort()

class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=True, sample_rate=None,
                 channels=None, convert=False):
        """
        Files should be a list of [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = int(self.length)
            out, sr = torchaudio.load(str(file), frame_offset=offset, num_frames=-1)
                
            target_sr = self.sample_rate or sr
            target_channels = self.channels or out.shape[0]

            if sr != target_sr:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                    f"{target_sr}, but got {sr}")
            if out.shape[0] != target_channels:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                    f"{target_channels}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return out, file
            # else:
            #     return out


class F0Dataset(torch.utils.data.Dataset):
    def __init__(self, traindata, segment_size, n_fft, num_mels,
                 hop_size, sampling_rate, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, train=True, length=4.5*16000, stride=1*16000, pad=True):
        self.traindata = traindata
        random.seed(1234)
        if shuffle:
            random.shuffle(self.traindata)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.cached_wav = None
        self.cached_clean_wav = None
        self.n_cache_reuse = n_cache_reuse
        self.device = device
        self.train = train
        self.num_examples = []
        if split == True:
            self.length = length
        else:
            self.length = None
        self.stride = stride or length
        self.pad = pad
        
        if train:
            self.rir_list = pd.read_csv('list_of_train_rir_wav_address.csv')
        else:
            self.rir_list = pd.read_csv('list_of_test_rir_wav_address.csv')


        clean = self.traindata[0]
        noisy = self.traindata[1]
        sort(noisy, clean)
        
        kw = {'length': self.length, 'stride': stride, 'pad': pad, 'sample_rate': sampling_rate, 'with_path': True}
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        ### shift augmentation
        augments = []
        shift = 8000
        shift_same = True
        augments.append(augment.Shift(shift, shift_same))
        self.augment = torch.nn.Sequential(*augments)
        
    def add_reverb(self, wav, rir_list):
        one_rir = np.random.choice(rir_list['path'])
        rir_sig, fs = sf.read(one_rir)
        if max(abs(rir_sig)) != 1:
            rir_sig = rir_sig / max(abs(rir_sig))
        cut = np.argmax(rir_sig)
        rir_sig = rir_sig[cut:]        
        wav = wav.squeeze().numpy()
        reverb = ss.convolve(rir_sig, wav)[:len(wav)]
        return torch.FloatTensor(reverb).unsqueeze(0)

    def __getitem__(self, index):        
        cleanaudio = self.clean_set[index][0]
        noisyaudio = self.noisy_set[index][0]

        filename = self.noisy_set[index][1].split('/')[-1].split('.')[0]

        if self.train:
            clean = cleanaudio
            noisy = noisyaudio
            
            distortion = np.random.choice(['noise', 'rir', 'both'], p=(0.3, 0.3, 0.4))
            noise = noisy - clean
            reverb = self.add_reverb(clean, self.rir_list)

            if distortion == 'both':
                fb_noisy = noise + reverb
            elif distortion == 'noise':
                fb_noisy = noisy
            elif distortion == 'rir':
                fb_noisy = reverb
                
            clean = clean.unsqueeze(0)
            fb_noisy = fb_noisy.unsqueeze(0)

            # Shift aumentation
            sources = torch.stack([fb_noisy, clean])
            sources = self.augment(sources)
            fb_noisy, clean = sources
            clean = clean.squeeze(0)
            fb_noisy = fb_noisy.squeeze(0)
            
            # Make the model train to estimate also from the clean speech
            noisyaudio = np.random.choice([fb_noisy, clean], p=(0.9, 0.1))
            cleanaudio = clean
        
        cleannpaudio = cleanaudio.squeeze().numpy().astype(np.float64)

        cleanf0, _ = pw.dio(cleannpaudio, self.sampling_rate, frame_period = self.hop_size/self.sampling_rate*1000)

        cleanf0 = torch.from_numpy(cleanf0)
        cleanf0 = cleanf0[:-1]
        cleanf0_quant = hz_to_onehot(cleanf0)

        return (cleanf0.squeeze(), cleanf0_quant.squeeze(), cleanaudio.squeeze(0), noisyaudio.squeeze(0), filename)


    def __len__(self):
        return len(self.noisy_set)
