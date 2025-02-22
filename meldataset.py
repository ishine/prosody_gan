# coding: utf-8

import random
import numpy as np
import random
import librosa
import pyworld as pw

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d

np.random.seed(1)
random.seed(1)


class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self, hps, fn_rawtexts):
        self.hps = hps
        self.fn_rawtexts = [x.split('|') for x in fn_rawtexts]
        if self.hps.mode == 'spectrum':
            random.shuffle(self.fn_rawtexts)
        self.data_path = hps.data_path
        self.mel_basis = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000)

    def get_cont_f0(self, f0):
        f0 += 150 if (f0 == 0).all() else 0
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        start_idx = np.where(f0 == start_f0)[0][0]
        end_idx = np.where(f0 == end_f0)[0][-1]
        f0[:start_idx] = start_f0
        f0[end_idx:] = end_f0
        nz_frames = np.where(f0 != 0)[0]
        f = interp1d(nz_frames, f0[nz_frames])
        cont_f0 = f(np.arange(0, f0.shape[0]))
        return cont_f0

    def get_mel_energy(self, y):
        y = np.append(y[0], y[1:] - 0.85 * y[:-1])
        linear = librosa.stft(y=y, n_fft=1024, hop_length=160, win_length=800)
        mag = np.abs(linear)  # (1+n_fft//2, T)
        energy = np.linalg.norm(mag, axis=0)
        mel = np.dot(self.mel_basis, mag)  # (n_mels, T)
        mel = np.log10(np.maximum(1e-10, mel))
        mel = mel.T.astype(np.float32)  # (T, n_mels)
        return mel, energy

    def front_end(self, filename, raw_text):
        ### This function needs to be improved to adapt to the frontend.

        # lp is the phoneme length, d is the dimension of linguistic features, and lc is the character length.
        ling_fea = None # linguistic features, with dimensions [lp, d]
        textid = None # Input for the BERT network, text IDs, with dimensions [lc]
        char_len = None # Number of phonemes per character, with dimensions [lc]. The sum of this vector must equal lp.
        return ling_fea, textid, char_len

    def __len__(self):
        return len(self.fn_rawtexts)

    def __getitem__(self, idx):
        if self.hps.mode in ['prosody_code', 'duration']:
            
            outs = []
            center_filename, _ = self.fn_rawtexts[(idx+self.hps.sentence_k) % len(self.fn_rawtexts)]
            for sentence_i in range(self.hps.sentence_k * 2 + 1):
                out = {}
                filename, raw_text = self.fn_rawtexts[(idx+sentence_i) % len(self.fn_rawtexts)]
                ndtag = np.array([int(filename[-1])], dtype=np.int32)
                ling_fea, textid, char_len = self.front_end(filename, raw_text)

                out['filename'] = filename
                out['ndtag'] = torch.Tensor(ndtag).long()
                out['ling'] = torch.Tensor(ling_fea)
                out['len_ling'] = torch.Tensor([len(ling_fea)])
                out['out_paragraph'] = torch.Tensor([1 if filename.split('_')[0] != center_filename.split('_')[0] else 0])
                out['sindex'] = torch.Tensor([sentence_i])

                if self.hps.mode == 'prosody_code':
                    out['psd_code'] = torch.Tensor(np.load(f'{self.data_path}/psd_code/{filename}.npy'))
                if self.hps.mode == 'duration':
                    out['dur'] = torch.Tensor(np.load(f'{self.data_path}/duration/{filename}.npy'))

                out['char_len'] = torch.Tensor(char_len)
                out['textid'] = torch.Tensor(textid).long()

                if self.hps.character_voice_encoder:
                    if self.hps.get('ref_wav',''):
                        out['wav'] = librosa.load(self.hps.ref_wav, sr=16000)[0]
                    else:
                        out['wav'] = librosa.load(f'{self.data_path}/wav/{filename}.wav', sr=16000)[0]
                outs.append(out)

            out = {}
            for key in outs[0]:
                if key in ['filename']:
                    out[key + '_all'] = '%'.join([x[key] for x in outs])
                elif key in ['wav']:
                    out[key + "_all"] = [x[key] for x in outs]
                elif key in ['textid', 'char_len']:
                    # This operation may need to be modified to accommodate the input format of the BERT model, such as adding [SEP] between sentences.
                    out[key + "_all"] = torch.cat([x[key] for x in outs]) 
                else:
                    out[key + "_all"] = torch.cat([x[key] for x in outs])

                out[key] = outs[len(outs) // 2][key]
            return out


        elif self.hps.mode == 'spectrum':
            out = {}
            filename, raw_text = self.fn_rawtexts[idx]
            ndtag = np.array([int(filename[-1])], dtype=np.int32)
            ling_fea, _, _ = self.front_end(filename, raw_text)
            wav, _ = librosa.load(f'{self.data_path}/wav/{filename}.wav', sr=16000)
            if len(wav) < 67*160:
                wav = np.pad(wav, (0, 67*160 - len(wav)), 'constant', constant_values=0)
            mel, energy = self.get_mel_energy(wav)
            dur = np.load(f'{self.data_path}/duration/{filename}.npy')
            dur[-1] += len(mel) - sum(dur)
            f0, _ = pw.dio(wav.astype(np.float64), 16000, frame_period=160 / 16000 * 1000)
            f0 = self.get_cont_f0(f0)
            out['filename'] = filename
            out['ndtag'] = torch.Tensor(ndtag).long()
            out['ling'] = torch.Tensor(ling_fea)
            out['len_ling'] = torch.Tensor([len(ling_fea)])
            out['pitch'] = torch.Tensor(f0)
            out['energy'] = torch.Tensor(energy)
            out['dur'] = torch.Tensor(dur)
            out['mel'] = torch.Tensor(mel)
            if self.hps.character_voice_encoder:
                if self.hps.get('ref_wav',''):
                    out['wav'] = librosa.load(self.hps.ref_wav, sr=16000)[0]
                else:
                    out['wav'] = wav
            return out

class Collater(object):
    def merge(self, key, samples):
        values = [s[key] for s in samples]
        size = max(v.shape[0] for v in values)
        oshape = (len(values), size) + values[0].shape[1:]
        res = values[0].new(*oshape).fill_(0)
        mask = values[0].new(*oshape[:2]).fill_(0)
        for i, v in enumerate(values):
            res[i, : len(v)] = v
            mask[i, : len(v)] = 1
        return res, mask

    def __call__(self, samples):
        out = {}
        for key in samples[0]:
            if key in ['mel', 'ling', 'ling_all', 'char_len_all']:
                x, x_mask = self.merge(key, samples)
                out.update({key: x, key + '_mask': x_mask})
            elif key in ['filename', 'filename_all', 'wav', 'wav_all']:
                out.update({key: [s[key] for s in samples]})
            else:
                out.update({key: self.merge(key, samples)[0]})
        return out



def build_dataloader(hps, validation=False, all_data = False):
    if all_data:
        fn_rawtexts = open(hps.val_data,'r').read().strip().split('\n') +  open(hps.train_data,'r').read().strip().split('\n')
    else:
        fn_rawtexts = open(hps.val_data if validation else hps.train_data,'r').read().strip().split('\n')
    dataset = FilePathDataset(hps, fn_rawtexts)
    collate_fn = Collater()
    data_loader = DataLoader(dataset, batch_size=hps.batch_size, shuffle=False, num_workers=2 if validation else 8, drop_last=(not validation), collate_fn=collate_fn, pin_memory=True)
    return data_loader
