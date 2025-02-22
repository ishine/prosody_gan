import torch
import torch.nn as nn
import torch.nn.functional as F

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    # [00011111]
    return mask


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(batch, (0, max_len - batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def mean_by_lengths(out, dur=None):
    if dur is None:
        return out.mean(1).unsqueeze(1)
    flag = 0
    if len(out.shape) == 2:
        out = out.unsqueeze(-1)
        flag = 1
    new = torch.zeros(dur.shape[0], dur.shape[1], out.shape[2]).to(out)

    for i in range(dur.shape[0]):
        s = 0
        j = 0
        while j < dur.shape[1] and dur[i, j] != 0:
            new[i, j] = out[i, s : s + int(dur[i, j].item())].mean(0)
            s += int(dur[i, j].item())
            j += 1
    if flag:
        new = new.squeeze(-1)
    return new


def repeat_by_lengths(x, duration):
    output = list()
    mel_len = list()
    if len(x.shape) == 2:
        x = x.unsqueeze(-1)
        for batch, expand_target in zip(x, duration):
            expanded = expand_by_length(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])
        output = pad(output)
        len(x.shape)
        return output.squeeze(-1)
    else:
        for batch, expand_target in zip(x, duration):
            expanded = expand_by_length(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])
        output = pad(output)
        len(x.shape)
        return output


def expand_by_length(batch, predicted):
    out = list()
    for i, d in enumerate(predicted):
        expand_size = d.item()
        out.append(batch[i].expand(int(expand_size), -1))
    out = torch.cat(out, 0)
    return out


def select_middle_by_lengths(out, dur=None):
    if dur is None:
        return out[:, out.shape[1] // 2 : out.shape[1] // 2 + 1]
    sw = dur.shape[1]
    if sw == 1:
        return out
    new = torch.zeros(out.shape).to(out)
    for i in range(out.shape[0]):
        wide = int(dur[i, sw // 2].item())
        start = int(dur[i, : sw // 2].sum().item())
        new[i, :wide] = out[i, start : start + wide]
    return new[:, : int(dur[:, sw // 2].max().item())]


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(output.device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, max_len, src_mask):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class ReferenceEncoder(nn.Module):
    def __init__(self, hps):
        super(ReferenceEncoder, self).__init__()
        self.hps = hps
        filters = [2, 1024, 512, 512, 256, 256, 128]

        K = len(filters) - 1
        convs = [nn.Conv1d(filters[i], filters[i + 1], kernel_size=3, padding=1) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features=filters[i + 1]) for i in range(K)])
        self.lstm1 = nn.LSTM(input_size=filters[-1], hidden_size=256 // 2, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256 + 1, hidden_size=256 // 2, batch_first=True, bidirectional=True)
        self.output_linear = nn.Linear(256, hps.psdcode_dim)

    def forward(self, sample):
        len_ling = sample['len_ling'].squeeze(-1).cpu()
        inputs_frame = torch.stack([torch.log(sample['pitch'] + 1), sample['energy']], -1)
        frame_length = sample['mel_mask'].sum(1).long()
        out = inputs_frame.transpose(1, 2)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)
        out = out.transpose(1, 2)
        x = nn.utils.rnn.pack_padded_sequence(out, frame_length.cpu(), batch_first=True, enforce_sorted=False)
        self.lstm1.flatten_parameters()
        out, _ = self.lstm1(x)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        dur = sample['dur'].unsqueeze(-1)
        dur = torch.cat([mean_by_lengths(out, dur), dur], -1)
        out = nn.utils.rnn.pack_padded_sequence(dur, len_ling, batch_first=True, enforce_sorted=False)
        self.lstm2.flatten_parameters()
        out, _ = self.lstm2(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.output_linear(out)
        return out


class CharactorVoiceEncoder(nn.Module):
    def __init__(self, hps, out_dim):
        super(CharactorVoiceEncoder, self).__init__()
        self.hps = hps
        self.nd_embedding = nn.Embedding(3, out_dim)
        nn.init.zeros_(self.nd_embedding.weight)
        if hps.character_voice_encoder:
            self.fc_layers = nn.Sequential(
                nn.Linear(self.hps.xvector_dim, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, out_dim)
            )
            self.ecapa_tdnn = None ### A pre-trained ECAPA-TDNN network needs to be added here. | wav [wav_len] -> xvector [xvector_dim]

    def forward(self, sample, discourse_scale=False):
        all = '_all' if discourse_scale else ''
        voice_embedding = self.nd_embedding(sample['ndtag' + all])
        if self.hps.character_voice_encoder:
            xvector = torch.stack([torch.stack([self.ecapa_tdnn(w) for w in ws]) for ws in sample['wav' + all]])
            xvector = xvector.detach() if not self.hps.get('tdnn_finetune',False) else xvector
            tdnn_embedding = self.fc_layers(xvector) * (sample['ndtag' + all] - 1).unsqueeze(-1)
            if self.hps.get('xvecter_x', None):
                tdnn_embedding *= self.hps.xvecter_x
            voice_embedding += tdnn_embedding
        voice_embedding = repeat_by_lengths(voice_embedding, sample['len_ling' + all])
        return voice_embedding


class CharactorVoiceEncoder_MelGAN(nn.Module):
    def __init__(self, hps):
        super(CharactorVoiceEncoder_MelGAN, self).__init__()
        self.hps = hps
        self.fc_layers = nn.Sequential(
            nn.Linear(self.hps.xvector_dim, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 96*40)
        )
        self.ecapa_tdnn = None ### A pre-trained ECAPA-TDNN network needs to be added here. | wav [wav_len] -> xvector [xvector_dim]

    def forward(self, sample):
        xvector = torch.stack([torch.stack([self.ecapa_tdnn(w) for w in ws]) for ws in sample['wav']])
        xvector = xvector.detach() if not self.hps.tdnn_finetune else xvector
        xvector = self.fc_layers(xvector) * (sample['ndtag'] - 1).unsqueeze(-1)
        return xvector.reshape(-1,96,40,1)
