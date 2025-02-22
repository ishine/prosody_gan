import torch
import torch.nn as nn
from modules import *
from gan_for_prosody import MultiSpectroGAN
from gan_for_mel import Discriminator2d
from conformer import ConformerEncoder

class ProsodyPredictor(nn.Module):
    '''Encoder'''

    def __init__(self, hps, output_dim, layer_num=4):
        super(ProsodyPredictor, self).__init__()
        self.hps = hps
        hidden_dim = 256
        self.encoder = ConformerEncoder(hps, input_dim=hps.input_dim + hps.bert_dim + 2, layer_num=layer_num)
        self.charactor_encoder = CharactorVoiceEncoder(hps, hidden_dim)
        self.decoder = ConformerEncoder(hps, input_dim=hidden_dim, layer_num=layer_num)
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        self.bert = None ### A pre-trained BERT network needs to be added here. | char_id [batch_size, char_len] -> BERT_embedding [batch_size, char_len, bert_dim]

    def forward(self, sample):
        if self.hps.without_bert:
            bert = torch.zeros(list(sample['textid_all'].shape) + [self.hps.bert_dim]).to(sample['ling_all'])
        else:
            bert = self.bert.sentence_encoder(sample['textid_all'])
        ling_with_bert = torch.cat(
            [
                sample['ling_all'],
                repeat_by_lengths(bert, sample['char_len_all']),
                repeat_by_lengths(sample['sindex_all'].unsqueeze(-1), sample['len_ling_all']),
                repeat_by_lengths(sample['out_paragraph_all'].unsqueeze(-1), sample['len_ling_all']),
            ],
            -1,
        )
        h = self.encoder(ling_with_bert, ~sample['ling_all_mask'].bool())
        h += self.charactor_encoder(sample, True)
        h = self.decoder(h, ~sample['ling_all_mask'].bool())
        h = self.output_linear(h)
        return h


class ProsodyPredictorModel(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.mseloss = nn.MSELoss()
        self.maeloss = nn.L1Loss()
        self.predictor = ProsodyPredictor(hps, 4 if hps.mode == 'prosody_code' else 1, 2 if hps.mode == 'duration' else 4)
        if hps.psd_gan:
            self.discriminator = MultiSpectroGAN(hps)
            if not hps.psd_gan_without_condition:
                self.discriminator_condition = ProsodyPredictor(hps, 256)

    def forward(self, sample, d_loss=False):
        watches = {}
        pred_z = self.predictor(sample)
        if self.hps.mode == 'prosody_code':
            watches['sample_num'] = sample['ling_all_mask'].sum()
            if self.hps.psd_gan:
                condition = torch.rand(pred_z.shape[0], pred_z.shape[1], 256).to(pred_z) if self.hps.psd_gan_without_condition else self.discriminator_condition(sample)
                watches.update(self.discriminator(sample['psd_code_all'], pred_z, condition, sample['ling_all_mask'].bool(), d_loss))
                if d_loss:
                    return watches['D_loss'], watches
                else:
                    watches['recon_loss'] = self.mseloss(pred_z[sample['ling_all_mask'].bool()], sample['psd_code_all'][sample['ling_all_mask'].bool()])
                    return self.hps.rclossw * watches['recon_loss'] + self.hps.gnlossw * watches['G_loss'] + self.hps.fmlossw * watches['FM_loss'], watches
            else:
                watches['recon_loss'] = self.mseloss(pred_z[sample['ling_all_mask'].bool()], sample['psd_code_all'][sample['ling_all_mask'].bool()])
                return watches['recon_loss'], watches

        if self.hps.mode == 'duration':
            watches['sample_num'] = sample['ling_all_mask'].sum()
            watches['recon_loss'] = self.mseloss(pred_z[sample['ling_all_mask'].bool()], torch.log(sample['dur_all'] + 1)[sample['ling_all_mask'].bool()].unsqueeze(-1))
            return watches['recon_loss'], watches

    def infer(self, sample, randomid=False):
        output = {}
        output['filename'] = sample['filename']
        if self.hps.mode == 'duration':
            pred_dur = self.predictor(sample)
            pred_dur = select_middle_by_lengths(pred_dur, sample['len_ling_all']).squeeze(-1)
            pred_dur = torch.round((torch.exp(pred_dur) - 1))
            pred_dur[pred_dur < 0] = 0
            output['dur'] = [x[sample['ling_mask'][i].bool()] for i, x in enumerate(pred_dur)]
        if self.hps.mode == 'prosody_code':
            pred_psd = self.predictor(sample)
            pred_psd = select_middle_by_lengths(pred_psd, sample['len_ling_all']).squeeze(-1)
            output['psd_code'] = [x[sample['ling_mask'][i].bool()] for i, x in enumerate(pred_psd)]
        return output


class SpectrumPredictorModel(nn.Module):
    def __init__(self, hps, hidden_dim=256):
        super().__init__()
        self.maeloss = nn.L1Loss()
        self.mseloss = nn.MSELoss()
        self.CEloss = nn.CrossEntropyLoss()
        self.hps = hps
        self.charactor_encoder = CharactorVoiceEncoder(hps, hidden_dim)
        self.encoder = ConformerEncoder(hps, input_dim=898, layer_num=4)
        self.length_regulator = LengthRegulator()
        self.decoder = ConformerEncoder(hps, layer_num=4)
        self.mel_linear = nn.Linear(hidden_dim, hps.mel_channels)
        self.prosody_extractor = ReferenceEncoder(hps)
        self.prosody_extractor_linear = nn.Linear(hidden_dim + hps.psdcode_dim, hidden_dim)
        if hps.mel_gan:
            self.discriminator = Discriminator2d(hps)

    def forward(self, sample, d_loss=False):
        # This module is not for the forward function, but for the computation of the loss.
        watches = {}
        total_loss = 0
        watches['sample_num'] = sample['ling_mask'].sum()
        mel_mask = sample['mel_mask'].bool()
        encoder_output = self.encoder(sample['ling'], ~sample['ling_mask'].bool())
        encoder_output += self.charactor_encoder(sample)
        psd_code = self.prosody_extractor(sample)
        x = torch.cat([encoder_output, psd_code], 2)
        x = self.prosody_extractor_linear(x)
        x, _ = self.length_regulator(x, sample['dur'], None,  ~sample['ling_mask'].bool())
        decoder_output = self.decoder(x, ~sample['mel_mask'].bool())
        pred_mel = self.mel_linear(decoder_output)

        if self.hps.mel_gan and d_loss:
            watches.update(self.discriminator.get_dloss(sample, pred_mel))
            return watches['D_loss'], watches

        mel = pred_mel[mel_mask]
        mel_target = sample['mel'][mel_mask]
        
        watches['mel_loss'] = self.maeloss(mel, mel_target)
        total_loss += watches['mel_loss']

        if self.hps.mel_gan and not d_loss:
            watches.update(self.discriminator.get_gloss(sample, pred_mel))
            total_loss += watches['G_loss']

        watches['sample_num'] = sample['mel_mask'].sum()
        watches['total_loss'] = total_loss
        return total_loss, watches

    def infer(self, sample, extract_prosody_code=False, real_code=False, real_dur=True):
        output = {}
        output['filename'] = sample['filename']

        if extract_prosody_code:
            code = self.prosody_extractor(sample)
            output['code'] = [code[i][: int(sample['ling_mask'].sum(1)[i])] for i in range(len(code))]
        else:
            encoder_output = self.encoder(sample['ling'], ~sample['ling_mask'].bool())
            encoder_output += self.charactor_encoder(sample)

            psd_code = self.prosody_extractor(sample) if real_code else sample['pz']
            encoder_output = torch.cat([encoder_output, psd_code], 2)
            x = self.prosody_extractor_linear(encoder_output)
            dur = sample['dur'] if real_dur else sample['pd']
            x, mel_len = self.length_regulator(x, dur , None,  ~sample['ling_mask'].bool())
            mel_mask = get_mask_from_lengths(mel_len)
            decoder_output = self.decoder(x, mel_mask)
            mel_output = self.mel_linear(decoder_output)
            output['mel'] = [mel_output[i][: int(dur[i].sum())] for i in range(len(mel_output))]
        return output
