import numpy as np
import torch

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

from collections import defaultdict

import argparse
import os, shutil

import glob
import yaml
from torch.utils.tensorboard import SummaryWriter
from models import *
from meldataset import build_dataloader


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def merge(values):
    size = max(v.shape[0] for v in values)
    oshape = (len(values), size) + values[0].shape[1:]
    res = values[0].new(*oshape).fill_(0)
    for i, v in enumerate(values):
        res[i, : len(v)] = v
    return res


def to_gpu(x):
    if isinstance(x, torch.Tensor):
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
        return torch.autograd.Variable(x)
    elif isinstance(x, list) and isinstance(x[0], list) and isinstance(x[0][0], torch.Tensor):  # sample['wav']
        return [list(map(to_gpu, xx)) for xx in x]
    else:
        return x


def save_checkpoint(model, optimizers, epoch, global_step, hps, filepath):
    print(f"Saving model and optimizer state at epoch {epoch} to {filepath}")
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'cuda_rng_state_all': torch.cuda.get_rng_state_all(),
        'random_rng_state': torch.random.get_rng_state(),
        'config': hps,
        'state_dict': model.state_dict(),
    }
    for i in range(len(optimizers)):
        checkpoint[f'optimizer_{i}'] = optimizers[i].state_dict()
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizers, filepath):
    print('restoring from %s' % filepath)
    checkpoint = torch.load(filepath, map_location='cpu')
    state_dict = checkpoint['state_dict']
    epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step']

    print(model.load_state_dict(state_dict, strict=False))
    for i in range(len(optimizers)):
        print(optimizers[i].load_state_dict(checkpoint[f'optimizer_{i}']))

    return epoch, global_step


def load_last_checkpoint(model, optimizers, ckpt_dir):
    files = list(filter(os.path.isfile, glob.glob(ckpt_dir + os.sep + "*")))
    files.sort(key=lambda x: os.path.getmtime(x))
    epoch, global_step = 0, 0
    if len(files) > 0:
        epoch, global_step = load_checkpoint(model, optimizers, files[-1])
    return epoch, global_step


def validate(model, valset, writer, epoch):
    model.eval()
    with torch.no_grad():
        print(f'validating!')
        total = defaultdict(lambda: 0)
        for i, batch in enumerate(valset):
            for key in batch:
                batch[key] = to_gpu(batch[key])
            _, watches = model(batch)
            for k in sorted(watches.keys()):
                total[k] += watches[k] if k in ['sample_num'] else (watches[k] * watches['sample_num'])
        for k in sorted(total.keys()):
            writer.add_scalar(f'val_loss_mean/{k}', float(total[k]) / float(total['sample_num']), epoch)
        print(', '.join([f'{k}: {float(total[k]) / float(total["sample_num"]):.5f}' for k in sorted(total.keys())]))
    model.train()


def adjust_learning_rate_warmup_sqrt(global_step, optimizer, lr, init_lr=1e-7, warmup_steps=4000):
    if global_step < warmup_steps:
        cur_lr = init_lr + (lr - init_lr) * global_step / warmup_steps
    else:
        cur_lr = lr * (global_step / warmup_steps) ** -0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    hps = AttrDict(yaml.safe_load(open(args.config)))

    if 'test' in hps.mode:
        return test_main(hps)

    hps.model_dir = os.path.join(hps.model_dir, hps.mode)
    hps.log_dir = os.path.join(hps.model_dir, 'tensorboards')
    hps.ckpt_dir = os.path.join(hps.model_dir, 'ckpt')
    os.makedirs(hps.log_dir, exist_ok=True)
    os.makedirs(hps.ckpt_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(hps.model_dir, 'hps.yml'))

    model = ProsodyPredictorModel(hps) if hps.mode in ['prosody_code', 'duration'] else SpectrumPredictorModel(hps)
    model.cuda()

    generator_params = []
    generator_params_name = []
    if hps.psd_gan or hps.mel_gan:
        generator_params_gan = []
        generator_params_gan_name = []
    for name, var in model.named_parameters():
        if var.requires_grad:
            if 'discriminator' in name:
                generator_params_gan.append(var)
                generator_params_gan_name.append(name)
            else:
                generator_params.append(var)
                generator_params_name.append(name)

    optimizer = torch.optim.Adam(generator_params, lr=1e-7, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-08)
    optimizers = [optimizer]
    if hps.psd_gan or hps.mel_gan:
        optimizer_gan = torch.optim.Adam(generator_params_gan, lr=1e-7, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-08)
        optimizers.append(optimizer_gan)

    start_epoch, global_step = load_last_checkpoint(model, optimizers, hps.ckpt_dir)
    train_dataset = build_dataloader(hps, False)
    val_dataset = build_dataloader(hps, True)
    writer = SummaryWriter(hps.log_dir)
    model.train()
    for epoch in range(start_epoch, hps.epochs):
        hps.tdnn_finetune = epoch >= hps.tdnn_finetune_epoch
        for _, batch in enumerate(train_dataset):
            for key in batch:
                batch[key] = to_gpu(batch[key])
            adjust_learning_rate_warmup_sqrt(global_step, optimizer, hps.lr)
            model.zero_grad()
            loss, watches = model(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], hps.grad_clip_thresh)
            writer.add_scalar(f'train_watches/grad_norm', grad_norm, global_step)
            optimizer.step()

            if hps.psd_gan or hps.mel_gan:
                adjust_learning_rate_warmup_sqrt(global_step, optimizer_gan, hps.lr)
                model.zero_grad()
                loss, watches_gan = model(batch, d_loss=True)
                watches.update(watches_gan)
                loss.backward()
                grad_norm_gan = torch.nn.utils.clip_grad_norm_(optimizer_gan.param_groups[0]['params'], hps.grad_clip_thresh)
                writer.add_scalar(f'train_watches/grad_norm_gan', grad_norm_gan, global_step)
                optimizer_gan.step()

            print(f"epoch {epoch}, step {global_step % len(train_dataset)}/{len(train_dataset)}, global_step {global_step}, lr: {optimizer.param_groups[0]['lr']:.2e}, {', '.join([f'{k}: {watches[k]: .5f}' for k in  sorted(watches.keys())])}")
            for k in sorted(watches.keys()):
                writer.add_scalar(f'train_loss/{k}', watches[k], global_step)
            writer.add_scalar(f'train_watches/learning_rate', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1

        if epoch % hps.save_freq == 0:
            validate(model, val_dataset, writer, epoch)
            checkpoint_path = os.path.join(hps.ckpt_dir, f"checkpoint_{int(epoch)}")
            save_checkpoint(model, optimizers, epoch, global_step, hps, checkpoint_path)
    writer.close()

def test_main(hps):
    from scipy.io.wavfile import write

    if hps.mode in ['test_all']:
        print('Genarating duration!')
        dur_hps = AttrDict(yaml.safe_load(open(f'{hps.model_dir}/duration/hps.yml').read()))
        for key in ['ref_wav', 'xvecter_x', 'val_data']:
            dur_hps[key] = hps[key]
        dur_model = ProsodyPredictorModel(dur_hps).cuda().eval()
        load_last_checkpoint(dur_model, [], f'{hps.model_dir}/duration/ckpt/')

        fn2dur = {}
        dur_val_dataset = build_dataloader(dur_hps, True)
        with torch.no_grad():
            for sample in dur_val_dataset:
                for key in sample:
                    sample[key] = to_gpu(sample[key])
                output = dur_model.infer(sample)
                for i, fn in enumerate(sample['filename']):
                    fn2dur[fn] = output['dur'][i].detach()
        del dur_model
        del dur_val_dataset

    if hps.mode in ['test_all']:
        print('Genarating prosody code!')
        psd_hps = AttrDict(yaml.safe_load(open(f'{hps.model_dir}/prosody_code/hps.yml').read()))
        for key in ['ref_wav', 'xvecter_x', 'val_data']:
            psd_hps[key] = hps[key]
        psd_model = ProsodyPredictorModel(psd_hps).cuda().eval()
        load_last_checkpoint(psd_model, [], f'{hps.model_dir}/duration/prosody_code/')

        fn2psd = {}
        psd_val_dataset = build_dataloader(psd_hps, True)
        with torch.no_grad():
            for sample in psd_val_dataset:
                for key in sample:
                    sample[key] = to_gpu(sample[key])
                output = psd_model.infer(sample)
                for i, fn in enumerate(sample['filename']):
                    fn2psd[fn] = output['psd_code'][i].detach()
        del psd_model
        del psd_val_dataset

    if hps.mode in ['test_all', 'test_extract_prosody_code']:
        
        vocoder = None  ### A vocoder needs to be added here. | mel [batch_size, mel_len, mel_dim] -> wav [batch_size, wav_len]
        SR = 24000

        mel_hps = AttrDict(yaml.safe_load(open(f'{hps.model_dir}/spectrum/hps.yml').read()))
        for key in ['ref_wav', 'xvecter_x', 'val_data']:
            mel_hps[key] = hps[key]
        mel_model = SpectrumPredictorModel(mel_hps).cuda().eval()
        load_last_checkpoint(mel_model, [], f'{hps.model_dir}/duration/spectrum/')
        
        with torch.no_grad():
            if hps.mode == 'test_extract_prosody_code':
                print('Extracting prosody code!')
                mel_all_dataset = build_dataloader(mel_hps, True, all_data=True)
                os.makedirs(f'{hps.data_path}/psd_code', exist_ok=True)
                for sample in mel_all_dataset:
                    for key in sample:
                        sample[key] = to_gpu(sample[key])
                    output = mel_model.infer(sample, extract_prosody_code=True)
                    for fn, code in zip(sample['filename'], output['code']):
                        np.save(f'{hps.data_path}/psd_code/{fn}.npy', code.detach().cpu().numpy())
                        print(f'{hps.data_path}/psd_code/{fn}.npy prosody code saved!')

            else:
                mel_val_dataset = build_dataloader(mel_hps, True)
                for sample in mel_val_dataset:
                    for key in sample:
                        sample[key] = to_gpu(sample[key])
                    if hps.mode == 'test_all':
                        sample['pd'] = merge([fn2dur[fn] for fn in sample['filename']])
                        sample['pz'] = merge([fn2psd[fn] for fn in sample['filename']])
                    output = mel_model.infer(sample)
                    for i in range(len(sample['filename'])):
                        breakpoint()
                        audio_float = vocoder(output['mel'][i].unsqueeze(0).transpose(1, 2)).squeeze().cpu().numpy()
                        write(hps.test_out_dir + '/' + output['filename'][i][7] + '_' + output['filename'][i] + '_' + hps.jobname + '.wav', SR, (audio_float * 32768).astype('int16'))


if __name__ == '__main__':
    main()
