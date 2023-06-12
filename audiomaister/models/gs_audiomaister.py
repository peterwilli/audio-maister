import os
import librosa
import torch
import numpy as np
from tqdm import tqdm
from ..tools.file.wav import save_wave
from ..tools.pytorch.pytorch_util import from_log

EPS = 1e-8

class AudioMaister(torch.nn.Module):
    def __init__(self, model):
        super(AudioMaister, self).__init__()
        self._model = model
        self._model.eval()

    def _load_wav_energy(self, path, sample_rate, threshold=0.95):
        wav_10k, _ = librosa.load(path, sr=sample_rate)
        stft = np.log10(np.abs(librosa.stft(wav_10k)) + 1.0)
        fbins = stft.shape[0]
        e_stft = np.sum(stft, axis=1)
        for i in range(e_stft.shape[0]):
            e_stft[-i - 1] = np.sum(e_stft[: -i - 1])
        total = e_stft[-1]
        for i in range(e_stft.shape[0]):
            if e_stft[i] < total * threshold:
                continue
            else:
                break
        return wav_10k, int((sample_rate // 2) * (i / fbins))

    def _load_wav(self, path, sample_rate, threshold=0.95):
        wav_10k, _ = librosa.load(path, sr=sample_rate)
        return wav_10k

    def _amp_to_original_f(self, mel_sp_est, mel_sp_target, cutoff=0.2):
        freq_dim = mel_sp_target.size()[-1]
        mel_sp_est_low, mel_sp_target_low = (
            mel_sp_est[..., 5 : int(freq_dim * cutoff)],
            mel_sp_target[..., 5 : int(freq_dim * cutoff)],
        )
        energy_est, energy_target = torch.mean(mel_sp_est_low, dim=(2, 3)), torch.mean(
            mel_sp_target_low, dim=(2, 3)
        )
        amp_ratio = energy_target / energy_est
        return mel_sp_est * amp_ratio[..., None, None], mel_sp_target

    def _trim_center(self, est, ref):
        diff = np.abs(est.shape[-1] - ref.shape[-1])
        if est.shape[-1] == ref.shape[-1]:
            return est, ref
        elif est.shape[-1] > ref.shape[-1]:
            min_len = min(est.shape[-1], ref.shape[-1])
            est, ref = est[..., int(diff // 2) : -int(diff // 2)], ref
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref
        else:
            min_len = min(est.shape[-1], ref.shape[-1])
            est, ref = est, ref[..., int(diff // 2) : -int(diff // 2)]
            est, ref = est[..., :min_len], ref[..., :min_len]
            return est, ref

    def _pre(self, model, input):
        input = input[None, None, ...]
        sp, _, _ = model.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = model.mel(sp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # return models.to_log(sp), models.to_log(mel_orig)
        return sp, mel_orig

    def remove_higher_frequency(self, wav, ratio=0.95):
        stft = librosa.stft(wav)
        real, img = np.real(stft), np.imag(stft)
        mag = (real**2 + img**2) ** 0.5
        cos, sin = real / (mag + EPS), img / (mag + EPS)
        spec = np.abs(stft)  # [1025,T]
        feature = spec.copy()
        feature = np.log10(feature + EPS)
        feature[feature < 0] = 0
        energy_level = np.sum(feature, axis=1)
        threshold = np.sum(energy_level) * ratio
        curent_level, i = energy_level[0], 0
        while i < energy_level.shape[0] and curent_level < threshold:
            curent_level += energy_level[i + 1, ...]
            i += 1
        spec[i:, ...] = np.zeros_like(spec[i:, ...])
        stft = spec * cos + 1j * spec * sin
        return librosa.istft(stft)

    @torch.no_grad()
    def restore_inmem(self, wav_10k, mode=0, your_vocoder_func=None):
        if mode == 0:
            self._model.eval()
        elif mode == 1:
            self._model.eval()
        elif mode == 2:
            self._model.train()  # More effective on seriously damaged speech
        res = []
        seg_length = 44100 * 128
        for i in tqdm(range(0, wav_10k.shape[0], seg_length), desc="Fixing audio segments..."):
            segment = wav_10k[i:i + seg_length]
            if mode == 1:
                segment = self.remove_higher_frequency(segment)
            sp, mel_noisy = self._pre(self._model, segment)
            out_model = self._model(mel_noisy)
            denoised_mel = from_log(out_model["mel"])
            if your_vocoder_func is None:
                out = self._model.vocoder(denoised_mel)
            else:
                out = your_vocoder_func(denoised_mel)
            # unify energy
            if torch.max(torch.abs(out)) > 1.0:
                out = out / torch.max(torch.abs(out))
                print("Warning: Exceed energy limit,", input)
            # frame alignment
            out, _ = self._trim_center(out, segment)
            res.append(out)
        out = torch.cat(res, -1)
        return out.squeeze(0)

    def restore(self, input, output, mode=0, your_vocoder_func=None):
        wav_10k = self._load_wav(input, sample_rate=44100)
        wav_10k = torch.tensor(wav_10k, device=self._model.device)
        out_wav = self.restore_inmem(
            wav_10k, mode=mode, your_vocoder_func=your_vocoder_func
        )
        save_wave(out_wav.numpy(), fname=output, sample_rate=44100)
