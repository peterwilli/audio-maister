import torch.utils
from ..tools.pytorch.mel_scale import MelScale
from ..tools.data_processing import AudioPreprocessing
from ..tools.pytorch.modules.fDomainHelper import FDomainHelper
from ..tools.pytorch.losses import get_loss_function
from ..tools.pytorch.pytorch_util import to_log
from .components.vocoder.vocoder import Vocoder
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data
import lightning as L

class BN_GRU(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,layer=1, bidirectional=False, batchnorm=True, dropout=0.0):
        super(BN_GRU, self).__init__()
        self.batchnorm = batchnorm
        if(batchnorm):self.bn = nn.BatchNorm2d(1)
        self.gru = torch.nn.GRU(input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=layer,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self,inputs):
        # (batch, 1, seq, feature)
        if(self.batchnorm):inputs = self.bn(inputs)
        out,_ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)

class Generator(torch.nn.Module):
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.hp = hp
        if(self.hp["task"]["gsr"]["gsr_model"]["voicefixer"]["unet"]):
            from .components.unet import UNetResComplex_100Mb
            self.analysis_module = UNetResComplex_100Mb(channels=hp["model"]["channels_in"])
        elif(self.hp["task"]["gsr"]["gsr_model"]["voicefixer"]["unet_small"]):
            from .components.unet_small import UNetResComplex_100Mb
            self.analysis_module = UNetResComplex_100Mb(channels=hp["model"]["channels_in"])
        elif(self.hp["task"]["gsr"]["gsr_model"]["voicefixer"]["bi_gru"]):
            n_mel = hp["model"]["mel_freq_bins"]
            self.analysis_module = torch.nn.Sequential(
                    torch.nn.BatchNorm2d(1),
                    torch.nn.Linear(n_mel, n_mel * 2),
                    BN_GRU(input_dim=n_mel*2, hidden_dim=n_mel*2, bidirectional=True, layer=2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_mel*4, n_mel*2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_mel*2, n_mel),
                )
        elif(self.hp["task"]["gsr"]["gsr_model"]["voicefixer"]["dtorch.nn"]):
            n_mel = hp["model"]["mel_freq_bins"]
            self.analysis_module = torch.nn.Sequential(
                    torch.nn.Linear(n_mel, n_mel * 2),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(1),
                    torch.nn.Linear(n_mel * 2, n_mel * 4),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(1),
                    torch.nn.Linear(n_mel * 4, n_mel * 8),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(1),
                    torch.nn.Linear(n_mel * 8, n_mel * 4),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(1),
                    torch.nn.Linear(n_mel * 4, n_mel * 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_mel * 2, n_mel),
                )
        else:
            pass # todo warning

    def forward(self, mel_orig):
        out = self.analysis_module(to_log(mel_orig))
        if(type(out) == type({})):
            out = out['mel']
        mel = out + to_log(mel_orig)
        return {'mel': mel}

class VoiceFixer(L.LightningModule):
    def __init__(self, hp, channels, type_target):
        super(VoiceFixer, self).__init__()

        self.val_step = 0
        self.lr = hp["train"]["learning_rate"]
        self.gamma = hp["train"]["lr_decay"]
        self.batch_size = hp["train"]["batch_size"]
        self.input_segment_length = hp["train"]["input_segment_length"]
        self.sampling_rate = hp["data"]["sampling_rate"]
        self.check_val_every_n_epoch = hp["train"]["check_val_every_n_epoch"]
        self.warmup_steps = hp["train"]["warmup_steps"]
        self.reduce_lr_every_n_steps = hp["train"]["reduce_lr_every_n_steps"]
        self.ap = AudioPreprocessing()
        self.save_hyperparameters()
        self.channels = channels
        self.vocoder = Vocoder(sample_rate=44100)
        
        # self.hparams['channels'] = 2
        self.simelspecloss = get_loss_function(loss_type="simelspec")
        self.l1loss = get_loss_function(loss_type="l1")
        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

        self.f_helper = FDomainHelper(
            window_size=hp["model"]["window_size"],
            hop_size=hp["model"]["hop_size"],
            center=True,
            pad_mode=hp["model"]["pad_mode"],
            window=hp["model"]["window"],
            freeze_parameters=True,
        )

        self.mel_freq_bins = hp["model"]["mel_freq_bins"]
        self.mel = MelScale(n_mels=self.mel_freq_bins,
                            sample_rate=self.sampling_rate,
                            n_stft=hp["model"]["window_size"] // 2 + 1)

        # masking
        self.generator = Generator(hp)
        self.hp = hp

    def pre(self, input):
        sp, _, _ = self.f_helper.wav_to_spectrogram_phase(input)
        mel_orig = self.mel(sp.permute(0,1,3,2)).permute(0,1,3,2)
        return sp, mel_orig

    def forward(self, mel_orig):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        return self.generator(mel_orig)

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam([{'params': self.generator.parameters()}],
                                       lr=self.lr, amsgrad=True, betas=(self.hp["train"]["betas"][0], self.hp["train"]["betas"][1]))

        steps = self.trainer.estimated_stepping_batches
        scheduler_g = {
            'scheduler': CosineAnnealingLR(optimizer_g, T_max=steps, eta_min=0),
            'interval': 'step',
            'frequency': 1
        }
        return ([optimizer_g], [scheduler_g])

    def preprocess(self, batch, train=False, cutoff=None):
        if not train:
            if(cutoff is None):
                low_quality = batch["noisy"]
                vocals = batch["vocals"]
                vocals, LR_noisy = vocals, low_quality
                return vocals, vocals, LR_noisy
            else:
                LR_noisy = batch["noisy"+"LR"+"_"+str(cutoff)]
                LR = batch["vocals" + "LR" + "_" + str(cutoff)]
                vocals = batch["vocals"]
                return vocals, LR, LR_noisy

    def training_step(self, batch):
        with torch.no_grad():
            batch = self.ap.preprocess_train(batch)
        _, mel_target = self.pre(batch['target'])
        _, mel_low_quality = self.pre(batch['low_quality'])
        generated = self.forward(mel_low_quality)
        target_loss = self.l1loss(generated['mel'], to_log(mel_target))
        self.log("train/target_loss", target_loss, on_step=True, on_epoch=False, logger=True, sync_dist=True, prog_bar=True)
        return target_loss

    def validation_step(self, batch, batch_idx):
        vocal, _, low_quality  = self.preprocess(batch, train=False)
        batch_size = batch['noisy'].shape[0]
        fname = [f"{batch_idx}_{i}" for i in range(batch_size)]
        _, mel_target = self.pre(vocal)
        _, mel_low_quality = self.pre(low_quality)
        estimation = self(mel_low_quality)['mel']
        val_loss = self.l1loss(estimation, to_log(mel_target))
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True, prog_bar=True, batch_size=1)
        return val_loss