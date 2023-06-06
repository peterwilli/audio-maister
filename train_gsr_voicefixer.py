import git
import sys

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from pynvml import *
from dataloaders.data_module import SrRandSampleRate
from tools.callbacks.base import *
from tools.callbacks.verbose import *
import lightning as L
import torch
import tools.utils
from tools.dsp.lowpass import *
from models.gsr_voicefixer import VoiceFixer

if (not torch.cuda.is_available()):
    raise RuntimeError("Hi bro, you need GPUs to run this program.")

hp, parser = tools.utils.get_hparams()

assert hp["data"]["sampling_rate"] == 44100
hp["root"]=git_root

for k in hp["data"]["train_dataset"].keys():
    for v in hp["data"]["train_dataset"][k].keys():
        hp["data"]["train_dataset"][k][v] = os.path.join(hp["root"], hp["data"]["train_dataset"][k][v])

for k in hp["data"]["val_dataset"].keys():
    for v in hp["data"]["val_dataset"][k].keys():
        hp["data"]["val_dataset"][k][v] = os.path.join(hp["root"], hp["data"]["val_dataset"][k][v])

hp["augment"]["params"]["rir_root"] = os.path.join(hp["root"], hp["augment"]["params"]["rir_root"])

args = parser.parse_args()

nvmlInit()
gpu_nums = int(nvmlDeviceGetCount())
accelerator = 'ddp'
distributed = gpu_nums > 1
model = VoiceFixer(hp, channels=1, type_target="vocals")
# print(model)
dm = SrRandSampleRate(hp, distributed)
torch.set_float32_matmul_precision('medium')
trainer = L.Trainer(accelerator="gpu", devices=1,
                                     callbacks = [initLogDir(hp, current_dir=os.getcwd())],
                                     strategy="ddp" if (torch.cuda.is_available()) else None,
                                     max_epochs=hp["train"]["max_epoches"],
                                     detect_anomaly=True,
                                     num_sanity_val_steps=2,
                                     sync_batchnorm=True,
                                     check_val_every_n_epoch=hp["train"]["check_val_every_n_epoch"],
                                     log_every_n_steps=hp["log"]["log_every_n_steps"])
dm.setup('fit')
trainer.fit(model, datamodule=dm)
