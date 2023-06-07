import git
import sys

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from pynvml import *
from dataloaders.data_module import SrRandSampleRate
from dataloaders.preprocessed_datamodule import PreProcessedDataModule
from tools.callbacks.base import *
from tools.callbacks.verbose import *
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor
import torch
import tools.utils
from tools.dsp.lowpass import *
from models.gsr_voicefixer import VoiceFixer
import datasets

if not torch.cuda.is_available():
    raise RuntimeError("Hi bro, you need GPUs to run this program.")

hp, parser = tools.utils.get_hparams()

assert hp["data"]["sampling_rate"] == 44100
hp["root"] = git_root

for k in hp["data"]["train_dataset"].keys():
    for v in hp["data"]["train_dataset"][k].keys():
        hp["data"]["train_dataset"][k][v] = os.path.join(
            hp["root"], hp["data"]["train_dataset"][k][v]
        )

for k in hp["data"]["val_dataset"].keys():
    for v in hp["data"]["val_dataset"][k].keys():
        hp["data"]["val_dataset"][k][v] = os.path.join(
            hp["root"], hp["data"]["val_dataset"][k][v]
        )

hp["augment"]["params"]["rir_root"] = os.path.join(
    hp["root"], hp["augment"]["params"]["rir_root"]
)

args = parser.parse_args()

nvmlInit()
gpu_nums = int(nvmlDeviceGetCount())
model = VoiceFixer(hp, channels=1, type_target="vocals")
dm = PreProcessedDataModule(
    # datasets.load_dataset("peterwilli/audio-maister-test")['train'],
    datasets.load_from_disk("./tmp_hf_dataset_train")['train'],
    datasets.load_dataset("peterwilli/audio-maister-val")['val'],
    batch_size=hp["train"]["batch_size"]
)
torch.set_float32_matmul_precision("medium")
logger = TensorBoardLogger(os.path.dirname(hp.model_dir), name=os.path.basename(hp.model_dir))
checkpoint_callback = ModelCheckpoint(
    filename="{epoch}-{step}-{val_l:.2f}",
    save_top_k=hp["train"]["save_top_k"],
    monitor="targ_l",
    every_n_train_steps=100
)
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    logger=logger,
    callbacks=[initLogDir(hp, current_dir=os.getcwd()), checkpoint_callback, lr_monitor],
    max_epochs=hp["train"]["max_epoches"],
    detect_anomaly=True,
    num_sanity_val_steps=2,
    sync_batchnorm=True,
    check_val_every_n_epoch=hp["train"]["check_val_every_n_epoch"],
    log_every_n_steps=hp["log"]["log_every_n_steps"],
)
dm.setup("fit")
trainer.fit(model, datamodule=dm)
