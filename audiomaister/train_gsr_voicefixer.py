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
from tools.others.audio_op import normalize_energy_torch
from tools.data_processing import AudioPreprocessing

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

def inspect_dataset(dm):
    from tools.file.wav import save_wave
    ap = AudioPreprocessing()
    for i, batch in enumerate(dm.train_dataloader()):
        sample_training_data_save_path = "training_data_sample"
        if(not os.path.exists(sample_training_data_save_path)):
            os.makedirs(sample_training_data_save_path, exist_ok=True)
        batch, individuals = ap.preprocess_train(batch, return_individual=True)
        for k in batch:
            for i in range(batch[k].size()[0]):
                save_wave(batch[k][i, ...].numpy(), os.path.join(sample_training_data_save_path, f"{i}_{k}.wav"), sample_rate=44100)
                save_wave(batch[k][i, ...].numpy(), os.path.join(sample_training_data_save_path, f"{i}_{k}.wav"), sample_rate=44100)
        for k in individuals:
            for i in range(individuals[k].size()[0]):
                save_wave(individuals[k][i, ...].numpy(), os.path.join(sample_training_data_save_path, f"{i}_{k}.wav"), sample_rate=44100)
                save_wave(individuals[k][i, ...].numpy(), os.path.join(sample_training_data_save_path, f"{i}_{k}.wav"), sample_rate=44100)
        break

def main():
    model = VoiceFixer(hp, channels=1, type_target="vocals")
    # train_dataset = datasets.load_from_disk("./tmp_hf_dataset_train")['train']
    train_dataset = datasets.load_dataset("peterwilli/audio-maister")['train'],
    dm = PreProcessedDataModule(
        train_dataset,
        datasets.load_dataset("peterwilli/audio-maister-val")['val'],
        batch_size=hp["train"]["batch_size"]
    )
    print("Inspecting data")
    inspect_dataset(dm)
    print("Inspecting data done")
    torch.set_float32_matmul_precision("medium")
    logger = TensorBoardLogger(os.path.dirname(hp.model_dir), name=os.path.basename(hp.model_dir))
    checkpoint_callback = ModelCheckpoint(
        filename="{epoch}-{step}-{val/loss:.2f}",
        save_top_k=hp["train"]["save_top_k"],
        monitor="val/loss",
        save_last=True,
        every_n_train_steps=100
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=hp["train"]["max_epoches"],
        detect_anomaly=True,
        num_sanity_val_steps=2,
        sync_batchnorm=True,
        check_val_every_n_epoch=hp["train"]["check_val_every_n_epoch"],
        log_every_n_steps=hp["log"]["log_every_n_steps"],
    )
    dm.setup("fit")
    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()