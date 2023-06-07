from datasets import Dataset, DatasetDict, Features, Audio
from dataloaders.data_module import SrRandSampleRate
import tools.utils
import git
import sys
import os
import gc
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
import io
from tools.file.flac import save_flac

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)

def init_hp():
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
    return hp, parser

def numpy_to_flac(arr):
    flac_bytes = io.BytesIO()
    save_flac(arr, flac_bytes)
    return flac_bytes.getvalue()

def data_gen(loader, keys):
    for data in loader:
        batch_size = data[keys[0]].shape[0]
        for batch_idx in range(batch_size):
            yield {
                k: numpy_to_flac(data[k][batch_idx, ...].numpy()) for k in keys
            }

def write_parquet_from_generator(gen, output_path):
    writer = None
    for data in tqdm(gen, desc="Caching data to PyArrow"):
        table = pa.Table.from_pydict(data)
        if writer is None:
            print(table.schema)
            writer = pq.ParquetWriter(output_path, table.schema, compression='gzip')
        writer.write_table(table)
    writer.close()
    
def main():
    hp, parser = init_hp()
    distributed = False
    dm = SrRandSampleRate(hp, distributed)
    dm.setup('fit')
    print(f"Total entries: {len(dm.train)}")
    keys = ['vocals', 'vocals_LR', 'vocals_aug_LR', 'noise_LR']
    dataset = Dataset.from_generator(
        data_gen,
        cache_dir=".dataset_gen_cache",
        gen_kwargs={ 'loader': dm.train_dataloader(), 'keys': keys },
        features=Features({k: Audio() for k in keys}))
    dataset = DatasetDict(train=dataset)
    # dataset.save_to_disk("tmp_hf_dataset_train")
    dataset.push_to_hub("peterwilli/audio-maister-test")

    keys = ['vocals', 'noisy']
    dataset = Dataset.from_generator(
        data_gen,
        cache_dir=".dataset_gen_cache",
        gen_kwargs={ 'loader': dm.val_dataloader(), 'keys': keys },
        features=Features({k: Audio() for k in keys}))
    dataset = DatasetDict(val=dataset)
    dataset.push_to_hub("peterwilli/audio-maister-val")

if __name__ == "__main__":
    main()