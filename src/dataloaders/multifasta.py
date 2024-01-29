from pathlib import Path
from typing import Dict, Union

import genvarloader as gvl
import numpy as np
import polars as pl
from attrs import define
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.dataloaders.base import SequenceDataset


def _tokenize(
    seq: NDArray,
    tokenize_table: Dict[bytes, int],
    add_eos: bool,
    eos_id: int,
    dtype=np.int32,
):
    length_axis = seq.ndim - 1
    if add_eos:
        shape = seq.shape[:length_axis] + (seq.shape[length_axis] + 1,)
        tokenized = np.empty(shape, dtype=dtype)
        for nuc, id in tokenize_table.items():
            tokenized[..., :-1][seq == nuc] = id
        tokenized[..., -1] = eos_id
    else:
        tokenized = np.empty_like(seq, dtype=dtype)
        for nuc, id in tokenize_table.items():
            tokenized[seq == nuc] = id
    return tokenized


@define
class Tokenize:
    name: str
    tokenize_table: Dict[bytes, int]
    add_eos: bool
    eos_id: int

    def __call__(self, batch: Dict[str, NDArray]):
        seq = _tokenize(
            batch[self.name], self.tokenize_table, self.add_eos, self.eos_id
        )
        batch = {
            self.name: seq[..., :-1],
            "target": seq[..., 1:].astype(np.int64),
        }
        return batch


NAME = "seq"
TOKENIZE = Tokenize(
    name=NAME,
    tokenize_table={b"A": 7, b"C": 8, b"G": 9, b"T": 10, b"N": 11},
    add_eos=True,
    eos_id=1,
)


class TorchMultiFasta(Dataset):
    tokenize = TOKENIZE

    def __init__(self, fastas: Dict[str, gvl.Fasta], bed: pl.DataFrame) -> None:
        self.fastas = fastas
        self.bed = bed

    def __len__(self) -> int:
        return self.bed.height

    def __getitem__(self, index: int) -> Dict[str, NDArray]:
        chrom, start, end, species = self.bed.row(index)
        batch = {NAME: np.char.upper(self.fastas[species].read(chrom, start, end))}
        return self.tokenize(batch)


class MultiFasta(SequenceDataset):
    _name_ = "multifasta_v2"

    def __init__(self, file_table: Union[Path, pl.DataFrame], bed: Union[Path, pl.DataFrame], batch_size: int, num_workers: int = 1):
        if isinstance(file_table, Path):
            fastas = pl.read_csv(file_table, separator="\t")["fasta"]
        else:
            fastas = file_table["fasta"]
        
        if isinstance(bed, Path):
            bed = pl.read_ipc(bed)
        self.beds = bed.partition_by("split", as_dict=True, include_key=False)
        
        species = fastas.str.split("/").list.get(-1).str.split(".").list.get(0)
        
        self.fastas = {
            s: gvl.Fasta(NAME, f, "N", "dna") for s, f in tqdm(zip(species, fastas), desc="Init FASTAs", total=len(species))
        }
        self.dl_kwargs = {
            "pin_memory": True,
            "batch_size": batch_size,
            "num_workers": num_workers
        }

    def setup(self):
        self.train_ds = TorchMultiFasta(self.fastas, self.beds["train"])
        self.val_ds = TorchMultiFasta(self.fastas, self.beds["valid"])
        self.test_ds = TorchMultiFasta(self.fastas, self.beds["test"])

    def train_dataloader(self, **kwargs):
        return DataLoader(self.train_ds, shuffle=True, **self.dl_kwargs)

    def val_dataloader(self, **kwargs):
        return DataLoader(self.val_ds, **self.dl_kwargs)

    def test_dataloader(self, **kwargs):
        return DataLoader(self.test_ds, **self.dl_kwargs)
