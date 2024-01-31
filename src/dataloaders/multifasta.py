from typing import Dict, Optional, Tuple, Union, cast

import genvarloader as gvl
import numpy as np
import polars as pl
import pytorch_lightning as lit
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
        return seq[..., :-1], seq[..., 1:].astype(np.int64)


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

    def __getitem__(self, index: int) -> Tuple[NDArray, NDArray]:
        chrom, start, end, species = self.bed.row(index)
        batch = {NAME: np.char.upper(self.fastas[species].read(chrom, start, end))}
        return self.tokenize(batch)


class MultiFasta(SequenceDataset):
    _name_ = "multifasta_v2"

    def __init__(
        self,
        file_table: Union[str, pl.DataFrame],
        bed: Union[str, pl.DataFrame],
        batch_size: int,
        num_workers: int = 1,
        limit_fastas: Optional[int] = None,
        *args,
        **kwargs,
    ):
        if isinstance(file_table, str):
            fastas = pl.read_csv(file_table, separator="\t")["fasta"]
        else:
            fastas = file_table["fasta"]
        
        if isinstance(bed, str):
            bed = pl.read_ipc(bed)

        if limit_fastas is not None:
            fastas = fastas.head(limit_fastas)
        
        species = fastas.str.split("/").list.get(-1).str.split(".").list.get(0)
        
        if limit_fastas is not None:
            bed = bed.filter(pl.col('species').is_in(species))
            
        self.beds = bed.partition_by("split", as_dict=True, include_key=False)
        # keep this around in case seqlen warmup used
        self.full_train = self.beds['train']

        self.fastas = {
            s: gvl.Fasta(NAME, f, "N", "dna")
            for s, f in tqdm(
                zip(species, fastas), desc="Init FASTAs", total=len(species)
            )
        }
        self.dl_kwargs = {
            "pin_memory": True,
            "batch_size": batch_size,
            "num_workers": num_workers,
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
    

class SeqLenWarmup(lit.Callback):
    def __init__(self, max_length: int, tokens_per_step: int, init_length: int = 1024):
        if max_length <= init_length:
            raise ValueError("max_length must be greater than init_length")
        
        self.tokens_per_step = tokens_per_step
        
        lengths = 2**np.arange(np.log2(init_length), np.ceil(np.log2(max_length)))
        lengths[-1] = max_length
        self.lengths = lengths
        self.len_idx = 0
        
    def on_fit_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        dm = cast(MultiFasta, trainer.datamodule)  # type: ignore[reportGeneralTypeIssues]
        self.full_train = dm.beds['train']
        
        dm.beds['train'] = self._with_length(self.full_train, self.lengths[self.len_idx], self.tokens_per_step)
    
    def on_train_epoch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        dm = cast(MultiFasta, trainer.datamodule)  # type: ignore[reportGeneralTypeIssues]
        self.len_idx += 1
        
        if self.len_idx == len(self.lengths):
            dm.beds['train'] = self.full_train
        else:
            dm.beds['train'] = self._with_length(self.full_train, self.lengths[self.len_idx], self.tokens_per_step)
    
    @staticmethod
    def _with_length(bed: pl.DataFrame, length: int, n_tokens: int):
        midpt = (pl.col('chromEnd') + pl.col('chromStart')) / 2
        
        return (
            bed
            .filter(pl.col('chromEnd') - pl.col('chromStart') >= length)
            .with_columns(
                chromEnd=(midpt + length / 2).round(),
                chromStart=(midpt - length / 2).round()
            )
            .sample(np.ceil(n_tokens / length), with_replacement=True)
        )