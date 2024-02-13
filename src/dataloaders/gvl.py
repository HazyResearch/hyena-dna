from copy import copy
from itertools import chain
from pathlib import Path
from typing import Dict, Hashable, Iterator, List, Literal, Optional, Tuple, Union

import genvarloader as gvl
import numpy as np
import polars as pl
from attrs import define
from genvarloader.util import random_chain, read_bedlike
from more_itertools import interleave_longest
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing_extensions import assert_never

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

    def __call__(self, batch: Dict[Hashable, NDArray]):
        seq = _tokenize(
            batch[self.name], self.tokenize_table, self.add_eos, self.eos_id
        )
        batch = {
            self.name: seq[..., :-1],
            "target": seq[..., 1:].astype(np.int64),
        }
        return batch


class Compose:
    mode: Literal["random", "interleave"]

    def __init__(self, *dataloaders: DataLoader, mode: Literal["random", "interleave"]):
        self.dataloader = dataloaders
        self.mode = mode

    def __iter__(self) -> Iterator[DataLoader]:
        if self.mode == "random":
            return iter(random_chain(*self.dataloader))
        elif self.mode == "interleave":
            return iter(interleave_longest(*self.dataloader))
        else:
            assert_never(self.mode)

    def __len__(self):
        return sum(len(it) for it in self.dataloader)


NAME = "seq"
TOKENIZER = Tokenize(
    name=NAME,
    tokenize_table={b"A": 7, b"C": 8, b"G": 9, b"T": 10, b"N": 11},
    add_eos=True,
    eos_id=1,
)


class Fasta(SequenceDataset):
    _name_ = "fasta"

    def __init__(
        self,
        fasta: Union[str, Path],
        bed: Union[str, Path, pl.DataFrame],
        max_length: int,
        batch_size: int,
        max_memory_gb: float,
        *args,
        **kwargs,
    ):
        self.fasta = gvl.Fasta("seq", fasta, "N", "dna")
        self.name = NAME

        self._max_length = max_length
        self._batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.transform = TOKENIZER

        if isinstance(bed, (str, Path)):
            self.bed = read_bedlike(bed)
        else:
            self.bed = bed
        if "name" not in self.bed:
            raise RuntimeError("Need name column to use for identifying splits.")
        self.bed = self.bed.rename({"name": "split"})
        self._sample_bed = None
        self.setup()

    def setup(self):
        partitions = self.bed.partition_by("split", as_dict=True)
        self.train_bed = partitions["train"]
        self.val_bed = partitions["valid"]
        self.test_bed = partitions["test"]

        self.init_datasets()

    @property
    def max_length(self):
        return self._max_length

    @max_length.setter
    def max_length(self, value: int):
        self._max_length = value
        self.init_datasets()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value
        self.init_datasets()

    def set(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.setup()

    def _gvl(self, split: str):
        if split == "fit":
            if self._sample_bed is None:
                bed = self.train_bed
            else:
                bed = self.train_bed.sample(self._sample_bed)
            shuffle = True
        elif split == "val":
            bed = self.val_bed
            shuffle = False
        elif split == "test":
            bed = self.test_bed
            shuffle = False
        else:
            raise ValueError(f"Invalid split: {split}")

        gvloader = gvl.GVL(
            self.fasta,
            bed=bed,
            fixed_length=self.max_length,
            batch_size=self.batch_size,
            max_memory_gb=self.max_memory_gb,
            shuffle=shuffle,
            transform=self.transform,
            return_tuples=[self.name, "target"],
        )
        return gvloader

    def init_datasets(self):
        self.train_dataset = self._gvl("fit").torch_dataset()
        self.val_dataset = self._gvl("val").torch_dataset()
        self.test_dataset = self._gvl("test").torch_dataset()

    def train_dataloader(self, **kwargs):
        return self.train_dataset.torch_dataloader()

    def val_dataloader(self, **kwargs):
        return self.val_dataset.torch_dataloader()

    def test_dataloader(self, **kwargs):
        return self.test_dataset.torch_dataloader()


class MultiFasta(SequenceDataset):
    _name_ = "multifasta"

    def __init__(
        self,
        files: Union[str, Path, pl.DataFrame],
        max_length: int,
        batch_size: int,
        max_memory_gb: float,
        bed: Optional[Union[str, Path, pl.DataFrame]] = None,
        seqlen_warmup: Optional[List[Tuple[int, int]]] = None,
        *args,
        **kwargs,
    ):
        self._max_length = max_length
        self._batch_size = batch_size
        self.max_memory_gb = max_memory_gb

        if not isinstance(files, pl.DataFrame):
            files = Path(files)
            separator = "," if files.suffix == ".csv" else "\t"
            files = pl.read_csv(files, separator=separator)
            
        if bed is not None:
            if isinstance(bed, (str, Path)):
                bed = pl.read_ipc(bed)
            beds = bed.partition_by("species", include_key=False)
            self.fastas = [
                Fasta(
                    fasta,
                    bed,
                    self.max_length,
                    self.batch_size,
                    self.max_memory_gb,
                )
                for (fasta, _), bed in tqdm(zip(files.iter_rows(), beds), total=files.height, desc="Initializing fastas")
            ]
        else:
            self.fastas = [
                Fasta(
                    fasta,
                    bed,
                    self.max_length,
                    self.batch_size,
                    self.max_memory_gb,
                )
                for fasta, bed in tqdm(files.iter_rows(), total=files.height, desc="Initializing fastas")
            ]

        self.warmup_fastas: List[Fasta] = []
        if seqlen_warmup is not None:
            n = max_length * batch_size
            # mamba used ~6 Gb per selqen, doubling seqlen each time
            for seqlen, tokens in seqlen_warmup:
                b = n // seqlen
                n_regions = np.array([f.train_bed.height for f in self.fastas])
                frac_bed = tokens / (n_regions * seqlen).sum()
                sizes = (frac_bed * n_regions).astype(int)
                for f, size in zip(self.fastas, sizes):
                    # checked that shallow copy doesn't share integer attributes
                    _f = copy(f)
                    _f.set(max_length=seqlen, batch_size=b, sample_bed=size)
                    self.warmup_fastas.append(_f)

    def setup(self):
        for fasta in tqdm(self.fastas, desc="Setting up fastas"):
            fasta.setup()

    def init_datasets(self):
        for fasta in self.fastas:
            fasta.init_datasets()

    @property
    def max_length(self):
        return self._max_length

    @max_length.setter
    def max_length(self, value: int):
        self._max_length = value
        for fasta in self.fastas:
            fasta._max_length = value
        self.init_datasets()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value
        for fasta in self.fastas:
            fasta._batch_size = value
        self.init_datasets()

    def train_dataloader(self, **kwargs):
        return chain(
            *(f.train_dataloader() for f in self.warmup_fastas),
            Compose(
                *(fasta.train_dataloader() for fasta in self.fastas), mode="random"
            ),
        )

    def val_dataloader(self, **kwargs):
        return Compose(
            *(fasta.val_dataloader() for fasta in self.fastas), mode="interleave"
        )

    def test_dataloader(self, **kwargs):
        return Compose(
            *(fasta.test_dataloader() for fasta in self.fastas), mode="interleave"
        )


class ThousandGP(SequenceDataset):
    _name_ = "1kgp"

    def __init__(
        self,
        ref: Union[str, Path],
        pgen: Union[str, Path],
        bed: Union[str, Path],
        max_length: int,
        batch_size: int,
        max_memory_gb: float,
        *args,
        **kwargs,
    ):
        self.name = NAME
        self.ref = gvl.Fasta("_", ref, "N", "dna")
        self.pgen = gvl.Pgen(pgen)
        self.varseq = gvl.FastaVariants(self.name, self.ref, self.pgen)
        self.bed = read_bedlike(bed).rename({"name": "split"})

        self._max_length = max_length
        self._batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.transform = TOKENIZER

        self.setup()

    def setup(self):
        partitions = self.bed.partition_by("split", as_dict=True)
        self.train_bed = partitions["train"]
        self.val_bed = partitions["valid"]
        self.test_bed = partitions["test"]

        self.init_datasets()

    def init_datasets(self):
        self.train_dataset = self._gvl("fit").torch_dataset()
        self.val_dataset = self._gvl("val").torch_dataset()
        self.test_dataset = self._gvl("test").torch_dataset()

    def _gvl(self, split: str):
        if split == "fit":
            bed = self.train_bed
            shuffle = True
        elif split == "val":
            bed = self.val_bed
            shuffle = False
        elif split == "test":
            bed = self.test_bed
            shuffle = False
        else:
            raise ValueError(f"Invalid split: {split}")

        return gvl.GVL(
            self.varseq,
            bed=bed,
            fixed_length=self.max_length,
            batch_size=self.batch_size,
            max_memory_gb=self.max_memory_gb,
            batch_dims=["sample", "ploid"],
            shuffle=shuffle,
            transform=self.transform,
            return_tuples=[self.name, "target"],
        )

    def train_dataloader(self, **kwargs):
        return self.train_dataset.torch_dataloader()

    def val_dataloader(self, **kwargs):
        return self.val_dataset.torch_dataloader()

    def test_dataloader(self, **kwargs):
        return self.test_dataset.torch_dataloader()

    @property
    def max_length(self):
        return self._max_length

    @max_length.setter
    def max_length(self, value: int):
        self._max_length = value
        self.init_datasets()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value
        self.init_datasets()


class ThousandGP_MultiFasta(SequenceDataset):
    _name_ = "1kgp_multifasta"

    def __init__(
        self,
        ref: Union[str, Path],
        pgen: Union[str, Path],
        bed: Union[str, Path],
        fasta_dir: Union[str, Path],
        max_length: int,
        batch_size: int,
        max_memory_gb: float,
        *args,
        **kwargs,
    ):
        self.name = NAME

        self._max_length = max_length
        self._batch_size = batch_size
        self.max_memory_gb = max_memory_gb

        self.thousandgp = ThousandGP(
            ref, pgen, bed, max_length, batch_size, max_memory_gb
        )
        self.multifasta = MultiFasta(fasta_dir, max_length, batch_size, max_memory_gb)

    def setup(self):
        self.thousandgp.setup()
        self.multifasta.setup()

    def init_datasets(self):
        self.thousandgp.init_datasets()
        self.multifasta.init_datasets()

    @property
    def max_length(self):
        return self._max_length

    @max_length.setter
    def max_length(self, value: int):
        self._max_length = value
        self.thousandgp._max_length = value
        self.multifasta._max_length = value
        self.init_datasets()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value
        self.thousandgp._batch_size = value
        self.multifasta._batch_size = value
        self.init_datasets()

    def train_dataloader(self, **kwargs):
        return random_chain(
            self.thousandgp.train_dataloader(), self.multifasta.train_dataloader()
        )

    def val_dataloader(self, **kwargs):
        return interleave_longest(
            self.thousandgp.val_dataloader(), self.multifasta.val_dataloader()
        )

    def test_dataloader(self, **kwargs):
        return interleave_longest(
            self.thousandgp.test_dataloader(), self.multifasta.test_dataloader()
        )
