from typing import Dict, Optional, Tuple, Union

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
        fasta_dir: Optional[str] = None,
        hg38: bool = False,
        *args,
        **kwargs,
    ):
        """Dataset of multiple FASTA files e.g. for training a model on multiple species.

        Parameters
        ----------
        file_table : Union[str, pl.DataFrame]
            Table with a column "fasta" containing the paths to the FASTA files.
        bed : Union[str, pl.DataFrame]
            Table with the columns "chrom", "chromStart", "chromEnd", "species", and "split".
            "split" should be one of "train", "valid", or "test".
        batch_size : int
        num_workers : int, optional
            by default 1
        limit_fastas : Optional[int], optional
            Whether to limit the number of FASTA files to use, by default None
        fasta_dir : Optional[str], optional
            Directory where the FASTA files are located, by default None. Will override the
            directory in the "fasta" column of `file_table`, keeping the rest of the path.
        hg38 : bool, optional
            Whether to only use the hg38 reference genome, by default False
        """
        if isinstance(file_table, str):
            fastas = pl.read_csv(file_table, separator="\t")["fasta"]
        else:
            fastas = file_table["fasta"]

        if isinstance(bed, str):
            bed = pl.read_ipc(bed)
        
        if hg38:
            fastas = fastas.filter(fastas.str.contains("Homo_sapiens"))
            bed = bed.filter(pl.col('species') == 'Homo_sapiens')

        if fasta_dir is not None:
            file_names = fastas.str.split("/").list.get(-1)
            fastas = fasta_dir.rstrip('/') + '/' + file_names
        else:
            fastas = fastas.str.replace(r"/iblm/netapp", r"/home/jovyan", literal=True)
        
        if limit_fastas is not None:
            fastas = fastas.head(limit_fastas)

        species = fastas.str.split("/").list.get(-1).str.split(".").list.get(0)

        if limit_fastas is not None:
            bed = bed.filter(pl.col("species").is_in(species))

        self.beds = bed.partition_by("split", as_dict=True, include_key=False)

        self.fastas = {
            s: gvl.Fasta(NAME, f, "N", "dna")
            for s, f in tqdm(
                zip(species, fastas), desc="Init FASTAs", total=len(species)
            )
        }
        # self.train_batch_size = batch_size
        self.batch_size = batch_size
        self.dl_kwargs = {
            "pin_memory": True,
            "num_workers": num_workers,
        }

    def setup(self):
        self.train_ds = TorchMultiFasta(self.fastas, self.beds["train"])
        self.val_ds = TorchMultiFasta(self.fastas, self.beds["valid"])
        self.test_ds = TorchMultiFasta(self.fastas, self.beds["test"])

    def train_dataloader(self, **kwargs):
        return DataLoader(self.train_ds, shuffle=True, batch_size=self.batch_size, **self.dl_kwargs)

    def val_dataloader(self, **kwargs):
        return DataLoader(self.val_ds, self.batch_size, **self.dl_kwargs)

    def test_dataloader(self, **kwargs):
        return DataLoader(self.test_ds, self.batch_size, **self.dl_kwargs)
