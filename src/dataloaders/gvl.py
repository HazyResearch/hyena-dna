from typing import Dict, Hashable

import genvarloader as gvl
import numpy as np
from attrs import define
from genvarloader.util import read_bedlike
from numpy.typing import NDArray

from src.dataloaders.base import SequenceDataset


def _tokenize(
    seq: NDArray,
    tokenize_table: Dict[bytes, int],
    add_eos: bool,
    eos_id: int,
    dtype=np.int32,
):
    LENGTH_AXIS = seq.ndim - 1
    if add_eos:
        shape = seq.shape[:LENGTH_AXIS] + (seq.shape[LENGTH_AXIS] + 1,)
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
class _Transform:
    name: str
    tokenize_table: Dict[bytes, int]
    add_eos: bool
    eos_id: int
    
    def __call__(self, batch: Dict[Hashable, NDArray]):
        seq = _tokenize(
            batch[self.name], self.tokenize_table, self.add_eos, self.eos_id
        )
        batch[self.name] = seq[..., :-1].astype(np.int32)
        batch['target'] = seq[..., 1:].astype(np.int64)
        return batch


class GVL_HG38(SequenceDataset):
    _name_ = 'gvl_hg38'
    
    def __init__(self, fasta: str, bed: str, max_length: int, batch_size: int, max_memory_gb: float, *args, **kwargs):
        self.fasta = gvl.Fasta('seq', fasta, 'N', 'dna', in_memory=True)
        self.name = 'seq'
        
        self._max_length = max_length
        self._batch_size = batch_size
        self.max_memory_gb = max_memory_gb
        self.transform = _Transform(
            name=self.name,
            tokenize_table={b"A": 7, b"C": 8, b"G": 9, b"T": 10, b"N": 11},
            add_eos=False,
            eos_id=0,
        )
        
        self.bed = read_bedlike(bed)
        if 'name' not in self.bed:
            raise RuntimeError('Need name column to use for identifying splits.')
        self.bed = self.bed.rename({'name': 'split'})
        self.setup()
    
    def setup(self):
        partitions = self.bed.partition_by('split', as_dict=True)
        self.train_bed = partitions['train']
        self.val_bed = partitions['valid']
        self.test_bed = partitions['test']
        
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
    
    def _gvl(self, split: str):
        if split == 'fit':
            bed = self.train_bed
            shuffle = True
        elif split == 'val':
            bed = self.val_bed
            shuffle = False
        elif split == 'test':
            bed = self.test_bed
            shuffle = False
        else:
            raise ValueError(f'Invalid split: {split}')
        
        return gvl.GVL(
            self.fasta,
            bed=bed,
            fixed_length=self._max_length,
            batch_size=self.batch_size,
            max_memory_gb=self.max_memory_gb,
            transform=self.transform,
            return_tuples=True,
            shuffle=shuffle,
        )
    
    def init_datasets(self):
        self.train_dataset = self._gvl('fit').torch_dataset()
        self.val_dataset = self._gvl('val').torch_dataset()
        self.test_dataset = self._gvl('test').torch_dataset()
    
    def train_dataloader(self, **kwargs):
        return self.train_dataset.torch_dataloader()
    
    def val_dataloader(self, **kwargs):
        return self.val_dataset.torch_dataloader()
    
    def test_dataloader(self, **kwargs):
        return self.test_dataset.torch_dataloader()