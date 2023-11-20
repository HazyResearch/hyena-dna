from typing import Dict, Hashable, Optional

import numpy as np
import seqpro as sp
from attrs import define
from numpy.typing import NDArray
from tqdm.auto import tqdm

from src.dataloaders.gvl import GVL_HG38


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
    last_batch: Optional[Dict[Hashable, NDArray]] = None
    
    def __call__(self, batch: Dict[Hashable, NDArray]):
        original_seq = batch[self.name]
        nuc_dist = sp.nucleotide_content(original_seq, length_axis=-1, alphabet=sp.DNA)
        seq = _tokenize(
            original_seq, self.tokenize_table, self.add_eos, self.eos_id
        )
        batch = {
            self.name: seq[..., :-1],
            'target': seq[..., 1:].astype(np.int64),
        }
        if self.last_batch is not None:
            try:
                np.testing.assert_equal(batch, self.last_batch)
            except AssertionError:
                pass
            else:
                print(np.unique(original_seq))
                print(nuc_dist)
                raise RuntimeError('Caught identical batches.')
        self.last_batch = batch
        return batch

if __name__ == '__main__':
    name='seq'
    fasta = '/cellar/users/dlaub/projects/HyenaDNA_collab/data/human/hg38.ml.fa'
    bed_file = '/cellar/users/dlaub/projects/HyenaDNA_collab/data/human/sequences.bed'
    fixed_length=1024
    batch_size=256
    max_memory_gb = 64
    
    # bed = read_bedlike(bed_file).rename({'name': 'split'})
    # bed = bed.filter(pl.col('split') == 'train')
    # bed = _set_fixed_length_around_center(bed, fixed_length)
    # chrom, start, end, split = bed.row(0)
    # fasta_reader = gvl.Fasta(name, fasta, 'N', 'dna', in_memory=True)
    # print(chrom, start, end)
    # print(fasta_reader.read(chrom, np.array([start]), np.array([end])))
    
    # transform = _Transform(
    #     name=name,
    #     tokenize_table={b"A": 7, b"C": 8, b"G": 9, b"T": 10, b"N": 11},
    #     add_eos=True,
    #     eos_id=1,
    # )
    
    # gvloader = gvl.GVL(
    #     fasta_reader,
    #     bed=bed,
    #     batch_size=batch_size,
    #     fixed_length=fixed_length,
    #     max_memory_gb=max_memory_gb,
    #     transform=transform,
    #     return_tuples=['seq', 'target']
    # )
    # dl = gvloader.torch_dataloader()
    
    dm = GVL_HG38(
        fasta=fasta,
        bed=bed_file,
        max_length=fixed_length,
        batch_size=batch_size,
        max_memory_gb=max_memory_gb,
    )
    dm.setup()
    dl = dm.train_dataloader()
    
    for batch in tqdm(dl):
        continue