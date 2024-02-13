import sys

import polars as pl

sys.path.insert(0, '/iblm/netapp/home/dlaub/projects/hyena-dna/')

from src.dataloaders.gvl import MultiFasta  # noqa: E402

files = pl.read_csv('/iblm/netapp/home/dlaub/projects/hyena-dna/data/cactus_arrow_dataset_65536.txt', separator='\t')

dm = MultiFasta(
    files,
    max_length=65536,
    batch_size=16,
    max_memory_gb=16,
)