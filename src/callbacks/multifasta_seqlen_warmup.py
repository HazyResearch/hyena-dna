from typing import cast

import numpy as np
import polars as pl
import pytorch_lightning as lit

from src.dataloaders.multifasta import MultiFasta


class SeqLenWarmup(lit.Callback):
    def __init__(self, max_length: int, tokens_per_step: int, init_length: int = 2**10):
        if max_length <= init_length:
            raise ValueError("max_length must be greater than init_length")

        self.tokens_per_step = tokens_per_step

        lengths = 2 ** np.arange(
            np.log2(init_length), np.ceil(np.log2(max_length)) + 1, dtype=int
        )
        lengths[-1] = max_length
        self.lengths = lengths.astype(int)
        self.step = 0
        self.og_beds = None
        self.max_tokens_per_batch = None

    def on_train_epoch_start(
        self, trainer: lit.Trainer, pl_module: lit.LightningModule
    ) -> None:
        dm = cast(MultiFasta, pl_module.dataset)  # type: ignore[reportGeneralTypeIssues]
        trainer.limit_val_batches = 0
        trainer.limit_test_batches = 0
        
        if self.og_beds is None and self.max_tokens_per_batch is None:
            self.og_beds = dm.beds
            self.max_tokens_per_batch = int(dm.batch_size * self.lengths[-1])

        current_length = int(self.lengths[min(self.step, len(self.lengths)-1)])
        dm.batch_size = self.max_tokens_per_batch // current_length
        pl_module.log("seqlen_warmup_step", self.step)
        pl_module.log("seqlen_warmup_length", current_length)
        pl_module.log("seqlen_warmup_batch_size", dm.batch_size)

        if self.step >= len(self.lengths) - 1:
            pl_module.log("seqlen_warmup_finished", True)
            dm.beds = self.og_beds
            trainer.limit_val_batches = 1
            trainer.limit_test_batches = 1
        else:
            pl_module.log("seqlen_warmup_finished", False)
            dm.beds = {
                split: self._with_length(bed, current_length, self.tokens_per_step, stage=split)
                for split, bed in self.og_beds.items()
            }
        self.step += 1

        dm.setup()
        trainer.reset_train_dataloader(pl_module)
        trainer.reset_val_dataloader(pl_module)
        trainer.reset_test_dataloader(pl_module)

    @staticmethod
    def _with_length(bed: pl.DataFrame, length: int, n_tokens: int, stage: str):
        midpt = (pl.col("chromEnd") + pl.col("chromStart")) / 2
        bed = bed.filter(
            pl.col("chromEnd") - pl.col("chromStart") >= length
        ).with_columns(
            chromEnd=(midpt + length / 2).round(),
            chromStart=(midpt - length / 2).round(),
        )
        if stage == "train":
            return bed.sample(np.ceil(n_tokens / length), with_replacement=True)
        else:
            return bed
