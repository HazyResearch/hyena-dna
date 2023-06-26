r"""
Sequence Length Warmup by Reloading
====================
Change sequence lengths according to a stage schedule. The stage parameters sets the sequence length
and batch size. 


TODO (not yet supported):
If batch size is not provided for that stage, calculate the batch size based on the
sequence length reshaping into the batch size.

"""

import numpy as np
from pytorch_lightning.callbacks import Callback

import src.utils as utils
from src.utils import registry


class SeqlenWarmupReload(Callback):

    def __init__(self, stage_params: list):
        """
        stage_params is a list of dicts
        e.g. stage_params = [
            {'seq_len': 512, 'epochs': 50},
            {'seq_len': 256, 'epochs': 30},
            {'seq_len': 128, 'epochs': 20},
        ]
        """
        super().__init__()
        assert len(stage_params) > 0, 'No stages specified'
        assert all([{'seq_len', 'epochs'} <= set(stage.keys()) for stage in stage_params]), \
            'stage_params must contain keys: seq_len and epochs'

        self.stage_params = stage_params
        self.stage_epochs_cume = np.cumsum([stage['epochs'] for stage in stage_params])

        self._current_stage = 0

    def _verify_stages(self, trainer, model):
        # Double-check that stage parameters are correct, otherwise we'll fail in the middle of training
        for stage in self.stage_params:
            if hasattr(stage, 'scheduler'):
                # Verify that we can actually create the scheduler when we need to update it in each stage
                scheduler = utils.instantiate(registry.scheduler, {**model.hparams.scheduler, **stage['scheduler']}, trainer.optimizers[0])
                del scheduler

    def on_train_start(self, trainer, model) -> None:
        # Verify all the stage parameters are correct
        self._verify_stages(trainer, model)

        print(f"Training starts at {trainer.current_epoch}")
        if trainer.current_epoch == 0:
            # Update the model to the first stage
            self._update_to_current_stage(trainer, model)
        else:
            # Preemption or resumption of progressive resizing
            # Update the stage to the current one
            self._current_stage = int(np.searchsorted(self.stage_epochs_cume - 1, trainer.current_epoch))
            self._starting_stage = np.any(trainer.current_epoch == self.stage_epochs_cume)

            print("Seq Len Warmup: Restarting at Stage {}".format(self._current_stage))
            if self._starting_stage:
                self._update_lr_scheduler(trainer, model)

            # Set the dataloader and model
            self._update_dataloaders(trainer, model)
            # self._update_model(trainer, model)  # we don't need to update the model, yet

        return super().on_train_start(trainer, model)

    def _update_lr_scheduler(self, trainer, model):

        if not hasattr(self.stage_params[self._current_stage], 'scheduler'):
            # No scheduler specified, so don't update the current scheduler
            return

        assert len(trainer.lr_schedulers) == 1
        # Reinitialize the scheduler
        # We don't need to carry over information from the last scheduler e.g. the last_epoch property,
        # because that will mess with the new scheduler when we step it
        hparams = {**model.hparams.scheduler, **self.stage_params[self._current_stage]['scheduler']}
        

        # Note that passing in the optimizer below is okay: the scheduler will be reinitialized and doesn't seem to inherit any current lr info from the optimizer
        trainer.lr_schedulers[0]['scheduler'] = utils.instantiate(registry.scheduler, hparams, trainer.optimizers[0])

        print("\tChanged scheduler to {}".format(hparams))

    def _update_dataloaders(self, trainer, model):
        # Set the train resolution and reset the dataloader

        # set new seq len and reset the dataloader
        # max_length should be set in the config of the dataloader
        seq_len = self.stage_params[self._current_stage]['seq_len']
        model.hparams.loader.max_length = seq_len

        # we need to resize the batch size too
        batch_size = self.stage_params[self._current_stage].get('batch_size', None)
        
        # need to change the dataset params, and the set the phase, which reinits the dataset
        model.dataset.max_length = seq_len  # progressively update the seq len
        # model.dataset.max_length_val = seq_len  # we update the val len to be same as train
        # model.dataset.max_length_test = seq_len  # we don't change the test set, always the longest
        model.dataset.batch_size = batch_size  # need to adjust the batch size
        # model.dataset.batch_size_eval = batch_size * 2  #

        # model.dataset.dataset_train.max_length = seq_len

        model.dataset.init_datasets()  # reinit the datasets with new batch size and seq len

        trainer.reset_train_dataloader(model)  # tells PTL to use the new dataloaders/datasets
        trainer.reset_val_dataloader(model)
        print('\tAt epoch {}, changed Seq Len to {}, and batch size to {}'.format(trainer.current_epoch, seq_len, batch_size))

    # def _update_model(self, trainer, model):
    #     if not hasattr(self.stage_params[self._current_stage], 'bandlimit'):
    #         return

        # Update the bandlimit value for the model: this is a hack to make sure the model is updated
        # Iterate over all the modules
        # for module in model.modules():
        #     if hasattr(module, 'bandlimit'):
        #         module.bandlimit = self.stage_params[self._current_stage]['bandlimit']

        # print('\tChanged bandlimit to {}'.format(self.stage_params[self._current_stage]['bandlimit']))

    def _update_to_current_stage(self, trainer, model):
        print("Seq Len Warmup: Moving to Stage {}".format(self._current_stage))
        # Update the train dataloader, model and scheduler
        self._update_dataloaders(trainer, model)
        # self._update_model(trainer, model)
        self._update_lr_scheduler(trainer, model)


    def on_train_epoch_end(self, trainer, model):
        """
        Check to see if new stage is reached for the next epoch, and if so, prepare the new stage by
        changing the dataloader.

        (We do next epoch so that the dataloader is prepared before the next epoch)
        """
        next_epoch = trainer.current_epoch + 1

        # Check if stage should be increased
        if next_epoch >= self.stage_epochs_cume[self._current_stage] and self._current_stage < len(self.stage_params) - 1:
            self._current_stage += 1
            self._update_to_current_stage(trainer, model)

        return super().on_train_epoch_end(trainer, model)
