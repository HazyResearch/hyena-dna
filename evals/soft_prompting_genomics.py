#!/usr/bin/env python3
import argparse
import yaml 
from tqdm import tqdm
import typing as tp
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
import torch 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat

import sys, os
FILEDIR = os.path.realpath(__file__)
sys.path.append(os.path.join(FILEDIR, '..'))
from src.models.sequence.long_conv_lm import ConvLMHeadModel
# from src.dataloaders.icl_genomics_dataloader import ICLGenomics
from src.dataloaders.genomics import ICLGenomics

def exists(x):
    return x is not None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def soft_prompting():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", help="Path to pretrained model checkpoint")
    parser.add_argument("--dataset", default='none')
    parser.add_argument("--config", default='./configs/evals/soft_prompting_genomics.yaml')
    parser.add_argument("--results", default='./results/soft_prompting')
    args = parser.parse_args()
    os.makedirs(args.results, exist_ok=True)

    # load configs
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    cfg_model = config['model'].copy()
    cfg_dataset = config['dataset'].copy()
    cfg_tuning = config['tuning'].copy()
    
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    rng = np.random.RandomState(config['seed'])

    # dataset_name                                num_seqs        num_classes     median_len    std
    # dummy_mouse_enhancers_ensembl       1210            2               2381          984.4  
    # demo_coding_vs_intergenomic_seqs    100_000         2               200           0
    # demo_human_or_worm                  100_000         2               200           0
    # human_enhancers_cohn                27791           2               500           0
    # human_enhancers_ensembl             154842          2               269           122.6
    # human_ensembl_regulatory            289061          3               401           184.3
    # human_nontata_promoters             36131           2               251           0
    # human_ocr_ensembl                   174756          2               315           108.1
    # chrom_names = [
    #     'chr11', 'chr13', 'chr15', 'chr17', 'chr19', 'chr21', 'chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr12',
    #     'chr14', 'chr16', 'chr18', 'chr20', 'chr22', 'chrX', 'chrY', 'chr1', 'chr3', 'chr5', 'chr7', 'chr9'
    # ]
    nuc_chars = list('ACGTN')
    characters = nuc_chars # + chrom_names
    label_to_token = {0: 'A', 1: 'N'}
    datasets = {
        'dummy_mouse_enhancers_ensembl': {
            'max_length': 3200,
            'd_output': 2,
            'characters': characters,
            'label_to_token': label_to_token,
        },
        # 'demo_coding_vs_intergenomic_seqs': {
        #     'max_length': 202,
        #     'd_output': 2,
        #     'characters': characters,
        #     'label_to_token': label_to_token
        # },
        # 'demo_human_or_worm': {
        #     'max_length': 202,
        #     'd_output': 2,
        #     'characters': characters,
        #     'label_to_token': label_to_token,
        # },
        'human_enhancers_cohn': {
            'max_length': 502,
            'd_output': 2,
            'characters': characters,
            'label_to_token': label_to_token,
        },
        'human_nontata_promoters': {
            'max_length': 251, #253
            'd_output': 2,
            'characters': characters,
            'label_to_token': label_to_token,
        },
        'human_enhancers_ensembl': {
            'max_length': 320,
            'd_output': 2,
            'characters': characters,
            'label_to_token': label_to_token,
        },
        'human_ensembl_regulatory': {
            'max_length': 600,
            'd_output': 3,
            'characters': characters,
            'label_to_token': {0: 'A', 1: 'G', 2: 'N'},
        },
        'human_ocr_ensembl': {
            'max_length': 420,
            'd_output': 2,
            'characters': characters,
            'label_to_token': label_to_token,
        }
    }

    df_results = []
    df_i = 0
    ds_iter = datasets.items() if args.dataset=='none' else zip([args.dataset], [datasets[args.dataset]])
    for dataset, dataset_cfg in ds_iter:
        print(f'\nDataset {dataset}...')

        for shots in cfg_dataset['shots']:
            print(f'...with {shots} shots...')
            
            cfg = cfg_dataset.copy()
            cfg.update(dataset_cfg)
            cfg['dataset_name'] = dataset
            cfg['shots'] = shots
            loader = ICLGenomics(**cfg)
            loader.setup()
                
            for soft_tokens in cfg_tuning['soft_tokens']:
                print(f'...and {soft_tokens} soft tokens...')

                # print('Pretrained model...')
                pretrained_model = load_model(
                    cfg_model=cfg_model,
                    ckpt_path=args.ckpt_path,
                    n_soft_tokens=soft_tokens,
                    soft_token_pdrop=cfg_tuning['soft_token_pdrop'],
                    max_length=cfg['max_length'] if shots>0 else None
                )
                pretrained_model.to(DEVICE)
                if soft_tokens>0: # we only tune when using soft tokens!
                    print('...tuning...')
                    pretrained_model = tune_model(
                        pretrained_model, #deepcopy(pretrained_model).to(DEVICE),
                        loader,
                        cfg_tuning,
                        rng=rng
                    )
                print('...evaluating...')
                acc = eval_on_loaders(pretrained_model, {dataset: loader})[dataset]
                df_results.append(
                    pd.DataFrame({
                        'dataset': dataset,
                        'model': 'pretrained',
                        'shots': shots,
                        'soft_tokens': soft_tokens,
                        'eval_acc': acc
                    }, index=[df_i])
                )
                df_i += 1
                pd.concat(df_results).to_csv(
                    os.path.join(
                    args.results,
                    f'soft_prompting_performance_{dataset}.csv'
                )
                )

                del pretrained_model


def load_model(
    cfg_model: tp.Dict,
    ckpt_path: str=None,
    n_soft_tokens: int=0,
    soft_token_pdrop: float=0.,
    max_length: int=None
):
    model = ConvLMHeadModel(**cfg_model)

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # loads model from ddp by removing prexix to single if necessary
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["state_dict"], "model."
        )
        model_state_dict = state_dict["state_dict"]
        # need to remove torchmetrics. to remove keys, need to convert to list first
        for key in list(model_state_dict.keys()):
            if "torchmetrics" in key:
                model_state_dict.pop(key)

        model.load_state_dict(model_state_dict)   
  
    return LitModel(model, n_soft_tokens=n_soft_tokens, soft_token_pdrop=soft_token_pdrop, max_length=max_length)    


class LitModel(pl.LightningModule):
    def __init__(self,
        model,
        n_soft_tokens: int=0,
        soft_token_pdrop: float=0.,
        max_length: int=None
    ):
        super().__init__()
        self.model = model
        requires_grad(self.model, False) # we only want to train soft tokens
        self.max_length = max_length
        d_model = self.model.lm_head.weight.shape[1]
        self.n_soft_tokens = n_soft_tokens
        soft_tokens = torch.nn.Parameter(torch.zeros(n_soft_tokens, d_model)) if n_soft_tokens>0 else None
        if exists(soft_tokens):
            torch.nn.init.normal_(soft_tokens, mean=0.0, std=0.02)
        self.soft_tokens = soft_tokens
        self.soft_tokens_drop = torch.nn.Dropout(soft_token_pdrop) if soft_token_pdrop>0 else torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        
        # get embeddings
        with torch.no_grad():
            hidden_states = self.model.backbone.embeddings(x)

        # attach soft tokens
        if exists(self.soft_tokens):
            hidden_states = torch.cat([
                repeat(self.soft_tokens_drop(self.soft_tokens), 'n d -> b n d', b=hidden_states.shape[0]),
                hidden_states
            ], dim=1)

        # forward
        residual = None
        for layer in self.model.backbone.layers:
            hidden_states, residual = layer(hidden_states, residual)
        dropped = self.model.backbone.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.model.backbone.ln_f(residual.to(dtype=self.model.backbone.ln_f.weight.dtype))

        return self.model.lm_head(hidden_states)
        
    def step(self, batch: tp.Tuple[torch.Tensor], phase: str='train'):
        
        # get ys
        x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
        labels_idx = x.shape[1]-1
        if exists(self.max_length):
            x = torch.cat([x, y], dim=1)
            labels_idx = self.get_labels_idx(x)
            y = x[:,labels_idx]

        # forward
        logits = self(x)
        logits = logits[:,self.n_soft_tokens:] # we exclude soft tokens
        logits = logits[:,labels_idx-1] # previous token predicts target
        if logits.ndim>2:
            logits = rearrange(logits, 'b n c -> (b n) c')
        if y.ndim==2:
            y = rearrange(y, 'b n -> (b n)')

        # compute loss/acc
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(axis=-1)
        acc = torch.mean((preds==y).to(torch.float32))
        return {'loss': loss, 'acc': acc}
    
    def get_labels_idx(self, x):
        return np.concatenate([
            [self.max_length+1],
            np.arange((2*self.max_length)+4, x.shape[1], self.max_length+3)
        ])


def tune_model(model, loader, cfg_tuning, verbose: bool=True, rng: np.random.RandomState=None):

    rng = np.random.RandomState(0) if rng is None else rng

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=float(cfg_tuning['weight_decay']),
        lr=float(cfg_tuning['lr'])
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=0
    )

    best_model = deepcopy(model)
    requires_grad(best_model, False)

    step = 0
    losses, accs, val_losses = [], [], []
    for epoch in range(cfg_tuning['max_epochs']):
        if verbose:
            print(f'Epoch {epoch}...')

        # train epoch:
        model.train()
        for i, (x,y) in enumerate(loader.train_dataloader()):
            batch = {'x': x, 'y': y}
            model.on_train_batch_start(batch=batch, batch_idx=step)
            with torch.cuda.amp.autocast():
                out = model.step(batch)
            loss, acc = out['loss'], out['acc']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_tuning.get('gradient_clip_val', 1.0))
            losses.append(loss.cpu().detach().numpy().mean())
            accs.append(acc.cpu().detach().numpy())

            # accumulate gradients of N batches
            if (i + 1) % cfg_tuning['accumulate_grad_batches'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                # update_ema(ema, model, decay=cfg_tuning['ema_decay'])
                step += 1

        # eval epoch:
        model.eval()
        val_loss = []
        with torch.no_grad():
            for x, y in loader.val_dataloader():
                batch = {'x': x, 'y': y}
                model.on_train_batch_start(batch=batch, batch_idx=step)
                out = model.step(batch)
                loss, acc = out['loss'], out['acc']
                val_loss.append(loss.cpu().detach().numpy())
            val_losses.append(np.mean(val_loss))

        if val_losses[-1]==np.min(val_losses): # also covers first epoch
            update_ema(best_model, model, decay=0)
        
        scheduler.step(val_losses[-1])

        if verbose:
            print(f'\tstep {step}; avg. val loss: {val_losses[-1]:1.4f}')

        if (epoch > 0 and sum(val_losses[-1] >= val_losses[:-1])>1) or (epoch+1)>=cfg_tuning['max_epochs']:
            break
    
    best_model = best_model.to(DEVICE)
    requires_grad(best_model, True) # we turn grads back on for completion, even though model will not be trained further...
    return best_model #, ema


@torch.no_grad()
def update_ema(ema_model, model, decay=0.999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def eval_on_loaders(model, loaders):

    results = {}
    for name, loader in loaders.items():
        print(f'Evaluating on {name} data...')

        all_acc = []
        val_loader = loader.val_dataloader()
        for x,y in tqdm(val_loader):
            x = x.to(DEVICE)
            
            with torch.no_grad():
                logits = model(x)
                logits = logits[:, -1]
                logits = logits.cpu().detach().numpy()
                batch_preds = logits.argmax(axis=-1)

            # batch_preds = np.array(batch_preds)
            y = y.cpu().detach().numpy()
            batch_preds = batch_preds.flatten()
            y = y.flatten()
            acc = (batch_preds == y).mean()
            all_acc.append(acc)
        
        results[name] = np.mean(all_acc)
        print(f"{name}; full eval. accuracy: {results[name]:1.4f}")

    return results


if __name__ == "__main__":
    soft_prompting()