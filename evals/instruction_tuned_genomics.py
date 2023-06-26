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
from einops import rearrange

import sys, os
FILEDIR = os.path.realpath(__file__)
sys.path.append(os.path.join(FILEDIR, '..'))
from src.models.sequence.long_conv_lm import ConvLMHeadModel
# from src.dataloaders.icl_genomics_dataloader import ICLGenomics
from src.dataloaders.genomics import ICLGenomics



# TODO: 
# Make use of maximum long context: either put entire downstream dataset in context
# or add many tunable soft tokens (soft prompting)!
# -> just fill the context up one way or another and show whats possible!



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def instruction_tuned_ICL():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", help="Path to pretrained model checkpoint")
    parser.add_argument("--config", default='./configs/evals/instruction_tuned_genomics.yaml')
    parser.add_argument("--results", default='./results/instruction_tuned_genomics')
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
    
    nuc_chars = list('ACGTN')
    characters = nuc_chars # + chrom_names
    label_to_token = {0: 'A', 1: 'N'}
    datasets = {
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

    print('\n\nEvaluating instruction-tuned ICL performance... ')
    df_results = []
    df_i = 0
    for tuning_samples in cfg_tuning['tuning_samples']:
        print(f'...when tuning on {tuning_samples} samples...')

        for shots in cfg_dataset['shots']:
            print(f'...with {shots} shots...')
        
            for dataset, dataset_cfg in datasets.items():
                print(f'...from dataset {dataset}...')

                print(f'Collecting tuning data...')
                cfg = cfg_dataset.copy()
                cfg.update(dataset_cfg)
                cfg['dataset_name'] = dataset
                cfg['shots'] = shots
                loader = ICLGenomics(**cfg)
                loader.setup()
                
                # collect tuning samples
                tuning_X = []
                train_loader = iter(loader.train_dataloader())
                samples_collected = 0
                for x, y  in tqdm(train_loader):
                    n = min(tuning_samples, x.shape[0])
                    tuning_X.append(torch.cat([x[:n], y[:n]], dim=1))
                    samples_collected += n
                    if samples_collected >= tuning_samples:
                        print(f'...stop becuase {tuning_samples} samples collected.')
                        break
                tuning_X = torch.cat(tuning_X, dim=0)
                if shots>0:
                    tuning_y_idx = np.concatenate([
                        [cfg['max_length']+1],
                        np.arange((2*cfg['max_length'])+4, tuning_X.shape[1], cfg['max_length']+3)
                    ])
                else:
                    tuning_y_idx = cfg['max_length']+1
                tuning_y = tuning_X[:,tuning_y_idx]
                tuning_loss_mask = tuning_y_idx-1 # prediction is always from previous token

                print('Tuning pretrained model...')
                pretrained_model = load_model(cfg_model, args.ckpt_path)
                pretrained_model.to(DEVICE)
                tuned_pretrained_model = tune_model(
                    deepcopy(pretrained_model).to(DEVICE),
                    tuning_X,
                    tuning_y,
                    cfg_tuning,
                    loss_mask=tuning_loss_mask,
                    rng=rng
                )
                
                # print('Tuning untrained model...')
                # scratch_model = load_model(cfg_model)
                # scratch_model.to(DEVICE)
                # tuned_scratch_model = tune_model(
                #     scratch_model,
                #     tuning_X,
                #     tuning_y,
                #     cfg_tuning,
                #     loss_mask=tuning_loss_mask,
                #     rng=rng
                # )

                print('Evaluating ICL performance...')
                for label, model in zip(
                    ['tuned_pretrained'], #, 'scratchtrained' 
                    [tuned_pretrained_model] # tuned_scratch_model
                ):
                    print(f'{label}:')
                    acc = eval_on_loaders(model, {dataset: loader})[dataset]
                    df_results.append(
                        pd.DataFrame({
                            'dataset': dataset,
                            'tuning_samples': tuning_samples,
                            'model': label,
                            'shots': shots,
                            'eval_acc': acc
                        }, index=[df_i])
                    )
                    df_i += 1
                    pd.concat(df_results).to_csv(
                        os.path.join(args.results, 'instruction_tuned_genomics.csv')
                    )

def load_model(cfg_model, ckpt_path: str=None):

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
  
    return LitModel(model)


class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model(x)[0]
    
    def step(self, batch: tp.Tuple[torch.Tensor], loss_mask: tp.Union[int, np.ndarray]=-1, phase: str='train'):
        x, y = batch['x'].to(DEVICE), batch['y'].to(DEVICE)
        loss_mask = -1 if loss_mask is None else loss_mask
        out = self(x)
        logits = out.logits[:,loss_mask]
        if logits.ndim>2:
            logits = rearrange(logits, 'b n c -> (b n) c')
        if y.ndim==2:
            y = rearrange(y, 'b n -> (b n)')
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(axis=-1)
        acc = torch.mean((preds==y).to(torch.float32))
        return {'loss': loss, 'acc': acc}


def tune_model(model, X, y, cfg_tuning, max_epochs: int=1, loss_mask=None, verbose: bool=True, rng: np.random.RandomState=None):

    rng = np.random.RandomState(0) if rng is None else rng

    # # we use expected moving average of model for downstream ICL...
    # ema = deepcopy(model).to(DEVICE)
    # requires_grad(ema, False)
    # update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    # ema.eval()
        
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=float(cfg_tuning['weight_decay']),
        lr=float(cfg_tuning['lr'])
    )
    
    # split train/eval
    n_samples = X.shape[0]
    train_idx = np.arange(n_samples)

    batch_size = min(len(train_idx), cfg_tuning['batch_size'])
    epoch = 0
    step = 0
    losses, accs = [], []
    stop_training = False
    while not stop_training:
        if verbose:
            print(f'Epoch {epoch}...')
    
        # train epoch:
        model.train()
        rng.shuffle(train_idx)
        batch_i, batch_start = 0, 0
        while batch_start+batch_size <= len(train_idx):
            
            idx = train_idx[batch_start:batch_start+batch_size]
            batch = {'x': X[idx], 'y': y[idx]}
            model.on_train_batch_start(batch=batch, batch_idx=step)
            out = model.step(batch, loss_mask=loss_mask)
            loss, acc = out['loss'], out['acc']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_tuning.get('gradient_clip_val', 1.0))
            losses.append(loss.cpu().detach().numpy().mean())
            accs.append(acc.cpu().detach().numpy())
            
            # accumulate gradients of N batches
            if (batch_i + 1) % cfg_tuning['accumulate_grad_batches'] == 0:
                optimizer.step()
                optimizer.zero_grad()
                # update_ema(ema, model, decay=cfg_tuning['ema_decay'])
                step += 1
                print(f'step: {step}; train loss: {losses[-1]}, acc: {accs[-1]}')
            batch_start += batch_size
            batch_i += 1
        
        epoch += 1
        if epoch>=max_epochs:
            stop_training = True

    return model #, ema


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
        for batch in tqdm(val_loader):
            x, y = batch
            x = x.to(DEVICE)
            
            with torch.no_grad():
                out = model(x)
                if type(out) == tuple: out = out[0]
                logits = out.logits[:, -1]
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
    instruction_tuned_ICL()