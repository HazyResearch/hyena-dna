import torch 

import argparse
import os
import sys
import yaml 
from tqdm import tqdm
import json 

from src.models.sequence.long_conv_lm import DNAEmbeddingModel
from src.tasks.decoders import SequenceDecoder
from src.dataloaders import SequenceDataset
import numpy as np

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from src.dataloaders.genomic_bench_dataloader import GenomicBenchmark
from src.dataloaders.nucleotide_transformer_dataloader import NucleotideTransformer

try:
    from tokenizers import Tokenizer  
except:
    pass


genomic_benchmark_datasets = ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm", "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_nontata_promoters", "human_ocr_ensembl"]
nucleotide_datasets = [""]

class HG38Inference:
    '''Model (backbone + decoder) inference, initially for enhancer model, but can be modified for other classification tasks as well.
    
        model_cfg, dict: config for entire model, backbone and decoder head
        ckpt_path, str: path to config
        max_seq_len, int: max seq len of model (technically in the model_cfg already, but more explicit)
    
    '''
    def __init__(self, cfg, ckpt_path, max_seq_len, use_dataloader=False):
        self.max_seq_len = max_seq_len
        self.backbone, self.decoder, self.tokenizer = self.load_model(cfg, ckpt_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self.backbone.to(self.device)
        self.decoder = self.decoder.to(self.device)

        # load dataloader if given
        if use_dataloader:
            self.loader = self.get_dataloader(cfg)

    def get_dataloader(self, config):
        cfg = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)

        dataset_name = cfg['dataset']["dataset_name"]

        if dataset_name in genomic_benchmark_datasets:
            loader = GenomicBenchmark(**cfg['dataset'])
        
        else:
            # assume the rest are in the nucleotide trans datasets
            loader = NucleotideTransformer(**cfg['dataset'])

        loader.setup()
        return loader

    def predict_on_list(self, seqs):
        """
        makes predictions just given a list of string sequences, handles all the tokenizers, and tensor conversion
        """
            
        preds = []

        # sample code to loop thru each sample and tokenize first (char level)
        for seq in tqdm(seqs):
            
            if isinstance(self.tokenizer, Tokenizer):
                seq = self.tokenizer.encode(seq).ids
            else:
                seq = self.tokenizer.encode(seq)

            # can accept a batch, shape [B, seq_len, hidden_dim]
            embeddings, _ = self.backbone(torch.tensor([seq]).to(device=self.device))

            pred = self.decoder(embeddings)
            preds.append(pred)

        # we provide the predictions (you can pass back embeddings if you wish)
        return preds
        
    def predict_from_loader(self):
        """
        Don't forget this returns a list of the labels too with the predictions
        """

        all_preds = []
        all_labels = []

        # by default we'll use the test dataloader, but you can grab val_dataloader or train_dataloader too
        for i, batch in enumerate(self.loader.test_dataloader()):
            print('batch {}'.format(i))
            x, y = batch
            x = x.to(self.device)
            # y = y.to(self.device)

            # save the labels y
            all_labels.append(y.cpu().detach().numpy())

            embeddings, _ = self.backbone(x)
            pred_batch = self.decoder(embeddings)

            # take argmax of the predictions
            pred_batch = torch.argmax(pred_batch, dim=1)
            all_preds.append(pred_batch.cpu().detach().numpy())

        # convert list to tensor
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return all_preds, all_labels


    def load_model(self, cfg, ckpt_path):
        # get the configs
        cfg = yaml.load(open(cfg, 'r'), Loader=yaml.FullLoader)
        train_cfg = cfg['train']  # grab section `train` section of config
        model_cfg = cfg['model']  # grab the `model` section of config

        self.d_output = train_cfg['d_output']  # number of classes the head was trained on

        # the state dict has both the backbone model and the decoder (normally as a Lightning module), but we need to instantiate both separately
        # when not using Lightning.

        # instantiate the model
        backbone = DNAEmbeddingModel(**model_cfg)   # instantiate the backbone separately from the decoder

        # instantiate the decoder
        decoder = SequenceDecoder(model_cfg['d_model'], d_output=self.d_output, l_output=0, mode='pool')  # needs to know the d_model

        state_dict = torch.load(ckpt_path, map_location='cpu')  # has both backbone and decoder
        
        # loads model from ddp by removing prexix to single if necessary
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict["state_dict"], "model."
        )

        model_state_dict = state_dict["state_dict"]

        # need to remove torchmetrics. to remove keys, need to convert to list first
        for key in list(model_state_dict.keys()):
            if "torchmetrics" in key:
                model_state_dict.pop(key)

        # the state_dict keys slightly mismatch from Lightning..., so we fix it here
        decoder_state_dict = {}
        decoder_state_dict['output_transform.weight'] = model_state_dict.pop('decoder.0.output_transform.weight')
        decoder_state_dict['output_transform.bias'] = model_state_dict.pop('decoder.0.output_transform.bias')

        # now actually load the state dict to the decoder and backbone separately
        decoder.load_state_dict(decoder_state_dict, strict=True)
        backbone.load_state_dict(model_state_dict, strict=True)

        # setup tokenizer
        tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],
            model_max_length=self.max_seq_len + 2,  # add 2 since default adds eos/eos tokens, crop later
            add_special_tokens=False,
        )

        return backbone, decoder, tokenizer
        
        
if __name__ == "__main__":
    
    """

    Example cmd for loading a pretrained model (that was finedtuned). This checkpoint was trained on the 'human_nontata_promoters path' dataset.

    # (from safari-internal-inf root, note the -m and no '.py')
    python -m evals.hg38_inference_decoder --config /home/workspace/eric/safari-internal/configs/evals/hg38_decoder.yaml \
    --ckpt_path /home/workspace/eric/safari-internal/outputs/2023-04-14/04-32-17-578382/checkpoints/val/accuracy.ckpt


    # enhancer (genomic benchmark)
    python -m evals.hg38_inference_decoder --config /home/workspace/eric/safari-internal/configs/evals/hg38_decoder.yaml \
    --ckpt_path /home/workspace/eric/safari-internal/outputs/2023-04-12/23-40-51-542457/checkpoints/val/mcc.ckpt --output_path /home/workspace/eric/safari-internal/outputs


    # config is located here:
    configs/evals/hg38_decoder.yaml

    # download the checkpoints from google drive, and put it in the outputs/ dir
    https://drive.google.com/drive/folders/11cDmLZgBHr3KkiCtS2V6sqI3Kf8lTW39?usp=share_link


    # enhancer weights, from nucleotide transformer, binary classification
    /home/workspace/eric/safari-internal/outputs/2023-04-12/23-40-51-542457/checkpoints/val/mcc.ckpt
    https://drive.google.com/drive/folders/1wIijtwlqWwzNe_0d3meAXSk7oYJ2POMC?usp=share_link

    # promoter tata weights
    /home/workspace/eric/safari-internal/outputs/2023-05-01/04-13-05-495708/checkpoints/val/f1_macro.ckpt
    note, this model is larger, 2 layers, d_model=256 (not 128!!), and d_inner=1024
    https://drive.google.com/drive/folders/1tbIUYwScEox4SLFqeZIFp7Z4YvmIN0M3?usp=share_link



    # In general, you need to make sure there config has the same model settings as it was trained on.

    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config",
        default=f"",
    )
    
    parser.add_argument(
        "--ckpt_path",
        default=f"",
        help="Path to model state dict checkpoint"
    )

    parser.add_argument(
        "--output_path",
        default=f"",
        help="Path to where to save npy file"
    )
        
    args = parser.parse_args()
        
    task = HG38Inference(args.config, args.ckpt_path, max_seq_len=1024, use_dataloader=True)

    # sample sequence, can pass a list of seqs (themselves a list of chars)
    # seqs = ["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"]
    
    # if you just have a list of sequences, as strings, you can use this function, returns list
    # preds = task.predict_on_list(seqs)  # return a list of predictions
    # print(preds[0].shape)  # shape is [batch, 2] for binary class prediction


    # OR...

    # or if you rather use the existing dataloader for the enhancer dataset, you can call this instead
    # returns a np array
    preds, labels = task.predict_from_loader()
    # print(preds.shape)  # shape is [batch, 2] for binary class prediction

    # calculate accuracy of preds vs labels
    acc = np.mean(preds.squeeze() == labels.squeeze())

    print("Acc: ", acc)

    breakpoint()

    pred_path = os.path.join(args.output_path, "preds.npy")
    label_path = os.path.join(args.output_path, "labels.npy")

    # save as numpy arr
    preds_np = np.array(preds)
    labels_np = np.array(labels)
    
    with open(pred_path, 'wb') as f:
        np.save(f, preds_np)

    with open(label_path, 'wb') as f:
        np.save(f, labels_np)
        