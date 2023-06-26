import torch 

import argparse
import os
import sys
import yaml 
from tqdm import tqdm
import json 

sys.path.append(os.environ.get("SAFARI_PATH", "."))

from src.models.sequence.long_conv_lm import ConvLMHeadModel

# from transformers import AutoTokenizer, GPT2LMHeadModel
# from spacy.lang.en.stop_words import STOP_WORDS
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

try:
    from tokenizers import Tokenizer  
except:
    pass

# https://github.com/openai/gpt-2/issues/131#issuecomment-492786058
# def preprocess(text):
#     text = text.replace("“", '"')
#     text = text.replace("”", '"')
#     return '\n'+text.strip()


class HG38Encoder:
    "Encoder inference for HG38 sequences"
    def __init__(self, model_cfg, ckpt_path, max_seq_len):
        self.max_seq_len = max_seq_len
        self.model, self.tokenizer = self.load_model(model_cfg, ckpt_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def encode(self, seqs):
            
        results = []

        # sample code to loop thru each sample and tokenize first (char level)
        for seq in tqdm(seqs):
            
            if isinstance(self.tokenizer, Tokenizer):
                tokenized_seq = self.tokenizer.encode(seq).ids
            else:
                tokenized_seq = self.tokenizer.encode(seq)
            
            # can accept a batch, shape [B, seq_len, hidden_dim]
            logits, __ = self.model(torch.tensor([tokenized_seq]).to(device=self.device))

            # Using head, so just have logits
            results.append(logits)

        return results
        
            
    def load_model(self, model_cfg, ckpt_path):
        config = yaml.load(open(model_cfg, 'r'), Loader=yaml.FullLoader)
        model = ConvLMHeadModel(**config['model_config'])
        
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

        model.load_state_dict(state_dict["state_dict"])

        # setup tokenizer
        if config['tokenizer_name'] == 'char':
            print("**Using Char-level tokenizer**")

            # add to vocab
            tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],
                model_max_length=self.max_seq_len + 2,  # add 2 since default adds eos/eos tokens, crop later
                add_special_tokens=False,
            )
            print(tokenizer._vocab_str_to_int)
        else:
            raise NotImplementedError("You need to provide a custom tokenizer!")

        return model, tokenizer
        
        
if __name__ == "__main__":
    
    SAFARI_PATH = os.getenv('SAFARI_PATH', '.')

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_cfg",
        default=f"{SAFARI_PATH}/configs/evals/hyena_small_150b.yaml",
    )
    
    parser.add_argument(
        "--ckpt_path",
        default=f"",
        help="Path to model state dict checkpoint"
    )
        
    args = parser.parse_args()
        
    task = HG38Encoder(args.model_cfg, args.ckpt_path, max_seq_len=1024)

    # sample sequence, can pass a list of seqs (themselves a list of chars)
    seqs = ["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"]
    
    logits = task.encode(seqs)
    print(logits)
    print(logits[0].logits.shape)

    breakpoint()

    