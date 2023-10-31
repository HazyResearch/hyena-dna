from pyfaidx import Fasta
import torch
from random import random 
from pathlib import Path

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer

def coin_flip():
    return random() > 0.5

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


class NucleotideTransformerDataset(torch.utils.data.Dataset):

    '''
    Loop thru fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        split,
        max_length,
        dataset_name=None,
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        return_mask=False,
    ):

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug
        self.return_mask = return_mask

        # change "val" split to "test".  No val available, just test
        if split == "val":
            split = "test"

        # use Path object
        base_path = Path(dest_path) / dataset_name 
        assert base_path.exists(), 'path to fasta file must exist'

        for file in (base_path.iterdir()):
            if str(file).endswith('.fasta') and split in str(file):
                self.seqs = Fasta(str(file), read_long_names=True)    

        self.label_mapper = {}
        for i, key in enumerate(self.seqs.keys()):
            self.label_mapper[i] = (key, int(key.rstrip()[-1]))


    def __len__(self):
        return len(self.seqs.keys())

    def __getitem__(self, idx):
        seq_id = self.label_mapper[idx][0]
        x = self.seqs[seq_id][:].seq # only one sequence
        y = self.label_mapper[idx][1] # 0 or 1 for binary classification

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length" if self.use_padding else 'do_not_pad',
            max_length=self.max_length,
            truncation=True,
        )
        seq_ids = seq["input_ids"]  # get input_ids
        seq_ids = torch.LongTensor(seq_ids)

        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        # need to wrap in list
        target = torch.LongTensor([y])  # offset by 1, includes eos

        if self.return_mask:
            return seq_ids, target, {'mask': torch.BoolTensor(seq['attention_mask'])}
        else:
            return seq_ids, target
