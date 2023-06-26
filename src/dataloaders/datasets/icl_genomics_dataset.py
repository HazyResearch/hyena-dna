
import torch
from random import random, randint
import numpy as np
from pathlib import Path

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import is_downloaded

"""

In-Context learning version of Genomic Benchmarks Dataset


"""


# helper functions

def exists(val):
    return val is not None

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


class ICLGenomicsDataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        split: str,
        shots: int, 
        max_length: int,
        label_to_token: dict=None,
        dataset_name="human_nontata_promoters",
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=None,
        add_eos=True,  # need this for current ICL setup
        eos_token=None, # end of sequence token (None defaults to tokenizer.sep_token)
        rc_aug=False,
    ):

        self.shots = shots
        self.label_to_token = {0: 'A', 1: 'N'} if label_to_token is None else label_to_token

        self.max_length = max_length
        self.use_padding = use_padding
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.eos_token = eos_token
        self.d_output = d_output  # needed for decoder to grab
        self.rc_aug = rc_aug

        if not is_downloaded(dataset_name, cache_path=dest_path):
            print("downloading {} to {}".format(dataset_name, dest_path))
            download_dataset(dataset_name, version=0, dest_path=dest_path)
        else:
            print("already downloaded {}-{}".format(split, dataset_name))

        # change "val" split to "test".  No val available, just test
        if split == "val":
            split = "test"

        # use Path object
        base_path = Path(dest_path) / dataset_name / split

        self.all_paths = []
        self.all_labels = []
        label_mapper = {}

        for i, x in enumerate(base_path.iterdir()):
            label_mapper[x.stem] = i

        for label_type in label_mapper.keys():
            for x in (base_path / label_type).iterdir():
                self.all_paths.append(x)
                self.all_labels.append(label_mapper[label_type])

        self.unique_labels = label_mapper.values()
        self.n_samples = len(self.all_paths)

    def __len__(self):
        return self.n_samples
    
    def get_sample_from_idx(self, idx):

        txt_path = self.all_paths[idx]

        with open(txt_path, "r") as f:
            content = f.read()
        x = content
        y = self.all_labels[idx]

        # apply rc_aug here if using
        if self.rc_aug and coin_flip():
            x = string_reverse_complement(x)

        seq = self.tokenizer(x,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else None,
            max_length=self.max_length,
            truncation=True,
        )  # add cls and eos token (+2)
        seq = seq["input_ids"]  # get input_ids

        if len(self.label_to_token[y])>1:
            # to get cls token, we can't use the normal self.tokenizer, which will split into separate chars,
            # we need to lookup the vocab dict directly, while using UNK by default if not found
            # use the chr_name as the cls token
            target = [self.tokenizer._vocab_str_to_int.get(self.label_to_token[y], self.tokenizer._vocab_str_to_int["[UNK]"])]
        else:
            target = self.tokenizer(self.label_to_token[y], add_special_tokens=False)['input_ids']

        # need to handle eos here
        eos_token = [self.tokenizer.sep_token_id] if not exists(self.eos_token) else self.tokenizer(self.eos_token, add_special_tokens=False)['input_ids']
        if self.add_eos:
            seq = seq + eos_token
        if self.add_eos:
            target = target + eos_token

        # convert to tensor
        seq = torch.LongTensor(seq)
        target = torch.LongTensor(target)

        return seq, target


    def __getitem__(self, idx):

        test_seq, test_target = self.get_sample_from_idx(idx)
        test_target = test_target[0].unsqueeze(0)
        if self.shots==0:
            return test_seq, test_target

        shot_indices = {}
        for label in self.unique_labels:
            label_indices = np.where(np.array(self.all_labels)==label)[0]
            label_indices = np.array([i for i in label_indices if i!=idx])
            shot_indices[label] = np.random.choice(label_indices, size=self.shots, replace=False)
                
        shots = []
        for shot in range(self.shots):
            for label in shot_indices:
                seq, target = self.get_sample_from_idx(shot_indices[label][shot])
                shots.append(torch.cat([seq, target],dim=0))
        
        # lets shuffle the shots to avoid always having the same order
        np.random.shuffle(shots)

        shots = torch.cat([torch.cat(shots, dim=0), test_seq], dim=0)
        return shots, test_target