from itertools import islice
from functools import partial
# import tensorflow as tf
import os
import functools
import json
from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random, randint
import numpy as np
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer


"""
Modifying the hg38 pretraining dataset to include the chromosome token as a class token at the end. This
will help introduce the concept of class appending for ICL in the down stream.

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


class FastaInterval():
    def __init__(
        self,
        *,
        fasta_file,
        max_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file), sequence_always_upper=True)
        self.return_seq_indices = return_seq_indices
        self.max_length = max_length # -1 for adding sos or eos token
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}

        for chr_name in self.seqs.keys():
            self.chr_lens[chr_name] = len(self.seqs[chr_name])


    def __call__(self, chr_name, start, end, return_augs = False):
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = self.chr_lens[chr_name]

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        # checks if not enough sequence to fill up the start to end
        if exists(self.max_length) and interval_length < self.max_length:
            extra_seq = self.max_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        # Added support!  need to allow shorter seqs
        if interval_length > self.max_length:
            end = start + self.max_length

        seq = str(chromosome[start:end])

        if self.rc_aug and coin_flip():
            seq = string_reverse_complement(seq)

        seq = ('.' * left_padding) + seq + ('.' * right_padding)

        return seq

class ICL_HG38Dataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        split,
        bed_file,
        fasta_file,
        max_length,
        min_length=None,
        variable_length=False,  # if you want a var length between min and max length, else len = max_length always
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        return_seq_indices=False,
        shift_augs=None,
        rc_aug=False,
        return_augs=False
    ):

        self.min_length = min_length if min_length is not None else 0.25 * max_length
        self.max_length = max_length
        self.variable_length = variable_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos

        bed_path = Path(bed_file)
        assert bed_path.exists(), 'path to .bed file must exist'

        # read bed file
        df_raw = pd.read_csv(str(bed_path), sep = '\t', names=['chr_name', 'start', 'end', 'split'])
        # select only split df
        self.df = df_raw[df_raw['split'] == split]

        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            max_length = max_length,
            return_seq_indices = return_seq_indices,
            shift_augs = shift_augs,
            rc_aug = rc_aug,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # sample a random row from df
        row = self.df.iloc[idx]
        # row = (chr, start, end, split)
        chr_name, start, end = (row[0], row[1], row[2])

        seq = self.fasta(chr_name, start, end, return_augs=self.return_augs)

        if self.variable_length:
            # sample a random len between min and max
            seq_len = randint(self.min_length, self.max_length)
            seq = seq[:seq_len]

        if self.variable_length:
            seq = self.tokenizer(seq, 
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=False,
            )
        else:
            # fixed size each time
            seq = self.tokenizer(seq,
                add_special_tokens=False,
                max_length=self.pad_max_length
            )
        
        seq = seq["input_ids"]  # get input_ids
        sep_token = self.tokenizer.sep_token_id

        # to get cls token, we can't use the normal self.tokenizer, which will split into separate chars,
        # we need to lookup the vocab dict directly, while using UNK by default if not found
        # use the chr_name as the cls token
        cls_token = self.tokenizer._vocab_str_to_int.get(chr_name, self.tokenizer._vocab_str_to_int["[UNK]"])

        # build token ICL sample structure
        # x = seq[1:] + sep + cls
        # remove 1 from left side (pad side) so that we can add an extra sep_token between, and still have max_length seq
        # need to wrap single tokens in a list to be able to add this way
        seq_sample = seq[1:] + [sep_token] + [cls_token]
        
        # convert to tensor
        seq_sample = torch.LongTensor(seq_sample)

        data = seq_sample[:-1].clone()  # remove cls token in data, (or input x)
        target = seq_sample[1:].clone()  # offset by 1, includes cls token

        return data, target