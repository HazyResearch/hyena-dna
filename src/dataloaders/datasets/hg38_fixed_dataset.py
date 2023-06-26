from pathlib import Path
from pyfaidx import Fasta
import torch


"""
Just a fixed length dataset for 2 test chromosomes, to ensure the test set is the same.

"""


# helper functions
def exists(val):
    return val is not None

class HG38FixedDataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    Returns a generator that retrieves the sequence.
    '''

    def __init__(
        self,
        fasta_file,
        chr_ranges,  # a dict of chr: (start, end) to use for test set
        max_length,
        pad_max_length=None,
        tokenizer=None,
        add_eos=False,
        rc_aug=False,  # not yet implemented
    ):

        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer = tokenizer
        self.add_eos = add_eos


        # create a list of intervals from chr_ranges, from start to end of size max_length
        self.intervals = self.create_fixed_intervals(chr_ranges, self.max_length)
        
        # open fasta file
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file), sequence_always_upper=True)


    def create_fixed_intervals(self, chr_ranges, max_length):
        """
        This will create a new df with non-overlapping sequences of max length, which ensures that the test set is the same every epoch.

        It loops thru the each chr and its start / end range, and creates a sample of max length.

        """

        print("creating new test set with fixed intervals of max_length...")
    
        intervals = []

        # loop thru each chr in chr_ranges, and create intervals of max_length from start to end
        for chr_name, (start, end) in chr_ranges.items():

            # create a list of intervals from start to end of size max_length
            for i in range(start, end, max_length):
                interval_end = min(i + max_length, end)
                intervals.append((chr_name, i, interval_end))
                
        return intervals

    def __len__(self):
        return len(self.intervals)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        row = self.intervals[idx]
        chr_name, start, end = (row[0], row[1], row[2])
        seq = str(self.seqs[chr_name][start:end])

        seq = self.tokenizer(seq,
            padding="max_length",
            max_length=self.pad_max_length,
            truncation=True,
            add_special_tokens=False)  # add cls and eos token (+2)

        seq = seq["input_ids"]  # get input_ids

        # need to handle eos here
        if self.add_eos:
            # # remove first token
            # seq = seq[1:]
            # append list seems to be faster than append tensor
            seq.append(self.tokenizer.sep_token_id)
        
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        # replace N token with a pad token, so we can ignore it in the loss
        seq = self.replace_value(seq, 11, self.tokenizer.pad_token_id)

        data = seq[:-1].clone()  # remove eos
        target = seq[1:].clone()  # offset by 1, includes eos

        return data, target