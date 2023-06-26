import os
from pathlib import Path
from pyfaidx import Fasta
import torch
import shutil
import gzip
import random
from typing import Optional, Union, Dict, List
from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer
import collections

"""
Dataset that randomly samples sequences of length (X) from a species' whole genome.

Given a specific species, it will...
    1. Randomly sample a chromosome from that species
    2. Randomly sample a sequence of length X from that chromosome

All sampled sequences will be the same size.
If a sequence is truncated by the end of a chromosome, it will be padded with 'N'

Char sequences (not one hots yet)

No augmentations yet.

"""

# Determine chromosomes to use for train/test split
SPECIES_CHROMOSOME_SPLITS = {
    'human' : {
        'train' : [ '2', '4', '6', '8','14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', ],
        'valid' : ['1', '3', '12', '13',],
        'test' : [ '5', '7', '9', '10', '11',],
    },
    'lemur' : {
        'train' : [ '2', '4', '6', '8','14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', 'X', 'Y', ],
        'valid' : ['1', '3', '12', '13',],
        'test' : [ '5', '7', '9', '10', '11',],
    },
    'goat' : {
        'train' : [ '2', '4', '6', '8','14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 'X', 'Y', ],
        'valid' : ['1', '3', '12', '13',],
        'test' : [ '5', '7', '9', '10', '11',],
    },
    'sheep' : {
        'train' : [ '2', '4', '6', '8','14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', 'X', 'Y', ],
        'valid' : ['1', '3', '12', '13',],
        'test' : [ '5', '7', '9', '10', '11',],
    },
    'pig' : {
        'train' : [ '2', '4', '6', '8','14', '15', '16', '17', '18', 'X', 'Y', ],
        'valid' : ['1', '3', '12', '13',],
        'test' : [ '5', '7', '9', '10', '11',],
    },
    'mouse' : {
        'train' : [ '2', '4', '6', '8', '14', '15', '16', '17', '18', '19', 'X', ],
        'valid' : ['1', '3', '12', '13',],
        'test' : [ '5', '7', '9', '10', '11',],
    },
    'gorilla' : {
        'train' : [ '2A', '2B', '4', '6', '8', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', ],
        'valid' : ['1', '3', '12', '13',],
        'test' : [ '5', '7', '9', '10', '11',],
    },
    'orangutan' : {
        'train' : [ '2A', '2B', '4', '6', '8', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', ],
        'valid' : ['1', '3', '12', '13',],
        'test' : [ '5', '7', '9', '10', '11',],
    },
    'chimpanzee' : {
        'train' : [ '2A', '2B', '4', '6', '8', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', ],
        'valid' : ['1', '3', '12', '13',],
        'test' : [ '5', '7', '9', '10', '11',],
    },
    'hippo' : {
        'train' : [ '2', '4', '6', '8', '14', '15', '16', '17', 'X', ],
        'valid' : ['1', '3', '12', '13',],
        'test' : [ '5', '7', '9', '10', '11',],
    }
}

class SpeciesDataset(torch.utils.data.Dataset):

    '''
    Loop thru fasta files (separated by chromosome) and return a sequence of length `max_length` from a random chromosome.
    '''

    def __init__(
        self,
        species: list,
        species_dir: str,
        split: str,
        max_length,
        total_size,
        pad_max_length=None,
        tokenizer=None,
        tokenizer_name=None,
        add_eos=False,
        rc_aug=False,
        return_augs=False,
        chromosome_weights: Optional[Union[Dict[str, List[float]], str]]='uniform',
        species_weights: Optional[Union[List[float], str]]='uniform',
        task='species_classification|next_token_pred',
        remove_tail_ends=False,
        cutoff_train=0.1,
        cutoff_test=0.2,
    ):
        """
        `chromosome_weights` => can be either...
            - String of form 'uniform|weighted_by_bp', in which case every species' chromosomes will be sampled accordingly
            - Dict of form {species: [chromosome weight1, chromosome weight 2, ...]
            
        `species_weights` => can be either...
            - String of form 'uniform|weighted_by_bp'
            - List of form [ species weight1, species weight2, ... ]
        """
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.return_augs = return_augs
        self.add_eos = add_eos
        self.species = species
        self.species_dir = species_dir
        self.split = split
        self.total_size = total_size
        self.task = task
        self.d_output = len(self.species) if task == 'species_classification' else None
        is_show_log: bool = False
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test
        
        if task == 'species_classification' and self.d_output < 2:
            print(f'Note that `d_output` should be >= 2 for task `{task}`, otherwise you are only predicting one class. Got {self.d_output}')

        # Store FASTAs for each species
        self.fastas: Dict[str, Dict[str, Fasta]] = collections.defaultdict(dict) # [key] = species -> dict where [key] = chromosome, [value] = Fasta object
        self.chromosomes: Dict[str, List[str]] = {} # [key] = species, [value] = list of chromosomes in this split
        self.chromosome_weights: Dict[str, List[float]] = {} # [key] = species, [value] = list where [idx] = self.chromosomes[species][idx], [value] = weight
        self.species_weights: List[float] = [] # [idx] = self.species[idx], [value] = weight

        # For every species in `self.species`, load all chromosomes belonging to `split`
        for spec in self.species:
            species_path = Path(self.species_dir) / spec
            assert species_path.exists(), f'The path `{species_path}` does not exist for species `{spec}`. Please point to a valid directory containing your species fna.gz files.'

            # Select chromosomes for this split
            assert spec in SPECIES_CHROMOSOME_SPLITS, f'Unrecognized species `{spec}`. Valid species are: {list(SPECIES_CHROMOSOME_SPLITS.keys())}.'
            self.chromosomes[spec] = SPECIES_CHROMOSOME_SPLITS[spec][split]

            # Load all .fna files of chromosomes in this split
            for chromosome in self.chromosomes[spec]:
                # Unzip if necessary
                gz_file_path = os.path.join(species_path, f'chr{chromosome}.fna.gz')
                if os.path.exists(gz_file_path) and not (
                    os.path.exists(os.path.join(species_path, f'chr{chromosome}.fna')) or
                    os.path.exists(os.path.join(species_path, f'chr{chromosome}.fa'))
                ):
                    if is_show_log:
                        print(f"Unzipping {gz_file_path}...")
                    with gzip.open(gz_file_path, 'rb') as f_in:
                        with open(os.path.join(species_path, f'chr{chromosome}.fna'), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                # Read .fna or .fa file, whichever we can find
                file_paths = [ os.path.join(species_path, x) for x in [ f'chr{chromosome}.fna', f'chr{chromosome}.fa' ] ]
                is_file_found: bool = False
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        if chromosome not in self.fastas[spec]:
                            self.fastas[spec][chromosome] = Fasta(file_path, sequence_always_upper=True)
                        is_file_found = True
                if not is_file_found:
                    raise FileNotFoundError(f'Could not find any of these files: `{file_paths}`. Please point to a valid directory containing all .fna files for species `{spec}`.\nExpected chromosomes: {self.chromosomes[spec]}.')

            if is_show_log:
                print(f"Species: {spec}")
                print(f"Split: {split}")
                print(f"Chromosomes: {self.chromosomes[spec]}")
                print(f"Loaded {len(self.fastas[spec])} FASTA files from {species_path}: {list(self.fastas[spec].keys())}")

        # Set chromosome weights for sampling
        if isinstance(chromosome_weights, dict):
            assert len(chromosome_weights) == len(self.species), f"`chromosome_weights` must have a weight for each species. Expected {len(self.species)} weights, instead got {len(chromosome_weights)}."
            self.chromosome_weights = chromosome_weights
        elif chromosome_weights == 'uniform':
            self.chromosome_weights = {
                spec: 'uniform'
                for spec in self.species
            }
        elif chromosome_weights == 'weighted_by_bp':
            self.chromosome_weights = {
                spec: 'weighted_by_bp'
                for spec in self.species
            }
        else:
            raise ValueError(f"Invalid chromosome_weights: {chromosome_weights}. Must be 'uniform', 'weighted_by_bp', or a dict of species -> chromosome weights.")
        
        for spec, strategy_or_weights in self.chromosome_weights.items():
            if isinstance(strategy_or_weights, str):
                if strategy_or_weights == 'uniform':
                    # Uniform weights
                    self.chromosome_weights[spec] = [1] * len(self.chromosomes[spec])
                elif strategy_or_weights == 'weighted_by_bp':
                    # Weight by number of base pairs in each chromosome
                    self.chromosome_weights[spec] = [
                        len(self.fastas[spec][chromosome])
                        for chromosome in self.chromosomes[spec]
                    ]
                    self.chromosome_weights[spec] = [w / sum(self.chromosome_weights[spec]) for w in self.chromosome_weights[spec]]
                else:
                    raise ValueError(f"Invalid chromosome_weights strategy: {strategy_or_weights}. Must be 'uniform' or 'weighted_by_bp'.")
            elif isinstance(strategy_or_weights, list):
                # Check that all chromosomes are accounted for
                assert set(strategy_or_weights.keys()) == set(self.chromosomes[spec]), f"`chromosome_weights` must have a weight for each chromosome. Expected {self.chromosomes[spec]}, instead got {strategy_or_weights.keys()}."
                self.chromosome_weights[spec] = strategy_or_weights
            else:
                raise ValueError(f"Invalid chromosome_weights: {chromosome_weights}. Must be 'uniform', 'weighted_by_bp', or a dict of species -> chromosome weights.")
            
        # Set species weights for sampling
        if isinstance(species_weights, list):
            assert len(species_weights) == len(self.species), f"`species_weights` must have a weight for each species. Expected {len(self.species)} weights, instead got {len(species_weights)}."
            self.species_weights = species_weights
        elif species_weights == 'uniform':
            # Uniform weights
            self.species_weights = [1] * len(self.species)
        elif species_weights == 'weighted_by_bp':
            # Weight by number of base pairs in each chromosome
            self.species_weights = [
                sum([ 
                    len(fasta) 
                    for fasta in self.fastas[spec].values() 
                ])
                for spec in self.species
            ]
            self.species_weights = [w / sum(self.species_weights) for w in self.species_weights]
        else:
            raise ValueError(f"Invalid species_weights: {species_weights}. Must be 'uniform', 'weighted_by_bp', or a dict of species -> chromosome weights.")
    
        if is_show_log:
            print(f"Species weights: {list(zip(self.species, self.species_weights))}")
            print(f"Chromosome weights: {self.chromosome_weights}")

    def __len__(self):
        assert self.total_size is not None, "Must set the `total_size` kwarg when you initialize `SpeciesDataset` before calling `__len__`."
        return self.total_size

    def __getitem__(self, idx):
        """Returns a sequence of length `max_length` from a random chromosome of a random species."""
        is_show_log: bool = False
        # sample a random species (according to weighting)
        # rand = random.Random() # maps idx -> random seed, without affecting global random state
        # rand.seed(idx)
        spec: str = random.choices(self.species, weights=self.species_weights, k=1)[0]

        # sample a random chromosome (according to weighting)
        # rand = random.Random() # maps idx -> random seed, without affecting global random state
        # rand.seed(idx + 1)
        chromosome = random.choices(self.chromosomes[spec], weights=self.chromosome_weights[spec], k=1)[0]

        # sample a random sequence of length `self.max_length` from this chromosome
        # print("****", spec, chromosome, self.fastas[spec].keys(), idx)
        fasta = self.fastas[spec][chromosome][0] # idx into 0 b/c only one fasta per chromosome
        chromosome_length: int = len(fasta)
        # rand = random.Random() # maps idx -> random seed, without affecting global random state
        # rand.seed(idx + 2)

        if self.remove_tail_ends:
            if self.split == 'train':
                cutoff = self.cutoff_train
            else:
                cutoff = self.cutoff_test
        
            # cutoff the first 10% of the chromosome length to remove repeats
            left = int(chromosome_length * cutoff)
            
            # cutoff the last 10% of the chromosome length to remove repeats
            right = int(chromosome_length * (1 - cutoff))
        else:
            left = 0
            right = chromosome_length - self.max_length

        start: int = random.randint(left, right)
        end: int = start + self.max_length
        seq = str(fasta[start:min(end, right)])
        
        # pad with Ns if necessary
        seq = seq.rjust(end - start, "N")
        assert len(seq) == self.max_length, f'Length of sequence ({len(seq)}) from interval ({start}, {end}) of chromosome {chromosome} (len={chromosome_length}) is not equal to `self.max_length` ({self.max_length})'
        
        if is_show_log:
            print(f"Sampled species: {spec}")
            print(f"Sampled chromosome: {chromosome}")
            print(f"Sampled sequence ({start}, {end}) of len={len(seq)}: {seq[:10]}...{seq[-10:]}")
        
        assert self.tokenizer is not None, f"Tokenizer cannot be `None`."
        if self.tokenizer_name == 'char':
            seq = self.tokenizer(seq, add_special_tokens=False)  # add cls and eos token (+2)
            seq = seq["input_ids"]  # get input_ids
            # need to handle eos here
            if self.add_eos:
                # append list seems to be faster than append tensor
                seq.append(self.tokenizer.sep_token_id)
        elif self.tokenizer_name == 'bpe':
            seq = self.tokenizer(seq, 
                padding="max_length",
                max_length=self.pad_max_length,
                truncation=True,
            )  # add cls and eos token (+2)
            # get input_ids
            if self.add_eos:
                seq = seq["input_ids"][1:]  # remove the bos, keep the eos token
            else:
                seq = seq["input_ids"][1:-1]  # remove both special tokens
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        data = seq[:-1].clone()  # remove eos
        if self.task == 'next_token_pred':
            target = seq[1:].clone()  # offset by 1, includes eos
        elif self.task == 'species_classification':
            target = self.species.index(spec)
        else:
            raise ValueError(f"Invalid task: {self.task}")

        if is_show_log:
            print(f"Sampled tokens of len={len(seq)}: {seq[:10]}...{seq[-10:]}")
            print(f"Sampled target: {target}")
        
        return data, target
