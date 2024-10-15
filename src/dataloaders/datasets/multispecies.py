import os
from pathlib import Path
from pyfaidx import Fasta
import torch
import random
from typing import Optional, Union, Dict, List
import collections

SPECIES_CHROMOSOME_SPLITS = {
    'human': {
        'train': [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'],  # Human chromosomes
        'valid': ['chr1', 'chr2'],  # Validation set (examples)
        'test': ['chr3', 'chr4'],   # Test set (examples)
    },
    'mouse': {
        'train': [f'chr{i}' for i in range(1, 20)] + ['chrX', 'chrY'],  # Mouse chromosomes
        'valid': ['chr1', 'chr2'],  # Validation set (examples)
        'test': ['chr3', 'chr4'],   # Test set (examples)
    },
    'lemur': {
        'train': [f'scaffold{i}' for i in range(1, 51)],  # Lemur scaffolds (example)
        'valid': [f'scaffold{i}' for i in range(51, 61)],  # Validation set
        'test': [f'scaffold{i}' for i in range(61, 74)],   # Test set
    },
    'squirrel': {
        'train': [f'scaffold{i}' for i in range(1, 12)],  # Squirrel scaffolds (example)
        'valid': [f'scaffold{i}' for i in range(12, 15)],  # Validation set
        'test': [f'scaffold{i}' for i in range(15, 18)],   # Test set
    }
}

class SpeciesDataset(torch.utils.data.Dataset):
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
        self.remove_tail_ends = remove_tail_ends
        self.cutoff_train = cutoff_train
        self.cutoff_test = cutoff_test
        
        if task == 'species_classification' and self.d_output < 2:
            print(f'Note that `d_output` should be >= 2 for task `{task}`, otherwise you are only predicting one class. Got {self.d_output}')

        # Store FASTAs for each species
        self.fastas: Dict[str, Fasta] = {}
        self.chromosomes: Dict[str, List[str]] = {}

        for spec in self.species:
            species_path = Path(self.species_dir) / spec / f'{spec}_genome.fna'
            assert species_path.exists(), f'The file `{species_path}` does not exist for species `{spec}`.'

            # Load the genome file for the species
            self.fastas[spec] = Fasta(species_path, sequence_always_upper=True)
            self.chromosomes[spec] = SPECIES_CHROMOSOME_SPLITS[spec][split]

        # Handle chromosome/scaffold weighting
        self.chromosome_weights = self._initialize_weights(chromosome_weights, self.chromosomes)
        self.species_weights = self._initialize_species_weights(species_weights)

    def _initialize_weights(self, chromosome_weights, chromosomes):
        weights = {}
        if isinstance(chromosome_weights, dict):
            weights = chromosome_weights
        elif chromosome_weights == 'uniform':
            weights = {spec: [1] * len(chromosomes[spec]) for spec in chromosomes}
        elif chromosome_weights == 'weighted_by_bp':
            weights = {
                spec: [len(self.fastas[spec][chromosome]) for chromosome in chromosomes[spec]]
                for spec in chromosomes
            }
            for spec in weights:
                total = sum(weights[spec])
                weights[spec] = [w / total for w in weights[spec]]
        return weights

    def _initialize_species_weights(self, species_weights):
        if isinstance(species_weights, list):
            assert len(species_weights) == len(self.species), f"`species_weights` must have a weight for each species."
            return species_weights
        elif species_weights == 'uniform':
            return [1] * len(self.species)
        elif species_weights == 'weighted_by_bp':
            return [
                sum(len(self.fastas[spec][chromosome]) for chromosome in self.chromosomes[spec])
                for spec in self.species
            ]
        return species_weights

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        spec = random.choices(self.species, weights=self.species_weights, k=1)[0]
        chromosome = random.choices(self.chromosomes[spec], weights=self.chromosome_weights[spec], k=1)[0]

        fasta = self.fastas[spec][chromosome]
        chromosome_length = len(fasta)

        if self.remove_tail_ends:
            left = int(chromosome_length * self.cutoff_train if self.split == 'train' else self.cutoff_test)
            right = int(chromosome_length * (1 - (self.cutoff_train if self.split == 'train' else self.cutoff_test)))
        else:
            left, right = 0, chromosome_length - self.max_length

        start = random.randint(left, right)
        end = start + self.max_length
        seq = str(fasta[start:end])

        seq = seq.ljust(self.max_length, 'N')
        if self.tokenizer_name == 'char':
            seq = self.tokenizer(seq, add_special_tokens=False)["input_ids"]
            if self.add_eos:
                seq.append(self.tokenizer.sep_token_id)
        else:
            raise ValueError(f"Invalid tokenizer name: {self.tokenizer_name}")

        data = torch.LongTensor(seq[:-1])
        if self.task == 'next_token_pred':
            target = torch.LongTensor(seq[1:])
        elif self.task == 'species_classification':
            target = self.species.index(spec)
        else:
            raise ValueError(f"Invalid task: {self.task}")

        return data, target