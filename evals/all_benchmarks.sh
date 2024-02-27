set -e


python -m train wandb.job_type=eval experiment=hg38/genomic_benchmark_mamba dataset.dataset_name=demo_coding_vs_intergenomic_seqs "$@"
python -m train wandb.job_type=eval experiment=hg38/genomic_benchmark_mamba dataset.dataset_name=demo_human_or_worm "$@"
python -m train wandb.job_type=eval experiment=hg38/genomic_benchmark_mamba dataset.dataset_name=dummy_mouse_enhancers_ensembl "$@"
python -m train wandb.job_type=eval experiment=hg38/genomic_benchmark_mamba dataset.dataset_name=human_enhancers_cohn "$@"
python -m train wandb.job_type=eval experiment=hg38/genomic_benchmark_mamba dataset.dataset_name=human_enhancers_ensembl "$@"
python -m train wandb.job_type=eval experiment=hg38/genomic_benchmark_mamba dataset.dataset_name=human_ensembl_regulatory "$@"
python -m train wandb.job_type=eval experiment=hg38/genomic_benchmark_mamba dataset.dataset_name=human_nontata_promoters "$@"
python -m train wandb.job_type=eval experiment=hg38/genomic_benchmark_mamba dataset.dataset_name=human_ocr_ensembl "$@"

python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=H3 "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=H3K4me1 "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=H3K4me2 "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=H3K4me3 "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=H3K9ac "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=H3K14ac "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=H3K36me3 "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=H3K79me3 "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=H4 "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=H4ac "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=promoter_all "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=promoter_non_tata "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=promoter_tata "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=splice_sites_acceptor "$@"
python -m train wandb.job_type=eval experiment=hg38/nucleotide_transformer_mamba  dataset.dataset_name=splice_sites_donor "$@"

# python -m train wandb.job_type=eval experiment=hg38/chromatin_profile_mamba "$@"
