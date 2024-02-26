
for d in mamba_checkpoints/artifacts/*/; do
	if [[ "$d" == *"model-b5bc3mol"* ]]; then
		g="Cactus 7M 64Kibp"
		m="model.config.d_model=256 model.config.n_layer=16 dataset.batch_size=4"
	else
		if [[ "$d" == *"model-l968ycve"* ]]; then
			g="Cactus 1.4M 64Kibp"
			m="dataset.batch_size=8"
		else
			g="HG38 1.4M 64Kibp"
			m="dataset.batch_size=8"
		fi
	fi
	echo "bash evals/all_benchmarks.sh \"wandb.group=$g\" trainer.devices=8 dataset.dest_path=/home/ubuntu/HyenaDNA_collab/hyena-dna-repo/data/mixed/ train.pretrained_model_path=${d}model.ckpt ++dataset.ref_genome_path=/home/ubuntu/HyenaDNA_collab/hyena-dna-repo/data/hg38/hg38.ml.fa ++dataset.ref_genome_version=hg38 ++dataset.data_path=/home/ubuntu/HyenaDNA_collab/hyena-dna-repo/data/chromatin_profile/dataset/ ++dataset.fasta=/home/ubuntu/HyenaDNA_collab/hyena-dna-repo/data/hg38/hg38.ml.fa ++dataset.bed=/home/ubuntu/HyenaDNA_collab/hyena-dna-repo/data/hg38/human-sequences.bed $m"
done
