set -e
for d in /home/ubuntu/HyenaDNA_collab/hyena-dna-repo/mamba_checkpoints/artifacts/*/; do
	if [[ "$d" == *"model-b5bc3mol"* ]]; then
		g="Cactus 7M 64Kibp"
		m="model.config.d_model=256 model.config.n_layer=16"
		b=4
	else
		b=8
		if [[ "$d" == *"model-l968ycve"* ]]; then
			g="Cactus 1.4M 64Kibp"
			m=""
		else
			g="HG38 1.4M 64Kibp"
			m=""
		fi
	fi
	#python -m train wandb.job_type=eval experiment=hg38/gvl_hg38_mamba trainer.limit_train_batches=1 optimizer.lr=1e-10 trainer.max_epochs=1 "wandb.group=$g" trainer.devices=8 "train.pretrained_model_path=${d}model.ckpt" ++dataset.fasta=/home/ubuntu/HyenaDNA_collab/hyena-dna-repo/data/hg38/hg38.ml.fa ++dataset.bed=/home/ubuntu/HyenaDNA_collab/hyena-dna-repo/data/hg38/human-sequences.bed $m "dataset.batch_size=$((b/4))"

	bash evals/all_benchmarks.sh "wandb.group=$g" trainer.devices=8 ++dataset.dest_path=/home/ubuntu/HyenaDNA_collab/hyena-dna-repo/data/mixed/ "train.pretrained_model_path=${d}model.ckpt" ++dataset.ref_genome_path=/home/ubuntu/HyenaDNA_collab/hyena-dna-repo/data/hg38/hg38.ml.fa ++dataset.ref_genome_version=hg38 ++dataset.data_path=/home/ubuntu/HyenaDNA_collab/hyena-dna-repo/data/chromatin_profile/dataset/ $m "dataset.batch_size=$b"
done
