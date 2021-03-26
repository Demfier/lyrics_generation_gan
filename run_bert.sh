#!/usr/bin/bash
#SBATCH --account=rrg-ovechtom
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=bert-base-lyrics-classifier
#SBATCH --output=outputs/%x.out

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/BERT_ENV
source $SLURM_TMPDIR/BERT_ENV/bin/activate
pip install transformers torch sklearn --no-index

python -u bert_classifier/train_bert.py  # replace this with your command

deactivate
