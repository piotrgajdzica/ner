#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=64GB
#SBATCH --time=72:00:00
#SBATCH -A lemkingpu
#SBATCH --partition=plgrid-gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gajdzica@agh.edu.pl

# podać nazwę eksperymentu
# name="test-5g-d-0.05-m-p-x-ugc"
name="final-filtered-test-nkjp-flair-word2vec"

module load plgrid/libs/openblas
module load plgrid/libs/atlas/3.10.3
module load plgrid/libs/lapack/3.8.0
module load plgrid/libs/hdf5/1.8.17
module load plgrid/tools/apex

module load plgrid/apps/cuda/9.0
module load plgrid/tools/python/3.6.5

source ~/ner/venv/bin/activate

export LD_LIBRARY_PATH=/net/scratch/people/plgpgajdzica/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/net/scratch/people/plgpgajdzica/cuda/lib64:$LIBRARY_PATH
export CPATH=/net/scratch/people/plgpgajdzica/cuda/include:$CPATH

echo "Is CUDA available?"
python -c "import torch; print(torch.cuda.is_available()); print(torch.backends.cudnn.enabled)"

mkdir -p "~/scratch2/taggers/${name}/"
#time python train_tagger.py "taggers/${name}" ../../data_simplified/ -m -a  -u
time python \
 ../train_tagger_different_embeddings.py taggers/${name} \
 training_datasets/nkjp \
 --base-data-directory /net/people/plgpgajdzica/scratch/ner/data/ \
 --max-epochs 1 \
 --dropout 0.2 \
 --use-space \
 --use-morph \
 --learning-rate 0.0005 \
 --use-lemma \
 --batch-size 128 \
 --forward-path flair/lm-polish-forward-v0.2.pt \
 --embeddings-paths flair-nkjp+wiki-lemmas-all-300-skipg-ns.txt \
 --backward-path flair/lm-polish-backward-v0.2.pt \

