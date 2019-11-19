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
name="test1"

module load plgrid/libs/openblas
module load plgrid/libs/atlas/3.10.3
module load plgrid/libs/lapack/3.8.0
module load plgrid/libs/hdf5/1.8.17

module load plgrid/apps/cuda/9.0
module load plgrid/tools/python/3.6.5

source ~/ner/venv/bin/activate

export LD_LIBRARY_PATH=/net/scratch/people/plgpgajdzica/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/net/scratch/people/plgpgajdzica/cuda/lib64:$LIBRARY_PATH
export CPATH=/net/scratch/people/plgpgajdzica/cuda/include:$CPATH

echo "Is CUDA available?"
python -c "import torch; print(torch.cuda.is_available()); print(torch.backends.cudnn.enabled)"

mkdir -p "~/scratch/ner/data/tagger/${name}/"
#time python train_tagger.py "taggers/${name}" ../../data_simplified/ -m -a  -u
time python \
 train_tagger.py tagger/${name} \
 tokens-with-entities-tags-and-classes \
 --base-data-directory /net/people/plgpgajdzica/scratch/ner/data/ \
 --prepare_dataset \
 --max_epochs 100 \
 --dropout 0.05 \
 --use-space \
 --use-morph \
 --use-lemma \
 --article-limit 100000
 --forward-path wiki+nkjp-small-f.pt \
 --backward-path wiki+nkjp-small-b.pt \
 /
 #-g -p   #-s 0.3 -m -x #-g #-m -g -i
